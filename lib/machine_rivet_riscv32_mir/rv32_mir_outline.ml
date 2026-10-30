(* Regions of a function that can run from a code overlay. A region is a run of
   consecutive blocks entered only at its first block, so the function jumps to
   it and it jumps back out: no call, and so no return address to keep. What
   the overlay manager clobbers (t0-t6) is saved below the stack pointer, which
   nothing else uses, by the jump into the region and restored by the overlay's
   first instructions. *)

open Machine_ir
module A = Machine_rivet_riscv32.Rv32_asm
module F = Rv32_mir_form
module Item = F.Item
module M = Rv32_mir_module
module Op = A.E.Operand

type func =
  ( Machine_target_riscv32.Rv32_op.t,
    Machine_target_riscv32.Rv32_op.test )
  Mir_phys.Func.t

(* The registers the manager's stubs and loader clobber, saved by the jump into
   an overlay, from just below the stack pointer. *)
let saved = [ "t0"; "t1"; "t2"; "t3"; "t4"; "t5"; "t6" ]

(* [(label, bytes)] of each block of the function in layout order, the items of
   each, and the index of each block by id. *)
type layout = {
  func : func;
  head : Item.t list;  (** before the first block: the jump to the entry *)
  labels : string array;
  runs : Item.t list array;
  index : (int, int) Hashtbl.t;
}

let layout (f : func) items =
  let blocks = Array.of_list f.Mir_phys.Func.blocks in
  let labels =
    Array.map (fun (b : (_, _) Mir_phys.Block.t) -> M.label_of f b.id) blocks
  in
  let index = Hashtbl.create 64 in
  Array.iteri
    (fun k (b : (_, _) Mir_phys.Block.t) ->
      Hashtbl.replace index (Mir_id.Block.to_int b.id) k)
    blocks;
  let runs = Array.make (Array.length blocks) [] in
  let current = ref (-1) and head = ref [] in
  let by_label = Hashtbl.create 64 in
  Array.iteri (fun k l -> Hashtbl.replace by_label l k) labels;
  List.iter
    (fun it ->
      (match it with
      | Item.Label l -> (
          match Hashtbl.find_opt by_label l with
          | Some k -> current := k
          | None -> ())
      | _ -> ());
      if !current >= 0 then runs.(!current) <- it :: runs.(!current)
      else head := it :: !head)
    items;
  Array.iteri (fun k r -> runs.(k) <- List.rev r) runs;
  { func = f; head = List.rev !head; labels; runs; index }

let successors t k =
  let b = List.nth t.func.Mir_phys.Func.blocks k in
  List.map
    (fun id -> Hashtbl.find t.index (Mir_id.Block.to_int id))
    (Mir_phys.Term.successors b.Mir_phys.Block.terminator)

let target = function
  | Item.Branch { label; _ } -> Some label
  | Item.Insn { mnemonic = "j"; ops = [ Op.Sym (Asm_core.Expr.Symbol l) ] } ->
      Some l
  | _ -> None

let bytes items = List.fold_left (fun n it -> n + Item.bytes it) 0 items

(* [depth]: how many loops the region's entry is inside, taking a branch back
   to an earlier block in layout order as a loop over the blocks between. *)
type candidate = { first : int; last : int; size : int; depth : int }

(* The labels a run defines, and those its branches and jumps name. *)
let defined run =
  List.filter_map (function Item.Label l -> Some l | _ -> None) run

let candidates t =
  let n = Array.length t.labels in
  let entry =
    Hashtbl.find t.index (Mir_id.Block.to_int t.func.Mir_phys.Func.entry)
  in
  let succ = Array.init n (successors t) in
  let is_block_label =
    let h = Hashtbl.create 64 in
    Array.iter (fun l -> Hashtbl.replace h l ()) t.labels;
    Hashtbl.mem h
  in
  let depth k =
    let d = ref 0 in
    Array.iteri
      (fun a ss ->
        List.iter (fun b -> if b <= a && b <= k && k <= a then incr d) ss)
      succ;
    !d
  in
  let found = ref [] in
  for first = 0 to n - 1 do
    for last = first to n - 1 do
      let inside k = k >= first && k <= last in
      let single_entry =
        (entry < first || entry > last)
        &&
        let ok = ref true in
        Array.iteri
          (fun a ss ->
            if not (inside a) then
              List.iter (fun b -> if inside b && b <> first then ok := false) ss)
          succ;
        !ok
      in
      if single_entry then begin
        let run =
          List.concat
            (Array.to_list (Array.sub t.runs first (last - first + 1)))
        in
        let own = defined run in
        let escapes =
          List.for_all
            (fun it ->
              match target it with
              | None -> true
              | Some l -> List.mem l own || is_block_label l)
            run
        in
        let outward =
          List.length
            (List.filter
               (fun it ->
                 match it with
                 | Item.Branch { label; _ } -> not (List.mem label own)
                 | _ -> false)
               run)
        in
        if escapes then
          found :=
            {
              first;
              last;
              size = bytes run + (4 * outward) + 4;
              depth = depth first;
            }
            :: !found
      end
    done
  done;
  List.rev !found

(* {1 Outlining} *)

type outlined = {
  entry : string;  (** the overlay's entry symbol *)
  items : Item.t list;
}

let sp_slot k = -4 * (k + 1)

let save r k =
  Item.Insn { mnemonic = "sw"; ops = [ A.reg r; A.mem "sp" (sp_slot k) ] }

let restore r k =
  Item.Insn { mnemonic = "lw"; ops = [ A.reg r; A.mem "sp" (sp_slot k) ] }

let jump l = Item.Insn { mnemonic = "j"; ops = [ A.sym l ] }

(* The name a resident block label is exported under. *)
let exported l =
  let l =
    if String.length l > 2 && String.sub l 0 2 = ".L" then
      String.sub l 2 (String.length l - 2)
    else l
  in
  l ^ "__x"

(* The function with [chosen] (disjoint candidates) moved out: the resident
   items, the overlays, and the resident labels the overlays jump to. *)
let split t ~name chosen =
  let n = Array.length t.labels in
  let chosen = List.sort (fun a b -> compare a.first b.first) chosen in
  let targets = ref [] in
  let export l =
    if not (List.mem l !targets) then targets := l :: !targets;
    exported l
  in
  let fresh = ref 0 in
  let overlays =
    List.mapi
      (fun k c ->
        let run =
          List.concat
            (Array.to_list (Array.sub t.runs c.first (c.last - c.first + 1)))
        in
        let own = defined run in
        let fix it =
          match it with
          | Item.Branch { mnemonic; regs; label } when not (List.mem label own)
            ->
              incr fresh;
              let over = Printf.sprintf ".Lovr%d" !fresh in
              [
                Item.Branch
                  { mnemonic = F.invert_mnemonic mnemonic; regs; label = over };
                jump (export label);
                Item.Label over;
              ]
          | Item.Insn
              { mnemonic = "j"; ops = [ Op.Sym (Asm_core.Expr.Symbol l) ] }
            when not (List.mem l own) ->
              [ jump (export l) ]
          | it -> [ it ]
        in
        let body = List.concat_map fix run in
        let tail =
          if c.last + 1 >= n then []
          else
            match List.rev run with
            | Item.Insn { mnemonic = "j" | "ret"; _ } :: _ -> []
            | _ -> [ jump (export t.labels.(c.last + 1)) ]
        in
        let entry = Printf.sprintf "%s_ovl%d" name k in
        ( c,
          {
            entry;
            items = List.mapi (fun i r -> restore r i) saved @ body @ tail;
          } ))
      chosen
  in
  let starts = Hashtbl.create 8 and inside = Hashtbl.create 64 in
  List.iter
    (fun (c, o) ->
      Hashtbl.replace starts c.first o;
      for k = c.first + 1 to c.last do
        Hashtbl.replace inside k ()
      done)
    overlays;
  let resident =
    t.head
    @ List.concat
        (List.init n (fun k ->
             if Hashtbl.mem inside k then []
             else
               match Hashtbl.find_opt starts k with
               | Some o ->
                   Item.Label t.labels.(k)
                   :: List.mapi (fun i r -> save r i) saved
                   @ [ jump o.entry ]
               | None -> t.runs.(k)))
  in
  let resident =
    List.concat_map
      (function
        | Item.Label l when List.mem l !targets ->
            [ Item.Label l; Item.Label (exported l) ]
        | it -> [ it ])
      resident
  in
  (resident, List.map snd overlays, List.map exported !targets)

(* Two disjoint candidates, as large as the smaller of them can be: what
   moving both out of the way saves is the smaller one's size, as they share
   one slot. *)
let best_pair cands =
  let disjoint a b = a.last < b.first || b.last < a.first in
  List.fold_left
    (fun best a ->
      List.fold_left
        (fun best b ->
          if a.first < b.first && disjoint a b then
            let v = min a.size b.size in
            match best with
            | Some (v', _) when v' >= v -> best
            | _ -> Some (v, [ a; b ])
          else best)
        best cands)
    None cands
  |> Option.map snd

let of_artifact ?mutation ?(choose = best_pair) artifact =
  let overlays = ref [] and globals = ref [] in
  let transform (f : func) items =
    let t = layout f items in
    match choose (candidates t) with
    | None | Some [] -> items
    | Some chosen ->
        let resident, ovls, exports =
          split t ~name:f.Mir_phys.Func.name chosen
        in
        overlays := !overlays @ ovls;
        globals := !globals @ exports;
        resident
  in
  Err.map
    (fun m -> (m, !overlays))
    (M.of_artifact ?mutation ~transform ~globals:(fun () -> !globals) artifact)

(* An overlay as a module of its own, entered at its symbol. *)
let module_ (o : outlined) =
  A.module_ o.entry (fun () ->
      [ A.text; A.align 4; A.global o.entry; A.label o.entry ]
      @ M.render o.items)
