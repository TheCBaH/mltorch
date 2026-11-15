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

(* A block's position in the function's layout order. *)
module Position =
  Core.Tagged_int.Make
    (struct
      let prefix = "pos"
    end)
    ()

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
  index : (Mir_id.Block.t, Position.t) Hashtbl.t;
}

let layout (f : func) items =
  let blocks = Array.of_list f.Mir_phys.Func.blocks in
  let labels =
    Array.map (fun (b : (_, _) Mir_phys.Block.t) -> M.label_of f b.id) blocks
  in
  let index = Hashtbl.create 64 in
  Array.iteri
    (fun k (b : (_, _) Mir_phys.Block.t) ->
      Hashtbl.replace index b.id (Position.of_int k))
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
  let b = List.nth t.func.Mir_phys.Func.blocks (Position.to_int k) in
  List.map
    (fun id -> Hashtbl.find t.index id)
    (Mir_phys.Term.successors b.Mir_phys.Block.terminator)

let target = function
  | Item.Branch { label; _ } -> Some label
  | Item.Insn { mnemonic = "j"; ops = [ Op.Sym (Asm_core.Expr.Symbol l) ] } ->
      Some l
  | _ -> None

let bytes items = Item.Size_math.total (List.map Item.bytes items)

(* [depth]: how many loops the region's entry is inside, taking a branch back
   to an earlier block in layout order as a loop over the blocks between. *)
type candidate = {
  first : Position.t;
  last : Position.t;
  size : Item.Size.t;
  depth : int;
}

(* The labels a run defines, and those its branches and jumps name. *)
let defined run =
  List.filter_map (function Item.Label l -> Some l | _ -> None) run

(* Whether a run names one of [names]: a call of it, or its address. *)
let mentions names run =
  List.exists
    (function
      | Item.Insn { ops; _ } ->
          List.exists
            (function
              | Op.Sym (Asm_core.Expr.Symbol n) -> List.mem n names | _ -> false)
            ops
      | Item.Branch _ | Item.Label _ -> false)
    run

let candidates ?(avoid = []) t =
  let n = Array.length t.labels in
  let entry =
    Position.to_int (Hashtbl.find t.index t.func.Mir_phys.Func.entry)
  in
  let succ =
    Array.init n (fun k ->
        List.map Position.to_int (successors t (Position.of_int k)))
  in
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
        if escapes && not (mentions avoid run) then
          found :=
            {
              first = Position.of_int first;
              last = Position.of_int last;
              size =
                Item.Size_math.(
                  add (bytes run)
                    (add
                       (scale outward (Item.Size.of_int 4))
                       (Item.Size.of_int 4)));
              depth = depth first;
            }
            :: !found
      end
    done
  done;
  List.rev !found

(* {1 Outlining} *)

let run_of t c =
  let first = Position.to_int c.first in
  List.concat
    (Array.to_list
       (Array.sub t.runs first (Position.to_int c.last - first + 1)))

type outlined = {
  entry : string;  (** the overlay's entry symbol *)
  items : Item.t list;
}

(* Which word below the stack pointer a saved register is kept in. *)
module Save_slot =
  Core.Tagged_int.Make
    (struct
      let prefix = "save"
    end)
    ()

let sp_slot k = A.mem "sp" (-4 * (Save_slot.to_int k + 1))
let save r k = Item.Insn { mnemonic = "sw"; ops = [ A.reg r; sp_slot k ] }
let restore r k = Item.Insn { mnemonic = "lw"; ops = [ A.reg r; sp_slot k ] }
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
  let chosen = List.sort (fun a b -> Position.compare a.first b.first) chosen in
  let targets = ref [] in
  let export l =
    if not (List.mem l !targets) then targets := l :: !targets;
    exported l
  in
  let fresh = ref 0 in
  let overlays =
    List.mapi
      (fun k c ->
        let run = run_of t c in
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
          if Position.to_int c.last + 1 >= n then []
          else
            match List.rev run with
            | Item.Insn { mnemonic = "j" | "ret"; _ } :: _ -> []
            | _ -> [ jump (export t.labels.(Position.to_int c.last + 1)) ]
        in
        let entry = Printf.sprintf "%s_ovl%d" name k in
        ( c,
          {
            entry;
            items =
              List.mapi (fun i r -> restore r (Save_slot.of_int i)) saved
              @ body @ tail;
          } ))
      chosen
  in
  let starts = Hashtbl.create 8 and inside = Hashtbl.create 64 in
  List.iter
    (fun (c, o) ->
      Hashtbl.replace starts (Position.to_int c.first) o;
      for k = Position.to_int c.first + 1 to Position.to_int c.last do
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
                   :: List.mapi (fun i r -> save r (Save_slot.of_int i)) saved
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
  let disjoint a b =
    Position.compare a.last b.first < 0 || Position.compare b.last a.first < 0
  in
  List.fold_left
    (fun best a ->
      List.fold_left
        (fun best b ->
          if Position.compare a.first b.first < 0 && disjoint a b then
            let v = Item.Size_math.min a.size b.size in
            match best with
            | Some (v', _) when Item.Size.compare v' v >= 0 -> best
            | _ -> Some (v, [ a; b ])
          else best)
        best cands)
    None cands
  |> Option.map snd

(* Disjoint regions for a slot an overlay of [slot] bytes already needs: for
   each slot size from [slot] up to the largest candidate, the disjoint regions
   that fit it, largest first, scored by what they move out less what the slot
   grows by; the best of them. *)
let fill ~slot cands =
  let disjoint a b =
    Position.compare a.last b.first < 0 || Position.compare b.last a.first < 0
  in
  let take limit =
    List.fold_left
      (fun taken c ->
        if
          Item.Size.compare c.size limit <= 0 && List.for_all (disjoint c) taken
        then c :: taken
        else taken)
      []
      (List.sort (fun a b -> Item.Size.compare b.size a.size) cands)
  in
  let sizes =
    List.sort_uniq Item.Size.compare
      (slot
      :: List.filter_map
           (fun c ->
             if Item.Size.compare c.size slot > 0 then Some c.size else None)
           cands)
  in
  let score limit taken =
    Item.Size_math.sub
      (Item.Size_math.total (List.map (fun c -> c.size) taken))
      (Item.Size_math.sub limit slot)
  in
  match
    List.fold_left
      (fun best limit ->
        let taken = take limit in
        let v = score limit taken in
        match best with
        | Some (v', _) when Item.Size.compare v' v >= 0 -> best
        | _ -> if taken = [] then best else Some (v, taken))
      None sizes
  with
  | None -> None
  | Some (_, taken) ->
      Some (List.sort (fun a b -> Position.compare a.first b.first) taken)

let of_artifact ?mutation ?(choose = best_pair) ?avoid artifact =
  let overlays = ref [] and globals = ref [] in
  let transform (f : func) items =
    let t = layout f items in
    match choose (candidates ?avoid t) with
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
