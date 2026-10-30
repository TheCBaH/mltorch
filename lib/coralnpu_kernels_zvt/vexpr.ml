module R = Machine_rivet_riscv32
module Item = R.Rv32_kernel_item
module Insn = R.Rv32_kernel_insn
module Imm = R.Rv32_kernel_imm
module Gpr = R.Rv32_kernel_reg.Gpr
module Vreg = R.Rv32_kernel_reg.Vreg

type bin = Add | Div | Mul | Sub
type cmp = Eq | Lt

type t = { id : int; desc : desc }

and desc =
  | Const of float
  | Bin of bin * t * t
  | Input of int
  | Select of cmp * t * t * t * t

let counter = ref 0

let node desc =
  incr counter;
  { id = !counter; desc }

let input k = node (Input k)

module Ops = struct
  type nonrec t = t

  let const x = node (Const x)
  let add a b = node (Bin (Add, a, b))
  let sub a b = node (Bin (Sub, a, b))
  let mul a b = node (Bin (Mul, a, b))
  let div a b = node (Bin (Div, a, b))
  let select_lt a b c d = node (Select (Lt, a, b, c, d))
  let select_eq a b c d = node (Select (Eq, a, b, c, d))
  let share x = x
end

module Plan = struct
  type nonrec t = { items : Item.t list; result : Vreg.t; peak : int }
end

exception Out_of_registers of string

module Int_map = Map.Make (Int)

let children n =
  match n.desc with
  | Const _ | Input _ -> []
  | Bin (_, a, b) -> [ a; b ]
  | Select (_, a, b, c, d) -> [ a; b; c; d ]

let is_const n = match n.desc with Const _ -> true | _ -> false

(* How many operand slots read each non-constant node. *)
let use_counts root =
  let seen = Hashtbl.create 64 and uses = Hashtbl.create 64 in
  let rec go n =
    if not (Hashtbl.mem seen n.id) then (
      Hashtbl.add seen n.id ();
      List.iter
        (fun c ->
          if not (is_const c) then
            Hashtbl.replace uses c.id
              (1 + Option.value (Hashtbl.find_opt uses c.id) ~default:0);
          go c)
        (children n))
  in
  go root;
  uses

(* Registers a subtree needs, evaluating the larger child first (Sethi-Ullman,
   counting a shared node each time: an estimate that orders the emission). *)
let need_table () =
  let memo = Hashtbl.create 64 in
  let rec go n =
    match Hashtbl.find_opt memo n.id with
    | Some v -> v
    | None ->
        let v =
          match n.desc with
          | Const _ | Input _ -> 1
          | Bin (_, a, b) ->
              let na = go a and nb = go b in
              if na = nb then na + 1 else max na nb
          | Select (_, a, b, c, d) ->
              1 + List.fold_left (fun m n -> max m (go n)) 0 [ a; b; c; d ]
        in
        Hashtbl.add memo n.id v;
        v
  in
  go

let ok = function
  | Ok x -> x
  | Error _ -> invalid_arg "Vexpr: a register or immediate"

let li ~scratch bits =
  match Err.payload (Insn.load_imm scratch bits) with
  | Ok is -> List.map Item.insn is
  | Error _ -> invalid_arg "Vexpr: a constant"

let emit ~lmul ~scratch ~free ~inputs root =
  try
    let uses = use_counts root in
    let need_of = need_table () in
    let free = ref (List.sort compare (List.map Vreg.index free)) in
    let live = ref (Array.length inputs) in
    let peak = ref !live in
    List.iter
      (fun i ->
        if i mod lmul <> 0 || i = 0 then
          invalid_arg
            (Printf.sprintf "Vexpr: v%d is not a usable register group" i))
      !free;
    let alloc what =
      match !free with
      | [] -> raise (Out_of_registers what)
      | r :: rest ->
          free := rest;
          incr live;
          peak := max !peak !live;
          ok (Vreg.of_index r)
    in
    let release r =
      decr live;
      free := List.sort compare (Vreg.index r :: !free)
    in
    let items = ref [] in
    let emit_item i = items := i :: !items in
    let emit_insn i = emit_item (Item.insn i) in
    let kept = Hashtbl.create 8 in
    Array.iter
      (fun (r, mode) ->
        match mode with
        | `Kept -> Hashtbl.replace kept (Vreg.index r) ()
        | `Consumed -> ())
      inputs;
    let reg_of = Hashtbl.create 64 in
    let remaining = Hashtbl.copy uses in
    let v0 = ok (Vreg.of_index 0) in
    let constant x =
      let r = alloc "a constant" in
      let bits = Int32.to_int (Int32.bits_of_float x) land 0xFFFF_FFFF in
      if bits = 0 then emit_insn (Insn.Vmv_v_x { dst = r; src = Gpr.zero })
      else (
        List.iter emit_item (li ~scratch bits);
        emit_insn (Insn.Vmv_v_x { dst = r; src = scratch }));
      r
    in
    let rec eval n =
      match Hashtbl.find_opt reg_of n.id with
      | Some r -> r
      | None ->
          let r =
            match n.desc with
            | Input k -> fst inputs.(k)
            | Const _ -> assert false
            | Bin (op, a, b) -> (
                match operands [ a; b ] with
                | [ ra; rb ] ->
                    let dst = finish [ (a, ra); (b, rb) ] "a result" in
                    (match op with
                    | Add ->
                        emit_insn
                          (Insn.Vv
                             { op = Insn.Vv_op.Vfadd; dst; a = ra; b = rb })
                    | Mul ->
                        emit_insn
                          (Insn.Vv
                             { op = Insn.Vv_op.Vfmul; dst; a = ra; b = rb })
                    | Sub ->
                        emit_insn
                          (Insn.Valu
                             { op = Insn.Alu_op.Vfsub; dst; a = ra; b = rb })
                    | Div ->
                        emit_insn
                          (Insn.Valu
                             { op = Insn.Alu_op.Vfdiv; dst; a = ra; b = rb }));
                    release_constants [ (a, ra); (b, rb) ];
                    dst
                | _ -> assert false)
            | Select (cmp, a, b, c, d) -> (
                match operands [ a; b; c; d ] with
                | [ ra; rb; rc; rd ] ->
                    emit_insn
                      (Insn.Vmcmp
                         {
                           cmp =
                             (match cmp with
                             | Eq -> Insn.Cmp.Feq
                             | Lt -> Insn.Cmp.Flt);
                           dst = v0;
                           a = ra;
                           b = rb;
                         });
                    (* the compare has read a and b: their registers may be dst *)
                    let dst =
                      finish
                        [ (a, ra); (b, rb); (c, rc); (d, rd) ]
                        "a selection"
                    in
                    emit_insn
                      (Insn.Vmerge_vvm { dst; otherwise = rd; chosen = rc });
                    release_constants [ (a, ra); (b, rb); (c, rc); (d, rd) ];
                    dst
                | _ -> assert false)
          in
          Hashtbl.add reg_of n.id r;
          r
    (* the registers of [ns] in order: the larger subtrees first, a constant
       built last, so that it is live for the shortest time *)
    and operands ns =
      let indexed = List.mapi (fun i n -> (i, n)) ns in
      let computed, constants =
        List.partition (fun (_, n) -> not (is_const n)) indexed
      in
      let computed =
        List.stable_sort
          (fun (_, x) (_, y) -> compare (need_of y) (need_of x))
          computed
      in
      let regs =
        List.map (fun (i, n) -> (i, eval n)) computed
        @ List.map
            (fun (i, n) ->
              (i, match n.desc with Const x -> constant x | _ -> assert false))
            constants
      in
      List.map (fun (i, _) -> List.assoc i regs) indexed
    (* count the operands' reads and free the registers read for the last
       time, then allocate the destination: it may be one of them *)
    and finish operands what =
      let dying = ref [] in
      List.iter
        (fun (n, r) ->
          if not (is_const n) then (
            let left = Hashtbl.find remaining n.id - 1 in
            Hashtbl.replace remaining n.id left;
            if
              left = 0
              && (not (Hashtbl.mem kept (Vreg.index r)))
              && not (List.exists (Vreg.equal r) !dying)
            then dying := r :: !dying))
        operands;
      List.iter release !dying;
      alloc what
    and release_constants operands =
      List.iter (fun (n, r) -> if is_const n then release r) operands
    in
    (* nodes in the order the sequence built them: a value is used soon after
       it is made, which keeps fewer registers live than a depth-first walk *)
    let reachable =
      let seen = Hashtbl.create 64 and acc = ref [] in
      let rec go n =
        if not (Hashtbl.mem seen n.id) then (
          Hashtbl.add seen n.id ();
          List.iter go (children n);
          acc := n :: !acc)
      in
      go root;
      List.sort (fun a b -> compare a.id b.id) !acc
    in
    List.iter
      (fun n ->
        match n.desc with
        | Const _ | Input _ -> ()
        | Bin _ | Select _ -> ignore (eval n))
      reachable;
    let result = eval root in
    Ok { Plan.items = List.rev !items; result; peak = !peak }
  with
  | Out_of_registers what -> Error ("out of vector registers for " ^ what)
  | Invalid_argument m -> Error m
