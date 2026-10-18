(* The state selection threads through a function, and the naming of every
   i64 value's halves before any block is selected. *)

open Machine_ir

module Refusal = struct
  type t =
    | Fallible_with_results of Mir_op.Callee.t
    | Invalid_selection of Mir_diagnostic.t
    | Missing_site of Mir_failure.t
    | Operation of string
    | Vector of Mir_vsplit.Refusal.t
    | Width of Mir_type.t

  let pp fmt = function
    | Fallible_with_results c ->
        Fmt.pf fmt
          "a call to %a that may fail, from or with results this slice cannot \
           leave undefined"
          Mir_op.Callee.pp c
    | Invalid_selection d ->
        Fmt.pf fmt "selection defect: %a" Mir_diagnostic.pp d
    | Missing_site f -> Fmt.pf fmt "no site-table entry for %a" Mir_failure.pp f
    | Operation o -> Fmt.pf fmt "%s is not selected for riscv32" o
    | Vector r -> Mir_vsplit.Refusal.pp fmt r
    | Width t ->
        Fmt.pf fmt "type %a has no riscv32 register in this slice" Mir_type.pp t
end

module Mutation = struct
  type t =
    | Contract
    | Dropped_carry
    | Missing_failure_word
    | Number_max
    | Pruned_live
    | Signed_compare
    | Unguarded_trunc
end

(* An i64 as two words: its low and high halves, and whether it is known to
   be the sign extension of its low half. *)
type pair = { lo : Mir_value.t; hi : Mir_value.t; sext : bool }

type st = {
  b : (Rv32.op, Rv32.test) Mir_select.t;
  esc : Refusal.t Err.Escape.t;
  mutation : Mutation.t option;
  sites : Mir_failure.Site_entry.t array;
  unlisted : Mir_failure.Unlisted.t;
  fallible : Mir_op.Callee.t -> bool;
  pairs : (int, pair) Hashtbl.t;  (** every generic i64 value's halves *)
  subst : (int, Mir_value.t) Hashtbl.t;
      (** a generic value that is another one's half: its uses read that *)
  float_loads : (int, Mir_value.t) Hashtbl.t;
      (** a load read only by one bitcast to binary32: the float it defines *)
  int_slices : (int, int * Mir_value.t) Hashtbl.t;
      (** four lanes of an integer register read only by extensions to int32:
          the first lane and the register *)
  shift_counts : (int, int64) Hashtbl.t;
      (** an integer splat read only as a lane shift count: the count *)
}

let key (v : Mir_value.t) = Mir_id.Value.to_int v.Mir_value.id
let refuse st r = Err.Escape.throw st.esc r
let mutated st m = st.mutation = Some m
let fresh st ty = Mir_select.fresh st.b ty
let push st ?order results op = Mir_select.push st.b ?order results op
let emit st ?result ty op = Mir_select.emit st.b ?result ty op
let ord st v = Mir_select.ord st.b v
let const_of st v = Mir_select.const_of st.b v
let ty (v : Mir_value.t) = v.Mir_value.ty
let is_i64 v = Mir_type.equal (ty v) Mir_type.i64

(* The value a generic operand reads. *)
let rec u st (v : Mir_value.t) =
  match Hashtbl.find_opt st.subst (key v) with Some w -> u st w | None -> v

let pair st (v : Mir_value.t) =
  match Hashtbl.find_opt st.pairs (key v) with
  | Some p -> { p with lo = u st p.lo; hi = u st p.hi }
  | None -> refuse st (Refusal.Width (ty v))

let lo st v = (pair st v).lo

(* The loads whose one reader is a bitcast to binary32: the load becomes
   [flw] defining the bitcast's value, sparing the move between files. *)
let float_loads (f : (Mir_op.t, Mir_terminator.t) Mir_func.t) =
  let uses = Hashtbl.create 256 and defs = Hashtbl.create 256 in
  let use v =
    Hashtbl.replace uses (key v)
      (1 + Option.value ~default:0 (Hashtbl.find_opt uses (key v)))
  in
  List.iter
    (fun (b : (Mir_op.t, Mir_terminator.t) Mir_block.t) ->
      List.iter
        (fun (i : Mir_op.t Mir_instr.t) ->
          List.iter use (Mir_op.operands i.Mir_instr.op);
          List.iter
            (fun v -> Hashtbl.replace defs (key v) i.Mir_instr.op)
            i.Mir_instr.results)
        b.Mir_block.body;
      List.iter use (Mir_terminator.operands b.Mir_block.terminator))
    f.Mir_func.blocks;
  let fused = Hashtbl.create 64 in
  List.iter
    (fun (b : (Mir_op.t, Mir_terminator.t) Mir_block.t) ->
      List.iter
        (fun (i : Mir_op.t Mir_instr.t) ->
          match (i.Mir_instr.op, i.Mir_instr.results) with
          | Mir_op.Bitcast (Mir_type.F32, a), [ r ]
            when Hashtbl.find_opt uses (key a) = Some 1 -> (
              match Hashtbl.find_opt defs (key a) with
              | Some (Mir_op.Load { Mir_op.Access.width = Mir_width.W32; _ }) ->
                  Hashtbl.replace fused (key a) r
              | _ -> ())
          | _ -> ())
        b.Mir_block.body)
    f.Mir_func.blocks;
  fused

(* Every i64 value's halves, named before any block is selected so a use
   finds them whatever the block order: the low half of an extension is its
   operand, a copy shares its operand's halves, and a 32-bit truncation or
   narrowing of an i64 reads its low half directly. *)
let name_pairs st (f : (Mir_op.t, Mir_terminator.t) Mir_func.t) =
  let copies = ref [] in
  let fresh_pair ?lo ~sext () =
    {
      lo = (match lo with Some l -> l | None -> fresh st Mir_type.i32);
      hi = fresh st Mir_type.i32;
      sext;
    }
  in
  let add v p = Hashtbl.replace st.pairs (key v) p in
  List.iter
    (fun (b : (Mir_op.t, Mir_terminator.t) Mir_block.t) ->
      List.iter
        (fun v -> if is_i64 v then add v (fresh_pair ~sext:false ()))
        b.Mir_block.params;
      List.iter
        (fun (i : Mir_op.t Mir_instr.t) ->
          match (i.Mir_instr.op, i.Mir_instr.results) with
          | Mir_op.Iext (k, Mir_width.W64, a), [ r ]
            when Mir_type.equal (ty a) Mir_type.i32 ->
              add r (fresh_pair ~lo:a ~sext:(k = Mir_op.Iext.Sext) ())
          | Mir_op.Copy a, [ r ] when is_i64 r -> copies := (r, a) :: !copies
          | Mir_op.Const c, [ r ] when is_i64 r ->
              let s = c.Mir_const.bits in
              add r
                (fresh_pair
                   ~sext:
                     (Int64.compare s (-0x8000_0000L) >= 0
                     && Int64.compare s 0x7FFF_FFFFL <= 0)
                   ())
          | _, rs ->
              List.iter
                (fun r -> if is_i64 r then add r (fresh_pair ~sext:false ()))
                rs)
        b.Mir_block.body)
    f.Mir_func.blocks;
  (* a copy's halves are its operand's, through chains of copies *)
  let rec resolve (r, a) =
    match Hashtbl.find_opt st.pairs (key a) with
    | Some p -> add r p
    | None -> (
        match List.find_opt (fun (r', _) -> Mir_value.equal r' a) !copies with
        | Some c ->
            resolve c;
            resolve (r, a)
        | None -> ())
  in
  List.iter resolve !copies;
  List.iter
    (fun (b : (Mir_op.t, Mir_terminator.t) Mir_block.t) ->
      List.iter
        (fun (i : Mir_op.t Mir_instr.t) ->
          match (i.Mir_instr.op, i.Mir_instr.results) with
          | ( ( Mir_op.Itrunc (Mir_width.W32, a)
              | Mir_op.Narrow (Mir_width.W32, a) ),
              [ r ] )
            when is_i64 a -> (
              match Hashtbl.find_opt st.pairs (key a) with
              | Some p -> Hashtbl.replace st.subst (key r) p.lo
              | None -> ())
          | _ -> ())
        b.Mir_block.body)
    f.Mir_func.blocks
