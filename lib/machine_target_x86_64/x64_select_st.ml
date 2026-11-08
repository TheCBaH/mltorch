(* The x86-64 selector's state, refusals and fault injections, and the
   helpers every part of selection shares. *)

open Machine_ir
open X64_op
module S = X64_stage.Sel

module Refusal = struct
  type t =
    | Fallible_with_results of Mir_op.Callee.t
    | Feature of Mir_target.Feature.t
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
    | Feature f ->
        Fmt.pf fmt "needs %s, which the program's features do not include"
          (Mir_target.Feature.name f)
    | Invalid_selection d ->
        Fmt.pf fmt "selection defect: %a" Mir_diagnostic.pp d
    | Missing_site f -> Fmt.pf fmt "no site-table entry for %a" Mir_failure.pp f
    | Operation o -> Fmt.pf fmt "%s is not selected for x86_64" o
    | Vector r -> Mir_vsplit.Refusal.pp fmt r
    | Width t ->
        Fmt.pf fmt "type %a has no x86_64 register in this slice" Mir_type.pp t
end

module Mutation = struct
  type t =
    | Commuted_sub
    | Contract
    | Division_swap
    | Fused_float_eq
    | Insert_next_lane
    | Mask_arms
    | Mask_lt_le
    | Max_no_nan
    | Missing_failure_word
    | No_parity
    | Pruned_live
    | Scaled_address
    | Signed_compare
    | Unguarded_trunc
end

type result = { selected : S.Verified.t; record : Mir_id.View.t }

type st = {
  b : (X64.op, X64.test) Mir_select.t;
  esc : Refusal.t Err.Escape.t;
  mutation : Mutation.t option;
  sites : Mir_failure.Site_entry.t array;
  unlisted : Mir_failure.Unlisted.t;
  fallible : Mir_op.Callee.t -> bool;
  features : Mir_target.Feature.t list;
  int_slices : (int, int * Mir_value.t) Hashtbl.t;
      (** four lanes of an integer register read only by extensions to int32
          ({!Mir_vsplit.widening_slices}) *)
  shift_counts : (int, int64) Hashtbl.t;
      (** an integer splat read only as a lane shift count: the count
          ({!Mir_vsplit.shift_counts}) *)
  narrowing_halves : (int, Mir_value.t) Hashtbl.t;
      (** an [i16x4] truncation read only by a join into an [i16x8]: its
          [int32x4] source ({!Mir_vsplit.narrowing_halves}) *)
}

let refuse st r = Err.Escape.throw st.esc r
let mutated st m = st.mutation = Some m
let fresh st ty = Mir_select.fresh st.b ty
let push st ?order results op = Mir_select.push st.b ?order results op
let emit st ?result ty op = Mir_select.emit st.b ?result ty op
let ord st v = Mir_select.ord st.b v
let ord_edge st e = Mir_select.ord_edge st.b e

let sz_of st (t : Mir_type.t) =
  match t with
  | Mir_type.Int Mir_width.W64 | Mir_type.Ptr -> Sz.Q
  | Mir_type.Int Mir_width.W32 | Mir_type.Pred -> Sz.L
  | _ -> refuse st (Refusal.Width t)

let fsz_of st (t : Mir_type.t) =
  match t with
  | Mir_type.F64 -> Fsz.D
  | Mir_type.F32 -> Fsz.S
  | _ -> refuse st (Refusal.Width t)

let need_feature st f =
  if not (List.mem f st.features) then refuse st (Refusal.Feature f)

let const st ?result (c : Mir_const.t) =
  match c.Mir_const.ty with
  | Mir_type.Int _ | Mir_type.Pred ->
      emit st ?result c.Mir_const.ty
        (Mov_imm (c.Mir_const.ty, c.Mir_const.bits))
  | Mir_type.F64 ->
      let g = emit st Mir_type.i64 (Mov_imm (Mir_type.i64, c.Mir_const.bits)) in
      emit st ?result Mir_type.F64 (Movq_from_gpr (Fsz.D, g))
  | Mir_type.F32 ->
      let g = emit st Mir_type.i32 (Mov_imm (Mir_type.i32, c.Mir_const.bits)) in
      emit st ?result Mir_type.F32 (Movq_from_gpr (Fsz.S, g))
  | t -> refuse st (Refusal.Width t)

let icond st (c : Mir_op.Icmp.t) =
  match c with
  | Mir_op.Icmp.Eq -> Cond.E
  | Mir_op.Icmp.Ne -> Cond.Ne
  | Mir_op.Icmp.Sle -> Cond.Le
  | Mir_op.Icmp.Slt -> Cond.L
  | Mir_op.Icmp.Ule ->
      if mutated st Mutation.Signed_compare then Cond.Le else Cond.Be
  | Mir_op.Icmp.Ult ->
      if mutated st Mutation.Signed_compare then Cond.L else Cond.B

let msz_of (w : Mir_width.t) =
  match w with
  | Mir_width.W8 -> Msz.B
  | Mir_width.W16 -> Msz.H
  | Mir_width.W32 -> Msz.L
  | Mir_width.W64 -> Msz.Q
