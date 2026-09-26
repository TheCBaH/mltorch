(* [Expr.Index.t]'s arithmetic with [Output] and [Reduce] replaced by loop
   variables. The role parameter is dropped: [Expr] already discharged it, so
   [Of_position] and [Assume_position] are the identity here and [Zero] is
   [Const 0]. Overflow is not part of the meaning of a node; see
   [Loop_failure.Index_overflow]. *)
type t =
  | Add of t * t
  | Ceil_div_pos of t * int
  | Clamp_low of t
  | Const of int
  | Floor_div_pos of t * int
  | Max of t * t
  | Min of t * t
  | Scale of int * t
  | Temp of Loop_temp.t
  | Var of Loop_var.t

type coord = t Expr.Coord.t
