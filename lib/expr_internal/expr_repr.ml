(* The one intentionally recursive representation boundary.  All subsequent
   units depend on this module, never on the [Expr] library façade. *)

type binary_op = Add | Sub | Mul | Div
type unary_op = Exp | Sqrt | Erf | Log
type reduction_kind = Sum | Max

type value =
  | Const of float
  | Binary of binary_op * value * value
  | Unary of unary_op * value
  | Select of bool_expr * value * value
  | Value_of_index of Role.Delta.t Index.t
  | Load of Source.t * Role.Position.t Index.t Coord.t
  | Round_f32 of value
  | Reduce of reduction
  | Intrinsic of Intrinsic.t

and bool_expr =
  | Value_lt of value * value
  | Index_eq of Role.Delta.t Index.t * Role.Delta.t Index.t

and reduction = {
  kind : reduction_kind;
  var : Reduce_var.t;
  lo : Role.Position.t Index.t;
  hi : Role.Delta.t Index.t;
  body : value;
}
