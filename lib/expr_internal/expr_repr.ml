(* The one intentionally recursive representation boundary.  All subsequent
   units depend on this module, never on the [Expr] library façade. *)

type binary_op = Add | Div | Mul | Sub
type unary_op = Erf | Exp | Log | Sqrt | Trunc
type reduction_kind = Max | Sum

type value =
  | Binary of binary_op * value * value
  | Const of float
  | Intrinsic of Intrinsic.t
  | Local of Local_var.t
  | Load of Source.t * Role.Position.t Index.t Coord.t
  | Reduce of reduction
  | Round_f32 of value
  | Select of bool_expr * value * value
  | Unary of unary_op * value
  | Value_of_index of Role.Delta.t Index.t

and bool_expr =
  | Index_eq of Role.Delta.t Index.t * Role.Delta.t Index.t
  | Value_lt of value * value

and reduction = {
  kind : reduction_kind;
  var : Reduce_var.t;
  lo : Role.Position.t Index.t;
  hi : Role.Delta.t Index.t;
  body : value;
}
