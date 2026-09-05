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
  | Local_at of Local_var.t * Role.Position.t Index.t
  | Local_scan_at of
      Local_var.t * Role.Position.t Index.t * Role.Position.t Index.t
  | Load of Source.t * Role.Position.t Index.t Coord.t
  | Reduce of reduction
  | Round_f32 of value
  | Scan_at of scan * Role.Position.t Index.t * Role.Position.t Index.t
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

(* [trace.(0, l) = init[lane := l]]; [trace.(s+1, l) = update[step := s, lane
   := l, prev := trace.(s, ·)]]. [lane] is bound in both [init] and [update]
   (two sibling scopes); [step] and [prev] are bound in [update] only. [prev]
   is read as [Local_at (prev, i)], an ordinary local read within [update] --
   the first place this language binds a [Local_var.t] rather than only
   naming a Region-supplied one. Row and lane are always two separate index
   arguments, never packed into one flattened index. *)
and scan = {
  width : int;
  steps : int;
  lane : Reduce_var.t;
  step : Reduce_var.t;
  prev : Local_var.t;
  init : value;
  update : value;
}
