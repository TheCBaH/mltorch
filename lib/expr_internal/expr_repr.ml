(* The one intentionally recursive representation boundary.  All subsequent
   units depend on this module, never on the [Expr] library façade. *)

type binary_op = Add | Div | Mul | Sub
type unary_op = Cos | Erf | Exp | Log | Sin | Sqrt | Trunc

(* [Argmax_index]/[Argmax_value] share [Max]/[Sum]'s [var]/[lo]/[hi]/[body]
   shape exactly -- [body] is still the per-position comparison key -- so they
   need no new field on [reduction]. What differs is the FOLD: [Max]/[Sum]
   combine consecutive [body] values with a commutative operator that never
   needs to know which position produced the winner, while the two [Argmax_*]
   kinds share ONE underlying paired (value, position) fold using
   [Max_op.pool_better] (ties keep the incumbent, a NaN retriggers) and differ
   only in which half of that pair they report -- [Argmax_value] the winning
   [body] value, [Argmax_index] the winning position, carried out as a value
   via the same conversion [Value_of_index] uses. Reusing [Max]'s [Float_max]
   comparator for the value half and inspecting [body]'s value again for the
   index half (two separate folds) is exactly the "fall out of step on NaN"
   defect [Intrinsic.Max_pool]'s own doc comment warns about, so both halves
   must run the identical [pool_better] fold. *)
type reduction_kind = Argmax_index | Argmax_value | Max | Sum

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
