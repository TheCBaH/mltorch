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

(* Carrier-indexed: every constructor below returns [float value], the only
   inhabited index today -- this is a behavior-preserving reshape into GADT
   form (see .ai/), not yet an admission of a second carrier. A future I64/Bool
   constructor is a new case returning [int64 value]/[bool value], added
   alongside these without disturbing them. *)
type _ value =
  | Binary : binary_op * float value * float value -> float value
  | Const : float -> float value
  | Intrinsic : Intrinsic.t -> float value
  | Local : Local_var.t -> float value
  | Local_at : Local_var.t * Role.Position.t Index.t -> float value
  | Local_scan_at :
      Local_var.t * Role.Position.t Index.t * Role.Position.t Index.t
      -> float value
  | Load : Source.t * Role.Position.t Index.t Coord.t -> float value
  | Reduce : reduction -> float value
  | Round_f32 : float value -> float value
  | Scan_at :
      scan * Role.Position.t Index.t * Role.Position.t Index.t
      -> float value
  | Select : bool_expr * float value * float value -> float value
  | Unary : unary_op * float value -> float value
  | Value_of_index : Role.Delta.t Index.t -> float value

and bool_expr =
  | Index_eq of Role.Delta.t Index.t * Role.Delta.t Index.t
  | Value_lt of float value * float value

and reduction = {
  kind : reduction_kind;
  var : Reduce_var.t;
  lo : Role.Position.t Index.t;
  hi : Role.Delta.t Index.t;
  body : float value;
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
  init : float value;
  update : float value;
}
