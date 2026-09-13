(* The one intentionally recursive representation boundary.  All subsequent
   units depend on this module, never on the [Expr] library façade. *)

type binary_op = Add | Div | Mul | Sub
type unary_op = Cos | Erf | Exp | Log | Sin | Sqrt | Trunc

(* Same-width modular two's-complement results (design's initial I64 policy,
   see .ai/) -- [Int64.add]/[sub]/[mul] already wrap this way, so [i64_binary]
   needs no overflow check of its own. Division is deliberately absent: it
   names a rounding mode (truncating vs flooring) and has two exceptional
   cases (zero divisor, [min_int / -1]) neither arithmetic op has, so it is
   its own constructor when it lands, not a fourth case here. *)
type i64_binary_op = I64_add | I64_mul | I64_sub

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

(* Carrier-indexed. [I64_binary]/[I64_const] are the first inhabitants of a
   second index ([int64 value]) -- see .ai/. They deliberately reach no
   [Source]/[Local_var]/[Reduce_var]: a typed [Load]/[Local]/[Reduce] at
   [int64 value] is later work (P2 storage access, P3 typed locals), so for
   now an [int64 value] tree is a closed, environment-free constant/arithmetic
   expression, safely total to evaluate with no [Env]/scan/depth-cutoff
   machinery (see [Value.eval_i64]). Every other constructor below still
   returns [float value], the original inhabited index. *)
type _ value =
  | Binary : binary_op * float value * float value -> float value
  | Const : float -> float value
  | I64_binary : i64_binary_op * int64 value * int64 value -> int64 value
  | I64_const : int64 -> int64 value
  | I64_to_float : int64 value -> float value
      (** Exact-to-working-float, potentially lossy above 2^53 (design's "I64 to
          Float" policy) -- no exceptional case, unlike the reverse direction.
          The child stays a closed [int64 value] tree: [Fold]/
          [Check]/[Scan_admission] must still charge its own size/depth and
          confirm it holds no [Scan_at]/binder (trivially true today, since
          [int64 value] has neither), rather than treating this constructor as a
          leaf. *)
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
