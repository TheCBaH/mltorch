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
type i64_binary_op = I64_add | I64_div | I64_mul | I64_sub

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

(* Carrier-indexed. [I64_binary]/[I64_const]/[I64_load]/[I64_local]/
   [I64_local_at]/[I64_of_index] are the inhabitants of a second index
   ([int64 value]) -- see
   .ai/. [I64_load] is the first of them to reach [Source.t]: it resolves
   through [Env.load_index] (already exact I64, previously reached only via
   [Data] index components) exactly as [Load] resolves through [Env.load], so
   an [int64 value] tree is no longer unconditionally closed/environment-free
   -- only a tree built without [I64_load]/[I64_local]/[I64_local_at] is (see
   [Value.eval_i64]'s own doc comment on what that means for its standalone
   callback shape). A typed reduction exists as [I64_sum] (an exact modular
   accumulator, sum only); a typed [Scan_at] (previous-row references) does
   not, and waits for an op that needs an int64 scan. [I64_local]/[I64_local_at] give scalar/vector locals the same
   typed treatment [Local]/[Local_at] already have at [float value]. Every
   other constructor below still returns [float value], the original
   inhabited index. *)
type _ value =
  | Binary : binary_op * float value * float value -> float value
  | Const : float -> float value
  | Float_to_i64 : float value -> int64 value
      (** Truncating, per the design's "Float to I64" policy: finite values in
          [-2^63, 2^63) truncate toward zero; NaN, infinities and
          out-of-range values are structured errors ([Value.i64_of_float]),
          never a wrapped/clamped result. Unlike [I64_to_float], the operand
          is the UNBOUNDED [float value] language (it can embed a [Load],
          [Reduce], [Scan_at] -- anything), so evaluating this needs the full
          environment-carrying evaluator for its child, not a closed
          standalone function. *)
  | I64_binary : i64_binary_op * int64 value * int64 value -> int64 value
  | I64_const : int64 -> int64 value
  | I64_load : Source.t * Role.Position.t Index.t Coord.t -> int64 value
      (** Exact I64 tensor read, the [int64 value] counterpart of [Load]:
          resolves through [Env.load_index] rather than [Env.load], so a value
          beyond float's 2^53 exact-mantissa range round-trips intact.
          Wrong-format/out-of-range binding errors are the same [Env.load_index]
          already reports for a [Data] index component -- this constructor is a
          second caller of that one resolver, not a new one. *)
  | I64_local : Local_var.t -> int64 value
      (** The [int64 value] counterpart of [Local]: an I64-typed scalar Region
          local, resolved by a caller-supplied typed reader exactly as [Local]
          is (see [Eval.value]'s [local_i64] parameter). *)
  | I64_local_at : Local_var.t * Role.Position.t Index.t -> int64 value
      (** The [int64 value] counterpart of [Local_at]: an I64-typed vector
          Region local read at a position. *)
  | I64_of_index : Role.Delta.t Index.t -> int64 value
      (** The [int64 value] counterpart of [Value_of_index]: an index carried
          into the value domain EXACTLY, not through a float ordinal. Unlike
          [Value_of_index] (which can lose precision converting a large index to
          binary64), this conversion is total and lossless -- every
          [int]-represented index, on any backend width, fits in [int64]. *)
  | I64_sum : i64_reduction -> int64 value
      (** Typed reduction over [i64_lo..i64_hi) of [i64_body], per [i64_kind]:
          [Sum] accumulates in int64 with the same modular two's-complement
          policy as [I64_binary] (exact, never a float; empty range [0L]);
          [Max]/[Argmax_value] is the signed maximum (empty range
          [Int64.min_int], the analogue of the float fold's -infinity);
          [Argmax_index] is the position of the first maximum (ties keep the
          incumbent; there is no NaN in int64), carried as an exact int64, and
          [lo] on an empty range, as the float [Argmax_index] does. *)
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
  | Select : bool_expr * 'a value * 'a value -> 'a value
  | Unary : unary_op * float value -> float value
  | Value_of_index : Role.Delta.t Index.t -> float value

(* Not part of the [_ value] GADT: a predicate always denotes [bool], so
   giving it its own index would only ever be instantiated at [bool], and
   [Select]'s carrier-crossing generality (see [Select]'s own doc comment
   above) already covers a predicate built from one. [I64_eq]/[I64_lt] are
   the first inhabitants at [int64 value] operands -- like [Value_lt]'s
   [float value] operands, they are the UNBOUNDED language (either can embed
   a [Float_to_i64]), so evaluating one needs the same environment-carrying
   machinery [Select]'s own [int64 value] branches now do, not a closed
   standalone function. *)
and bool_expr =
  | I64_eq of int64 value * int64 value
  | I64_lt of int64 value * int64 value
  | Index_eq of Role.Delta.t Index.t * Role.Delta.t Index.t
  | Value_eq of float value * float value
      (** IEEE numerical equality, matching OCaml's [( = )] on [float]: NaN
          compares unequal to everything including itself, signed zeros compare
          equal. This is the primitive [Value_lt]'s own doc comment (see
          [semantics.ml]) says is deliberately absent -- it is now needed by the
          Bool-cast "nonzero test", which [Value_lt]-only formulas cannot
          express (both [lt 0 x] and [lt x 0] are false for NaN, so a
          double-[Value_lt] "nonzero" test wrongly reports NaN as zero; see the
          design's "Float to Bool" policy). *)
  | Value_lt of float value * float value

and i64_reduction = {
  i64_kind : reduction_kind;
  i64_var : Reduce_var.t;
  i64_lo : Role.Position.t Index.t;
  i64_hi : Role.Delta.t Index.t;
  i64_body : int64 value;
}

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
