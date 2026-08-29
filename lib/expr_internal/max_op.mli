(* The engine's two maximum semantics, in one place.

   They exist as a shared module rather than as open-coded [Float.max]/[>] at each
   site because THREE interpreters have to agree bit-for-bit: [Direct] runs the
   computation, the expression evaluator grounds the symbolic form against it, and
   (see .ai/native_transform_verify.md) the transformation verifier probes against
   both. A divergence between any two is a verifier that disagrees with the engine it
   verifies — which is exactly what happened before this module existed: [Direct]
   pooled with [Float.max] while the symbolic evaluator pooled with a strict [>], so
   the two disagreed on -0. vs +0. and on NaN.

   It lives in [lib/expr] rather than [lib/native] for that same reason. [Expr.Eval]
   cannot depend on [native], so leaving it there would force a second copy — which
   is precisely the duplication that produced the original divergence. [native]
   reaches it as [Expr.Max_op].

   The two operators are genuinely different and must not be conflated:
   [Pool_max] takes the LAST NaN in iteration order, [Float_max] propagates
   whichever NaN [Float.max] picks. Neither may be reassociated; [Pool_max] is not
   even commutative under NaN. *)

type t =
  | Float_max  (** generic [Reduce Max_reduce] — [Float.max] *)
  | Pool_max  (** ATen's max-pool predicate — see [pool_better] *)

(* ATen's max-pool selection predicate: the candidate wins on strict greater-than
   OR on NaN. Transcribed from aten/src/ATen/native/cpu/MaxPoolKernel.cpp:215

     bool mask = (val > maxval) || is_nan(val);
     out_data[d2] = mask ? val   : maxval;
     ind[d2]      = mask ? index : maxindex;

   and matching the vectorised path at :116, which blends on the same mask, so
   ATen has one well-defined answer here (unlike [amax] — see below).

   Two consequences the callers rely on. A NaN anywhere in the window propagates,
   and because every NaN re-triggers the predicate the LAST NaN wins, together
   with its index. An ordinary tie leaves the incumbent in place, which is what
   preserves the first/smallest-flattened-index contract documented in
   .ai/native_symbolic_language.md.

   The value and the index must be updated TOGETHER under this one predicate;
   updating them separately is how they fell out of step originally. *)
val pool_better : best:float -> value:float -> bool

(* [apply Float_max] is [Float.max]; [apply Pool_max] is the value half of
   [pool_better] (the index half has no [float -> float -> float] form).

   [Float_max] does NOT match ATen's [amax], and deliberately so: ATen dispatches
   both a scalar reducer ([zmath.h:210], first operand on a signed-zero tie, a
   canonical [quiet_NaN]) and a vector one ([vec_base.h:862], second operand on a
   tie, the operand's own NaN bits), and which runs depends on shape and vector
   width. There is no single semantics to target, so this stays [Float.max] until
   an empirical study says otherwise. What matters here is that all three
   interpreters share whatever it is. *)
val apply : t -> float -> float -> float
