(** Multiply-add contraction: under a policy that permits it and on a target
    that has a guaranteed fused operation, [a + x * y] (either operand order)
    becomes one {!Loop_expr.Fma} with a single rounding.

    It runs on the vector program, after the planner has scheduled the sums (a
    sum's accumulate is no longer a statement to rewrite: a scheduled sum whose
    term is a product is marked [fused] instead), so reduction recovery never
    sees a contracted accumulate. The scalar statements of a vector program (the
    remainders, the code around its loops) are contracted the same way, so a
    lane and its remainder compute alike.

    Only [Add] contracts: [a * b - c] and [c - a * b] would need a negation and
    are left as they were. Nothing else is rewritten, and a failure site is
    carried unchanged, so site numbering is the program's own. *)

val expr : 'a Loop_expr.t -> 'a Loop_expr.t
(** Contraction of a scalar expression, bottom-up. *)

val program :
  ?fuse_reductions:bool ->
  ?scalar:bool ->
  Loop_vector.program ->
  Loop_vector.program
(** [fuse_reductions] (default [false]) also marks each scheduled sum whose term
    is a product [fused], so its accumulate is one fused multiply-add. Measured
    on native NEON it is slower than the separate multiply and add (a
    sixteen-lane dot product: 3.3x over scalar binary64 fused, 6.2x unfused), so
    it is a permission the default plan does not use; the contraction of
    everything else is unconditional where the target has a fused operation.

    [scalar] (default [true]) also contracts the scalar statements around and
    after the vector loops. A target whose fused operation is vector-only
    (relaxed SIMD's multiply-add has no scalar form) passes [false]: only the
    vector expressions fuse, and the remainders stay a multiply and an add. *)
