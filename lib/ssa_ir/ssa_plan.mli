(** The resolved numerical plan of one kernel: which working precision it runs
    in and the program it is emitted from.

    Under [Reference_f64], or with no target, every kernel is binary64. Under a
    [Simd_fp32_*] policy a kernel the planner vectorizes in binary32 runs
    entirely in binary32; one it does not (or that {!Ssa_numerics.admit}
    refuses) is planned exactly as under [Reference_f64], so it stays under the
    bitwise binary64 shadow. This is the Loop planner's decision restated over
    the SSA program: binary32 is chosen by whether the binary32 target
    vectorizes at least one loop or schedules at least one sum. *)

type t = {
  numerics : Ssa_numerics.t;
  precision : Ssa_numerics.Precision.t;
  program : Ssa_program.t;
      (** the program to emit: its vector loops are planned against the target
          at [precision] *)
  refusal : Ssa_numerics.Refusal.t option;
      (** why an fp32 policy left this kernel binary64, when admission was the
          reason *)
  vectorized : Ssa_vectorize.report;
  sums : Ssa_vector_sum.report;
      (** the sums scheduled along their own axis; empty unless the policy
          permits reassociation *)
  blocked : int;  (** how many row loops were blocked around a vector loop *)
  contracted : int;  (** fused multiply-adds the plan holds *)
}

val resolve :
  ?target:Ssa_target.t ->
  ?alias:Ssa_effects.policy ->
  numerics:Ssa_numerics.t ->
  Ssa_program.t ->
  t
(** The program is the lowered scalar program: the plan runs the pre-vector
    passes itself, then (for binary32) the precision rewrite, the vectorizer,
    the sum scheduler where the policy permits reordering, the scalar blocking,
    contraction where the policy permits it and the target has a fused
    operation, and the clean-up passes. [alias] defaults to the conservative
    policy: a caller that allocates every buffer on its own says
    {!Ssa_effects.Distinct_buffers}. *)

val oracle : t -> Ssa_program.t
(** The scalar program the plan's kernel is defined to compute: the plan's
    program with every vector spelled out lane by lane ({!Ssa_vec_expand}). Run
    at the plan's precision it is the independent answer a generated kernel is
    checked against, bit for bit. *)
