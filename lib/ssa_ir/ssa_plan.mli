(** The resolved numerical plan of one kernel: which working precision it runs
    in and the program it is emitted from.

    Under [Reference_f64], or with no target, every kernel is binary64. Under a
    [Simd_fp32_*] policy a kernel the planner vectorizes in binary32 runs
    entirely in binary32; one it does not (or that {!Ssa_numerics.admit}
    refuses) is planned exactly as under [Reference_f64], so it stays under the
    bitwise binary64 shadow. This is the Loop planner's decision restated over
    the SSA program: binary32 is chosen by whether the binary32 target
    vectorizes at least one loop or schedules at least one sum. Under
    [Fp32_ordered] every kernel is binary32 whether or not anything vectorizes,
    and one {!Ssa_numerics.admit} refuses has no plan: {!admitted} is its error.
    With a target, [Fp32_ordered] is {!fp32_vector}: loops vectorize on the
    exact binary32 form, holding binary32 vectors only, and every other loop
    stays the exact scalar form. *)

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
  ?contractions:bool ->
  numerics:Ssa_numerics.t ->
  Ssa_program.t ->
  t
(** The program is the lowered scalar program: the plan runs the pre-vector
    passes itself, then (for binary32) the precision rewrite, the vectorizer,
    the sum scheduler where the policy permits reordering, the scalar blocking,
    contraction where the policy permits it and the target has a fused
    operation, and the clean-up passes. [alias] defaults to the conservative
    policy: a caller that allocates every buffer on its own says
    {!Ssa_effects.Distinct_buffers}. With [contractions] (default [false]) a
    binary32 plan ([Fp32_ordered], [Fp32_contracted]) makes each matrix product
    {!Ssa_contract} recognizes one [Contract], ordered or fused as the policy
    is, before it vectorizes; only the Machine IR route realizes one. *)

val vectorize_f32 :
  ?alias:Ssa_effects.policy ->
  ?contract:Contraction.Numerics.t ->
  ?elements:Ssa_type.scalar list ->
  target:Ssa_target.t ->
  exact:bool ->
  Ssa_program.t ->
  (Ssa_program.t * Ssa_vectorize.report, Ssa_numerics.Refusal.t) result
(** The binary32 vector program a plan starts from, before any later pass: the
    pre-vector passes, admission, {!Ssa_precision.to_f32} (with [exact] as
    there) and the vectorizer against the binary32 target with [elements] as
    {!Ssa_vectorize.program} takes them, with its decisions. [exact:false] is
    the narrowed form (each read widened, then narrowed lane by lane). For a
    census of what each form vectorizes; {!resolve} is the plan. *)

val fp32_vector :
  ?alias:Ssa_effects.policy ->
  ?contract:Contraction.Numerics.t ->
  target:Ssa_target.t ->
  Ssa_program.t ->
  (t, Ssa_numerics.Refusal.t) result
(** The [Fp32_ordered] vector plan: the exact binary32 form vectorized with only
    binary32 vectors admitted, then the row blocking and the clean-up passes. A
    kernel with no vector loop is the scalar [Fp32_ordered] plan. The plan holds
    no binary64 vector ({!Ssa_numerics.wide_values}): [Invalid_argument] if it
    would. [Error] when admission refuses. *)

val binary32_scalar :
  ?alias:Ssa_effects.policy -> Ssa_program.t -> Ssa_program.t
(** The scalar binary32 form a plan starts from (the passes every plan runs
    first, then the exact narrowing): where a matrix product is still one
    ordered sum inside its loops ({!Ssa_contract}). *)

val admitted : t -> (t, Ssa_numerics.Refusal.t) result
(** [Error] when the plan's policy is [Fp32_ordered] and admission left the
    kernel binary64: that policy has no binary64 kernels, so the refusal is an
    error, never a silent fallback. Every other plan is [Ok]. *)

val oracle : t -> Ssa_program.t
(** The scalar program the plan's kernel is defined to compute: the plan's
    program with every vector spelled out lane by lane ({!Ssa_vec_expand}). Run
    at the plan's precision it is the independent answer a generated kernel is
    checked against, bit for bit. *)
