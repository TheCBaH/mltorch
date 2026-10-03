(** The resolved numerical plan of one kernel: which working precision it runs
    in and the vector program it is emitted from.

    Under [Reference_f64] every kernel is binary64. Under a [Simd_fp32_*] policy
    a kernel the planner vectorizes in binary32 runs entirely in binary32; one
    it does not (or that {!Loop_numerics.admit} refuses) is planned exactly as
    under [Reference_f64], so it stays under the bitwise f64 shadow. *)

type t = {
  numerics : Loop_numerics.t;
  precision : Loop_numerics.Precision.t;
  vector : Loop_vector.program option;
      (** the program to emit when a target was given: its vector loops are
          planned against the target at [precision] *)
  refusal : Loop_numerics.Refusal.t option;
      (** why an fp32 policy left this kernel binary64, when admission was the
          reason *)
  report : Loop_vectorize.report;  (** the decisions of [vector]'s planning *)
  blocked : int;  (** how many row loops {!Loop_block} blocked in [vector] *)
}

val resolve :
  ?target:Loop_target.t ->
  ?fuse_reductions:bool ->
  numerics:Loop_numerics.t ->
  Loop_program.t ->
  t
(** [fuse_reductions] is {!Loop_contract.program}'s: off by default, because a
    fused accumulate measured slower than the separate multiply and add. *)

val force :
  ?target:Loop_target.t ->
  precision:Loop_numerics.Precision.t ->
  Loop_program.t ->
  (t, Loop_numerics.Refusal.t) result
(** A kernel's precision chosen by the caller rather than by the policy: [F32]
    on a program {!Loop_numerics.admit} refuses is an error, and the kernel is
    binary32 whether or not anything vectorized. For tests and scalar emission.
    The recorded policy is [Simd_fp32_ordered] for [F32]. *)

val oracle : t -> Loop_program.t -> Loop_program.t
(** The scalar program the plan's kernel is defined to compute: the plan's
    vector program expanded lane by lane ({!Loop_vector_expand}), scheduled sums
    spelled out as the accumulator statements their definition gives, or the
    program itself when nothing was vectorized. Run by {!Loop_interp} at the
    plan's precision it is the independent answer a generated kernel is checked
    against, bit for bit. *)
