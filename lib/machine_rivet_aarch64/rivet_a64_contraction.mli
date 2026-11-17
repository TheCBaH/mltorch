(** The contraction helpers an artifact calls, generated as AArch64 text.
    Binary32 [Ordered] contractions only: each product and each sum rounded
    separately in the reference's order, so the bits equal
    {!Contraction.eval}'s, with the seed from zero or from C. Anything else is a
    typed refusal. *)

val module_ :
  ?mutation:Rivet_a64_form.Mutation.t ->
  Contraction.t list ->
  ( Aarch64_encode.Instruction.t Asm_core.Normalized_ast.module_,
    Rivet_a64_refusal.t )
  Err.t

val of_artifact :
  ( Machine_target_aarch64.A64_op.t,
    Machine_target_aarch64.A64_op.test )
  Machine_model.Mir_artifact.t ->
  Contraction.t list
(** The descriptors the artifact's generated functions carry. *)
