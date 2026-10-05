(** Executing a lowered program over bound tensors, with the contract of
    [Kernel_eval.run_plan]: every caller-bound input is validated, in input
    order, before anything runs; an output is fresh and zero-filled; the result
    holds exactly the stored outputs. Failures are the rows the reference
    reports, never a translation table between two vocabularies. *)

type error =
  [ Ssa_ir.Ssa_interp.failure
  | `Binding_mismatch of Kernel_eval.Binding_mismatch.t
  | `Invalid_program of Ssa_ir.Ssa_verify.diagnostic
  | `Unbound_input of Tensor_id.t ]

val pp_error : Format.formatter -> [< error ] -> unit

val run :
  ?counters:Ssa_ir.Ssa_interp.Counters.t ->
  Fusion_plan.t ->
  Ssa_ir.Ssa_program.t ->
  bind:(Tensor_id.t -> Tensor.packed option) ->
  (Tensor.packed Tensor_id.Map.t, error) Err.t
