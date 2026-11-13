(** The RV32 route of a model bundle: each invocation's generic program is
    selected onto ILP32, allocated, framed and published, then emitted as a
    typed Rivet module and run as one process under qemu-riscv32. The allocator
    instances the implementation applies are not exported. *)

module Art = Machine_model.Mir_artifact
module Route = Machine_model.Mir_model_route
module H = Machine_rivet_riscv32.Rv32_harness
module Profile = Machine_rivet_riscv32.Rv32_profile

(** Which allocator makes the physical program. *)
module Allocation : sig
  type t =
    | Reference  (** the stack-based reference allocator *)
    | Scanned  (** sink scheduling, then split linear scan *)

  val name : t -> string
end

(** How wide a vector register the selection uses: one register of four binary32
    lanes, or an aligned group of four registers of sixteen (LMUL 4), which only
    the scanned allocator places. *)
module Registers : sig
  type t = Grouped | Grouped8 | Single

  val name : t -> string
  val bytes : t -> int64
end

type published = {
  artifact :
    ( Machine_target_riscv32.Rv32_op.t,
      Machine_target_riscv32.Rv32_op.test )
    Art.t;
  record : Machine_ir.Mir_id.Region.t;
}

val publish :
  ?select_mutation:Machine_target_riscv32.Rv32_select.Mutation.t ->
  ?scan_mutation:Machine_alloc.Mir_scan.Mutation.t ->
  ?registers:Registers.t ->
  allocation:Allocation.t ->
  sites:Machine_ir.Mir_failure.Site_entry.t array ->
  Machine_ir.Mir_verify.Generic.t ->
  (published, string) result
(** Selection, allocation, frame realization and publication. *)

val harness_slots :
  record:Machine_ir.Mir_id.Region.t ->
  ('a, 'b) Rv32_mir_module.Table.Art.t ->
  H.slot list

val generated : ('a, 'b) Art.t -> Contraction.t list
val attentions : ('a, 'b) Art.t -> Attention_block.t list

val contraction_engine :
  ('a, 'b) Art.t -> Contraction.t -> Contraction.Select.Engine.t option

val profile : ('a, 'b) Art.t -> Profile.t

val rvv_module :
  ?mutation:Coralnpu_kernels.Rvv_contraction.Mutation.t ->
  ?plan:Coral_transfer.Transfer_plan.Witness.t ->
  Contraction.t ->
  (Riscv32_encode.Instruction.t Asm_core.Normalized_ast.module_, string) result

val helper_modules :
  ?kernel_mutation:Coralnpu_kernels.Rvv_contraction.Mutation.t ->
  ?plan:
    (Contraction.t ->
    (Coral_transfer.Transfer_plan.Witness.t option, string) result) ->
  ?other:
    (Contraction.t ->
    ( Riscv32_encode.Instruction.t Asm_core.Normalized_ast.module_,
      string )
    result
    option) ->
  ?attention:
    (Attention_block.t ->
    ( Riscv32_encode.Instruction.t Asm_core.Normalized_ast.module_,
      string )
    result) ->
  ('a, 'b) Art.t ->
  ( Riscv32_encode.Instruction.t Asm_core.Normalized_ast.module_ list,
    string )
  result

val prepare :
  ?mutation:Rv32_mir_module.F.Mutation.t ->
  ?kernel_mutation:Coralnpu_kernels.Rvv_contraction.Mutation.t ->
  published ->
  (H.prepared, string) result

val regions_program :
  ('a, 'b) Machine_ir.Mir_phys.Program.t -> ('c, 'd) Machine_ir.Mir_program.t

val region_bytes :
  Machine_interp.Mir_memory.t ->
  Machine_interp.Mir_interp.Binding.t ->
  Machine_ir.Mir_id.Region.t ->
  int ->
  (Machine_interp.Mir_memory.Key.t * string, string) result

val run_once :
  ?cpu:string ->
  H.prepared ->
  published:published ->
  Machine_interp.Mir_memory.t ->
  Machine_interp.Mir_interp.Binding.t ->
  (int32, string) result

val exec_of :
  ?mutation:Rv32_mir_module.F.Mutation.t ->
  ?kernel_mutation:Coralnpu_kernels.Rvv_contraction.Mutation.t ->
  ?select_mutation:Machine_target_riscv32.Rv32_select.Mutation.t ->
  ?scan_mutation:Machine_alloc.Mir_scan.Mutation.t ->
  ?registers:Registers.t ->
  ?cpu:string ->
  allocation:Allocation.t ->
  sites:Machine_ir.Mir_failure.Site_entry.t array ->
  Machine_ir.Mir_verify.Generic.t ->
  (Route.Exec.t, string) result

val route :
  ?mutation:Rv32_mir_module.F.Mutation.t ->
  ?kernel_mutation:Coralnpu_kernels.Rvv_contraction.Mutation.t ->
  ?select_mutation:Machine_target_riscv32.Rv32_select.Mutation.t ->
  ?scan_mutation:Machine_alloc.Mir_scan.Mutation.t ->
  ?registers:Registers.t ->
  ?cpu:string ->
  ?allocation:Allocation.t ->
  unit ->
  Route.Route.t
