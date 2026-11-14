(** The process route of a model bundle: each invocation's generic program is
    selected, allocated, framed and published, then lowered through typed Rivet
    modules into an image of its own, laid out once. A run copies the
    invocation's storage into the image's data block, executes it as a static
    process ({!Rivet_x64_process.runner}), and copies the results back. The
    allocator instances the implementation applies are not exported. *)

module Art = Machine_model.Mir_artifact
module Route = Machine_model.Mir_model_route

(** Which allocator makes the physical program. *)
module Allocation : sig
  type t =
    | Reference  (** the stack-based reference allocator *)
    | Scanned  (** sink scheduling, then split linear scan *)

  val name : t -> string
end

type published = {
  artifact :
    (Machine_target_x86_64.X64_op.t, Machine_target_x86_64.X64_op.test) Art.t;
  record : Machine_ir.Mir_id.Region.t;
}

val publish :
  ?features:Machine_ir.Mir_target.Feature.t list ->
  allocation:Allocation.t ->
  sites:Machine_ir.Mir_failure.Site_entry.t array ->
  Machine_ir.Mir_verify.Generic.t ->
  (published, string) result
(** Selection, allocation, frame realization and publication. *)

val regions_program :
  ('a, 'b) Machine_ir.Mir_phys.Program.t -> ('c, 'd) Machine_ir.Mir_program.t

val modules :
  ?entry_mutation:Rivet_x64_module.Entry_mutation.t ->
  ?binding:Rivet_x64_module.binding ->
  runtime:Machine_rivet_common.Rivet_runtime.t ->
  (Machine_target_x86_64.X64_op.t, Machine_target_x86_64.X64_op.test) Art.t ->
  ( string * X86_64_encode.Instruction.t Rivet_x64_module.N.module_ list,
    string )
  result

val load :
  entry:string ->
  X86_64_encode.Instruction.t Asm_core.Normalized_ast.module_ list ->
  (Rivet_x64_image.t, string) result

val exec_of :
  allocation:Allocation.t ->
  runtime:Machine_rivet_common.Rivet_runtime.t ->
  sites:Machine_ir.Mir_failure.Site_entry.t array ->
  Machine_ir.Mir_verify.Generic.t ->
  (Route.Exec.t, string) result

val route :
  ?allocation:Allocation.t ->
  ?runtime:Machine_rivet_common.Rivet_runtime.t ->
  unit ->
  Route.Route.t
