(** The process route of a model bundle: each invocation's generic program is
    selected, allocated, framed and published, then lowered through typed Rivet
    modules into an image of its own. The allocator instances the implementation
    applies are not exported. *)

module Art = Machine_model.Mir_artifact
module Route = Machine_model.Mir_model_route
module M = Rivet_a64_module
module I = Rivet_a64_image

(** Which allocator makes the physical program. *)
module Allocation : sig
  type t =
    | Reference  (** the stack-based reference allocator *)
    | Scanned  (** sink scheduling, then split linear scan *)

  val name : t -> string
end

type published = {
  artifact :
    (Machine_target_aarch64.A64_op.t, Machine_target_aarch64.A64_op.test) Art.t;
  record : Machine_ir.Mir_id.Region.t;
}

val publish :
  allocation:Allocation.t ->
  sites:Machine_ir.Mir_failure.Site_entry.t array ->
  Machine_ir.Mir_verify.Generic.t ->
  (published, string) result
(** Selection, allocation, frame realization and publication. *)

val region_symbol : Machine_ir.Mir_id.Region.t -> string

val regions_program :
  ('a, 'b) Machine_ir.Mir_phys.Program.t -> ('c, 'd) Machine_ir.Mir_program.t

val modules :
  ?mutation:M.F.Mutation.t ->
  ?binding:M.binding ->
  runtime:Machine_rivet_common.Rivet_runtime.t ->
  (Machine_target_aarch64.A64_op.t, Machine_target_aarch64.A64_op.test) Art.t ->
  (string * M.F.I.t M.N.module_ list, string) result

val load :
  entry:string ->
  Aarch64_encode.Instruction.t Asm_core.Normalized_ast.module_ list ->
  (I.t, string) result

val image :
  ?mutation:M.F.Mutation.t ->
  runtime:Machine_rivet_common.Rivet_runtime.t ->
  (Machine_target_aarch64.A64_op.t, Machine_target_aarch64.A64_op.test) Art.t ->
  (I.t, string) result

type kernel = { loaded : I.t }

val kernel : I.t -> kernel

val exec_of :
  ?mutation:M.F.Mutation.t ->
  ?check:(entry:string -> M.F.I.t M.N.module_ list -> (unit, string) result) ->
  allocation:Allocation.t ->
  runtime:Machine_rivet_common.Rivet_runtime.t ->
  sites:Machine_ir.Mir_failure.Site_entry.t array ->
  Machine_ir.Mir_verify.Generic.t ->
  (Route.Exec.t, string) result

val route :
  ?mutation:M.F.Mutation.t ->
  ?check:(entry:string -> M.F.I.t M.N.module_ list -> (unit, string) result) ->
  ?allocation:Allocation.t ->
  ?runtime:Rivet_a64_runtime.t ->
  unit ->
  Route.Route.t
