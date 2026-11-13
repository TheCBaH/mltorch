(** One invocation's generic program made executable on a route: the generic
    interpreter, or a target's selected program — as selected or scheduled — on
    its semantic interpreter, or that program reference-allocated, verified
    physically and checked symbolically, on the physical interpreter — or
    realized with its frames and published as an artifact, which re-verifies and
    re-checks it, and only then run. A checker rejection is a compiler error:
    the kernel is refused, never run. The per-target instances the
    implementation applies are not exported. *)

open Machine_ir
open Machine_interp

module Stage : sig
  type t =
    | Allocated  (** reference allocation *)
    | Realized  (** reference allocation, frames, publication *)
    | Scanned
        (** sink scheduling, linear-scan allocation, frames, publication *)
    | Scheduled of Machine_alloc.Mir_schedule.Policy.t
    | Selected

  val name : t -> string
end

(** What a context needs of a kernel: its storage, and a run's status. *)
module Exec : sig
  type t = {
    instantiate :
      Mir_memory.t ->
      shared:(Mir_id.Region.t -> Mir_memory.Key.t option) ->
      (Mir_interp.Binding.t, string) result;
    run :
      ?fuel:int64 ->
      Mir_memory.t ->
      Mir_interp.Binding.t ->
      invocation:int32 ->
      Mir_observation.Status.t;
    artifact : (Format.formatter -> unit) option;
        (** the published artifact's summary, on the realized stage *)
    traffic : unit -> Mir_phys_interp.Traffic.t option;
        (** what this kernel's runs so far executed of the allocation's making,
            on an allocated stage *)
  }
end

(** A route the caller supplies: a back end this library does not link (native
    code), made executable from an invocation's generic program. *)
module Custom : sig
  type t = {
    name : string;
    plan : Ssa_ir.Ssa_program.t -> (Ssa_ir.Ssa_program.t, string) result;
        (** Final implementation decisions, before generic MIR lowering. *)
    exec :
      sites:Mir_failure.Site_entry.t array ->
      Mir_verify.Generic.t ->
      (Exec.t, string) result;
  }
end

module Route : sig
  type t =
    | Aarch64 of Stage.t
    | Custom of Custom.t
    | Generic
    | Riscv32 of Stage.t
    | Riscv32_grouped of Stage.t
        (** sixteen-lane vectors in groups of four registers (LMUL 4); only the
            scanned allocator places them *)
    | X86_64 of Stage.t

  val name : t -> string
end

val finalize :
  Route.t -> Ssa_ir.Ssa_program.t -> (Ssa_ir.Ssa_program.t, string) result

val instantiate :
  ('a, 'b) Mir_program.t ->
  Mir_memory.t ->
  shared:(Mir_id.Region.t -> Mir_memory.Key.t option) ->
  (Mir_interp.Binding.t, string) result

val regions_program : ('a, 'b) Mir_phys.Program.t -> ('c, 'd) Mir_program.t

val models_of : Mir_verify.Generic.t -> Mir_helper_model.t list
(** The helper models a program's interpreters run: the math helpers, and one
    per contraction it declares. *)

(** Seconds spent in each back-end stage of one kernel, and what its published
    artifact holds; the clock is the caller's. *)
module Timing : sig
  type t = {
    select : float;
    schedule : float;  (** sink scheduling *)
    allocate : float;  (** reference allocation *)
    scan : float;  (** linear-scan allocation of the scheduled program *)
    realize : float;
    publish : float;  (** re-verification and re-checking included *)
    relocations : int;
    symbols : int;
  }
end

val generic : Mir_verify.Generic.t -> Exec.t

val measure :
  Route.t ->
  now:(unit -> float) ->
  sites:Mir_failure.Site_entry.t array ->
  Mir_verify.Generic.t ->
  (Timing.t, string) result
(** A target route's stage timings for one kernel; the generic route has no back
    end. *)

val pressure :
  Route.t ->
  sites:Mir_failure.Site_entry.t array ->
  Mir_verify.Generic.t ->
  (Machine_alloc.Mir_pressure.t, string) result
(** A target route's pressure for one kernel, whatever its stage. *)

val exec :
  Route.t ->
  sites:Mir_failure.Site_entry.t array ->
  Mir_verify.Generic.t ->
  (Exec.t, string) result
