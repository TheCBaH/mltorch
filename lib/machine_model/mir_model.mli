(** Whole-model bundles through generic Machine IR. Each invocation's placed
    kernel is lowered to SSA by the bundle path's own producer
    ({!Ssa_backends.program}) and then to generic Machine IR; the bundle's
    schedule, its argument convention (every program buffer, positionally, bound
    to an edge) and its failure-record invocation numbering are kept.

    A {!Context} owns the storage: one memory instance per graph tensor, which
    persists across the context's calls, so a kernel that read a tensor it had
    not been given — or its own output before writing it — would see bytes of an
    earlier call. Neither is allowed to: before each invocation its output
    tensors' bytes become undefined, and the interpreter reports a read of an
    undefined byte as a defect. Each invocation's own scratch (locals, the scan
    meter, parameter tables) is fresh for that invocation. *)

open Graph_ir
open Machine_ir

(** Why an invocation has no Machine IR kernel. *)
module Reason : sig
  type t =
    | Lowering of Machine_lower.Mir_lower.Refusal.t
    | Route of string
        (** the route's selection refusal, or its allocation's rejection *)
    | Source of string  (** the SSA producer's own refusal *)

  val pp : Format.formatter -> t -> unit
end

module Refusal : sig
  type t = { invocation : int32; node : Node_id.t; reason : Reason.t }
  (** [invocation]: the position in the schedule, as a failure record numbers it
  *)

  val pp : Format.formatter -> t -> unit
end

module Route = Mir_model_route.Route
(** Where a kernel runs: the generic interpreter, or a target's selected
    program, or that program reference-allocated, verified and checked. *)

module Stage = Mir_model_route.Stage

val sites : Loop_ir.Loop_bundle.invocation -> Mir_failure.Site_entry.t array
(** An invocation's failure-site table as a record decodes it: its Loop
    program's table, of which only local and scan entries name anything. *)

type t

val prepare :
  ?route:Route.t ->
  ?blocking:Mir_blocking.Policy.t ->
  pipeline:Ssa_backends.Pipeline.t ->
  Loop_ir.Loop_bundle.t ->
  (t, Refusal.t list) result
(** Every invocation lowered (and made executable on [route], [Generic] by
    default, its failure sites bound against the invocation's Loop table), or
    every refusal: a [Planned] pipeline is refused for each invocation (no
    binary32 or vector slice is admitted here). [blocking] ([Unblocked] by
    default) needs the [Exact] pipeline; [Feedback] measures through the route's
    target, and on the generic route, which has none, stays unblocked. *)

val invocations : t -> int
(** How many invocations the schedule holds. *)

val blocking : t -> (Node_id.t * Mir_blocking.Decision.t) list
(** Each feedback decision, by the node of its invocation, in schedule order. *)

val generic : t -> Mir_verify.Generic.t list
(** Each invocation's generic program, in schedule order. *)

(** Why a call stopped. *)
module Stop : sig
  type t =
    | At of {
        invocation : int32;
        node : Node_id.t;
        status : Mir_observation.Status.t;
      }  (** the first invocation that did not succeed, and how *)
    | Bind of string  (** a region the interpreter could not bind *)
    | Missing of Tensor_id.t  (** a tensor the call was not given *)
    | Undefined_output of Tensor_id.t
        (** a graph output with a byte no invocation wrote *)

  val pp : Format.formatter -> t -> unit
end

module Context : sig
  type model := t
  type t

  val create :
    model ->
    constants:(Tensor_id.t -> Tensor.packed option) ->
    (t, Stop.t) result
  (** Storage for every constant, written once. *)

  val run :
    ?fuel:int64 ->
    t ->
    inputs:(Tensor_id.t -> Tensor.packed option) ->
    (Tensor.packed list, Stop.t) result
  (** One call: the inputs written, every invocation run in schedule order on
      the model's route with the math helpers' models, and the graph's outputs
      read back in order. [fuel] bounds each invocation. *)

  val run_prefix :
    ?fuel:int64 ->
    t ->
    inputs:(Tensor_id.t -> Tensor.packed option) ->
    count:int ->
    (Tensor_id.t list, Stop.t) result
  (** An explicit invocation sequence: the inputs written and the first [count]
      invocations run; the tensors they wrote, in schedule order. *)

  val tensor : t -> Tensor_id.t -> (Tensor.packed, Stop.t) result
  (** A tensor's current bytes, decoded by its signature. *)
end
