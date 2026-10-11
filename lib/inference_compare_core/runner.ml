(* The ordinary graph-runner interface the coordinator calls. It is an
   integration surface, not a requirement on an engine's internals: [Graph.t]
   is the engine's existing graph type, [Value.t] whatever handle it already
   uses. A runner sees a self-contained graph and its inputs, nothing about
   clusters, policies or donors. *)

open Inference_compare_schema
module L = Inference_compare_codecs.Logical_value

module type GRAPH_RUNNER = sig
  module Graph : sig
    type t
  end

  module Prepared : sig
    type t
  end

  module Value : sig
    type t
  end

  module Error : sig
    type t

    val pp : Format.formatter -> t -> unit

    val is_resource_exhaustion : t -> bool
    (** Whether the failure was memory or storage exhaustion, which the
        coordinator reports as such rather than as a transfer or execution
        failure. *)
  end

  val identity : unit -> Engine_identity.t

  val prepare : Graph.t -> (Prepared.t, Error.t) Err.t
  (** Once per extracted graph and configuration. *)

  val import :
    L.Context.t ->
    Prepared.t ->
    Tensor_spec.t ->
    L.t ->
    (Value.t, Error.t) Err.t
  (** Copies or anchors the logical value; the result owns its storage for the
      whole run. *)

  val export : L.Context.t -> Value.t -> (L.t, Error.t) Err.t
  (** An owned, immutable snapshot: later runs cannot change it. *)

  val run :
    Prepared.t ->
    inputs:(Tensor_ref.t * Value.t) list ->
    ((Tensor_ref.t * Value.t) list, Error.t) Err.t
  (** Returns when every declared output is available (an asynchronous engine
      waits on its completion token first). Ids are the extracted graph's. *)

  val release_value : Value.t -> unit
  val close : Prepared.t -> unit
end
