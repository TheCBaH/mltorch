(** One endpoint as the coordinator drives it: a packed ordinary graph runner
    plus the dialect step that builds its graph from a planned extraction. The
    engine's graph, handle and error types stay hidden behind closures; the
    coordinator sees logical values, extracted tensor ids and rendered failures.
    A handle imported or produced here is released by the closure that created
    it, on every path. *)

open Inference_compare_schema

type endpoint =
  | Endpoint : {
      runner :
        (module Runner.GRAPH_RUNNER with type Graph.t = 'g and type Error.t = 'e);
      build :
        Inference_compare_extract.Extraction.t ->
        ('g, Inference_compare_extract.Extract_error.t) Err.t;
    }
      -> endpoint

type error = { phase : Phase.t; code : Failure.code; text : string }
(** [text] renders the engine's own error; it never selects [code]. *)

type imported = {
  readback : (Tensor_ref.t * Inference_compare_codecs.Logical_value.t) list;
      (** The imported handles exported back, in extracted ids, owned by the
          caller. *)
  run :
    unit ->
    ( (Tensor_ref.t * Inference_compare_codecs.Logical_value.t) list,
      error )
    result;
      (** Executes the graph; owned snapshots of every declared output in the
          engine's returned order, extracted ids. Output handles are released
          before returning. *)
  release : unit -> unit;
      (** Releases the imported input handles; idempotent. *)
}

type t = {
  identity : Engine_identity.t;
  prepare : Inference_compare_extract.Extraction.t -> (unit, error) result;
  import :
    Id.Unit.t ->
    (Tensor_ref.t * Inference_compare_codecs.Logical_value.t) list ->
    (imported, error) result;
  close : unit -> unit;  (** Closes every prepared graph; idempotent. *)
}

val make : Inference_compare_codecs.Logical_value.Context.t -> endpoint -> t
