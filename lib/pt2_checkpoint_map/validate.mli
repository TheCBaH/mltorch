(** The checks that need the graph but no source file: run them before any host
    I/O, so a map for the wrong graph, or one that leaves a capture unaccounted,
    costs no download.

    The graph side is read three independent ways -- its signature, its two
    payload configs and the producer's [captures.json] -- and the map must agree
    with all of them. None is trusted because another matches it. *)

type graph = {
  artifact_id : string;
      (** From the contract/manifest layer, never from the map under check. *)
  captures : Captures.t;
  constants : Pytorch_weights_config.ModelWeightsConfig.t;
  graph_digest : Pt2_sha256.Digest.t;  (** Of the raw [model.json] bytes. *)
  program : Pytorch_types.ExportedProgram.t;
  weights : Pytorch_weights_config.ModelWeightsConfig.t;
}

val signature_captures :
  Pytorch_types.ExportedProgram.t ->
  ((string * Fault.capture_kind) list, [> Fault.error ]) Err.t
(** Every [PARAMETER], [BUFFER] and [CONSTANT_TENSOR] input spec, unused ones
    included, in signature order. A repeated target is an error. *)

val supported_conversions : Document.t -> (unit, [> Fault.error ]) Err.t
(** Only [Identity], [BF16] to [F32] and [F16] to [F32] are implemented. Any
    other schema-permitted cast is refused here, before a source is opened. *)

val check : Document.t -> graph -> (unit, [> Fault.error ]) Err.t
(** In order: artifact identity, graph digest, signature against
    [captures.json], the map's coverage of exactly the captures, the configs
    against each capture, then per capture the dtype, shape, dense layout and
    value digest, and finally {!supported_conversions}. The first failure is
    reported, in capture-target order. *)
