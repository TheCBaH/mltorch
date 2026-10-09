(** The producer's [captures.json]: an inventory of every captured tensor of the
    graph -- parameters, buffers and constant tensors, including those no node
    reads -- with the dtype, shape and value digest each must have. It is
    evidence about the graph, decoded independently of the map so the two can be
    compared; its own consistency is not assumed. *)

module Capture : sig
  type t = {
    dtype : Dtype.t;
    kind : Fault.capture_kind;
    live : bool;  (** Some node still reads the value. *)
    referenced : bool;
    scalar : bool;
    shape : int64 list;
    target : string;  (** The weights/constants config key. *)
    value_sha256 : Pt2_sha256.Digest.t;
  }
end

type t = {
  artifact_id : string;
  captures : Capture.t list;  (** Document order. *)
  graph_sha256 : Pt2_sha256.Digest.t;
}

val of_string : ?limits:Limits.t -> string -> (t, [> Fault.error ]) Err.t
(** Duplicate targets are rejected. Unrecognized members (the producer adds
    descriptive ones) are skipped. *)
