(* Public encode/decode helpers for native graphs and tensors.
   Codecs live in [Graph_ir.graph_jsont] and [Tensor.jsont]; this module
   just wires them through [Jsont_bytesrw]. *)

type error = [ `Jsont of string ]
(** Jsont's own message — the decoder position and expectation it produced. A
    THIRD-PARTY payload, named for its source: this module does not author it
    and cannot classify it further, so the tag says where it came from rather
    than pretending to a case of its own.

    Inside [Err.t] rather than a bare [(_, string) result] so callers compose
    with [let*] and get a wrapper carrying provenance; the four entry points
    below used to make each caller re-match. *)

val pp_error : Format.formatter -> [< error ] -> unit

val encode_graph :
  ?format:Jsont.format -> Graph_ir.graph -> (string, [> error ]) Err.t

val decode_graph : string -> (Graph_ir.graph, [> error ]) Err.t

val encode_tensor :
  ?format:Jsont.format ->
  ?max_elts:int ->
  Tensor.packed ->
  (string, [> error ]) Err.t
(** [~max_elts]: if provided and tensor has more elements, the payload data is
    elided; otherwise the full array payload is emitted. *)

val decode_tensor : string -> (Tensor.packed, [> error ]) Err.t
(** Errors when the payload data was elided. *)
