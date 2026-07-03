(* Public encode/decode helpers for native graphs and tensors.
   Codecs live in [Graph_ir.graph_jsont] and [Tensor.jsont]; this module
   just wires them through [Jsont_bytesrw]. *)

val encode_graph :
  ?format:Jsont.format -> Graph_ir.graph -> (string, string) result

val decode_graph : string -> (Graph_ir.graph, string) result

val encode_tensor :
  ?format:Jsont.format ->
  ?max_elts:int ->
  Tensor.packed ->
  (string, string) result
(** [~max_elts]: if provided and tensor has more elements, the payload data is
    elided; otherwise the full array payload is emitted. *)

val decode_tensor : string -> (Tensor.packed, string) result
(** Errors when the payload data was elided. *)
