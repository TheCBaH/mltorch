(* Public encode/decode API for native graphs and tensors.  The codecs
   themselves live in [Graph_ir] (graph/op/node) and [Tensor] (packed tensors);
   this module is a thin shell that routes them through [Jsont_bytesrw]. *)

let encode_graph ?(format = Jsont.Minify) g =
  Jsont_bytesrw.encode_string ~format Graph_ir.graph_jsont g

let decode_graph s = Jsont_bytesrw.decode_string Graph_ir.graph_jsont s

(* [~max_elts]: if provided and the tensor has strictly more elements, the
   payload is encoded as {"None":null} instead of {"Array":[...]}. *)
let encode_tensor ?(format = Jsont.Minify) ?max_elts packed =
  Jsont_bytesrw.encode_string ~format (Tensor.jsont ?max_elts ()) packed

(* Returns [Ok (Tensor _)] when data is present; errors when it was elided
   ({"None":null} — re-encode without [~max_elts] to get data back). *)
let decode_tensor s = Jsont_bytesrw.decode_string (Tensor.jsont ()) s
