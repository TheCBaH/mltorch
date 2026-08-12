(* Public encode/decode API for native graphs and tensors.  The codecs
   themselves live in [Graph_ir] (graph/op/node) and [Tensor] (packed tensors);
   this module is a thin shell that routes them through [Jsont_bytesrw]. *)

type error = [ `Jsont of string ]

let pp_error ppf : [< error ] -> unit = function `Jsont m -> Fmt.string ppf m

(* Jsont hands back [('a, string) result] and its message is a THIRD-PARTY
   payload — a decoder position and expectation this module did not author and
   cannot classify further. Lifting it into [Err.t] is still worth doing: it
   gives every caller a wrapper with provenance and lets them compose with
   [let*] instead of re-matching at each of the four entry points.

   [Err.import], not [Err.fail]: this is the inbound boundary, so the event is
   [Import] and there is no [Detect] — Jsont detected the failure, this module
   only carried it across. *)
let of_jsont r = Err.import ~pos:__POS__ (fun m -> `Jsont m) r

let encode_graph ?(format = Jsont.Minify) g =
  of_jsont (Jsont_bytesrw.encode_string ~format Graph_ir.graph_jsont g)

let decode_graph s =
  of_jsont (Jsont_bytesrw.decode_string Graph_ir.graph_jsont s)

(* [~max_elts]: if provided and the tensor has strictly more elements, the
   payload is encoded as {"None":null} instead of {"Array":[...]}. *)
let encode_tensor ?(format = Jsont.Minify) ?max_elts packed =
  of_jsont
    (Jsont_bytesrw.encode_string ~format (Tensor.jsont ?max_elts ()) packed)

(* Returns [Ok (Tensor _)] when data is present; errors when it was elided
   ({"None":null} — re-encode without [~max_elts] to get data back). *)
let decode_tensor s = of_jsont (Jsont_bytesrw.decode_string (Tensor.jsont ()) s)
