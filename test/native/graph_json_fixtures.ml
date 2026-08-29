(* Shared helpers for the Graph_json codec tests split out of what was
   graph_json_test.ml: the [error] type unifying build/JSON
   failures, the encode/decode wrappers that lift into it, shape builders, and
   the small pretty-printers each split file's roundtrip tests format their
   result with. *)

open Graph_ir

type error =
  [ `Build of Graph_builder.error | `Json of string | `Message of string ]

let pp_error ppf : [< error ] -> unit = function
  | `Build e -> Graph_builder.pp_error ppf e
  | `Json msg | `Message msg -> Format.pp_print_string ppf msg

let pp_result pp_ok = Core.Pretty.err_result ~ok:pp_ok ~error:pp_error

let lift_build (r : ('a, Graph_builder.error) Err.t) : ('a, error) Err.t =
  Err.map_error (fun e -> `Build e) r

let lift_json (r : ('a, Graph_json.error) Err.t) : ('a, error) Err.t =
  Err.map_error (fun (`Jsont m) -> `Json m) r

let s n t d h w c = Vec6.shape ~n ~t ~d ~h ~w ~c
let s1c c = s 1 1 1 1 1 c
let pos = Op_config.Pos.of_int
let chan c = Dim.to_int (Vec6.get c Axis.C)

let conv_axis ~kernel ~stride ~pad : Conv.Conv2d.axis_window =
  {
    kernel = Dim.extent kernel;
    stride = Op_config.Pos.of_int stride;
    pad_before = Op_config.Nonneg.of_int pad;
    pad_after = Op_config.Nonneg.of_int pad;
    dilation = Op_config.Pos.of_int 1;
  }

let encode_graph g = lift_json (Graph_json.encode_graph ~format:Jsont.Indent g)
let decode_graph s = lift_json (Graph_json.decode_graph s)

let encode_tensor ?max_elts t =
  lift_json (Graph_json.encode_tensor ~format:Jsont.Indent ?max_elts t)

let decode_tensor s = lift_json (Graph_json.decode_tensor s)

(* [Graph_json] exposes no op codec, so this reaches Jsont directly and lifts
   its bare message the way [Graph_json.of_jsont] does. *)
let encode_op op =
  match
    Jsont_bytesrw.encode_string ~format:Jsont.Indent Graph_ir.op_jsont op
  with
  | Ok v -> Err.return v
  | Error m -> Err.fail (`Json m)

let first_node g =
  match g.Graph.nodes with
  | node :: _ -> Err.return node
  | [] -> Err.fail (`Message "expected graph to contain at least one node")

let pp_graph_json_and_op ppf (graph_json, op_json) =
  Format.fprintf ppf "graph JSON:@.%s@.op JSON:@.%s" graph_json op_json

let pp_json_and_graph ppf (json, g) =
  Format.fprintf ppf "JSON:@.%s@.decoded graph:@.%a" json Graph_ir.pp g

let pp_original_and_graph ppf (g, decoded) =
  Format.fprintf ppf "original:@.%a@.decoded:@.%a" Graph_ir.pp g Graph_ir.pp
    decoded

let pp_decoded_graph ppf g =
  Format.fprintf ppf "decoded graph:@.%a" Graph_ir.pp g

let pp_decoded ppf g = Format.fprintf ppf "decoded:@.%a" Graph_ir.pp g

let pp_original_and_tensor ppf (original, decoded) =
  Format.fprintf ppf "original: %a@.decoded:  %a" Tensor.pp original Tensor.pp
    decoded

let pp_full_and_elided ppf (full_json, elided_json) =
  Format.fprintf ppf "full (8 elts):@.%s@.elided (max_elts=4):@.%s" full_json
    elided_json
