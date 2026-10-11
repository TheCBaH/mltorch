(* The lowered Native graph as the Native side sees it: an ordinary graph of
   tensor dataflow (tensors are numbered edges, nodes are numbered), with the
   complete record of which Native nodes each exporter node produced. The
   description carries operators and dataflow; each op's parameters are
   folded into the extraction digest through {!node_attributes}. *)

open Inference_compare_schema
module Tid = Graph_ir.Tensor_id
module Nid = Graph_ir.Node_id

type t = {
  lowered : Pt2_native_graph.t;
  capture : Native_interp.Capture.t;
  graph : Graph_ir.graph;
  key : Graph_key.t;
  desc : Graph_desc.t;
}

let ( let* ) = Result.bind
let text = Graph_model.text
let tensor_local = Graph_model.tensor_local
let node_local = Graph_model.node_local
let tensor_ref = Graph_model.tensor_ref
let node_ref = Graph_model.node_ref
let spec_of_sig = Graph_model.spec_of_sig
let sig_of = Graph_model.sig_of

let describe ~key ~scope ~(g : Graph_ir.graph) ~nodes ~inputs ~outputs ~roles =
  Graph_model.describe ~ops:Graph_model.native_ops ~key ~scope ~g ~nodes ~inputs
    ~outputs ~roles

let of_lowered ~hashers ~algorithm (lowered : Pt2_native_graph.t) capture :
    (t, string) result =
  let g = lowered.Pt2_native_graph.graph in
  let* graph_text = Jsont_bytesrw.encode_string Graph_ir.graph_jsont g in
  let* hex =
    match
      Inference_compare_codecs.Payload_hasher.digest_string hashers algorithm
        graph_text
    with
    | Ok d -> Ok (Digest.Payload.hex d)
    | Error _ -> Error "digest backend unavailable"
  in
  let* graph_digest =
    Digest.Graph.make algorithm ~hex |> Result.map_error (fun _ -> "digest")
  in
  let* dialect = text (Id.Dialect.of_string "native") in
  let* version = text (Id.Graph_version.of_string "pt2-lowered") in
  let key = { Graph_key.dialect; version; graph_digest } in
  let* desc =
    describe ~key ~scope:[] ~g ~nodes:g.Graph_ir.Graph.nodes ~inputs:g.inputs
      ~outputs:g.outputs ~roles:`Graph
  in
  Ok { lowered; capture; graph = g; key; desc }

(* The text standing for one node in a digest: its operation and every
   parameter, which the dataflow description does not carry. *)
let node_attributes (n : Graph_ir.node) =
  match Jsont_bytesrw.encode_string Graph_ir.op_jsont n.op with
  | Ok s -> s
  | Error m -> m

let attributes_of t (r : Node_ref.t) =
  let s = Id.Node_local.to_string r.local_id in
  match int_of_string_opt (String.sub s 1 (String.length s - 1)) with
  | None -> ""
  | Some i -> (
      match
        List.find_opt
          (fun (n : Graph_ir.node) -> Graph_ir.Node_id.to_int n.id = i)
          t.graph.nodes
      with
      | Some n -> node_attributes n
      | None -> "")
