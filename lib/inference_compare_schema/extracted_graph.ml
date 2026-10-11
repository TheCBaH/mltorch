module W = Wire
module Unit_id = W.Text (Id.Unit)

type t = {
  source_graph : Graph_key.t;
  extracted_key : Graph_key.t;
  unit : Id.Unit.t;
  inputs : Tensor_ref.t list;
  outputs : Tensor_ref.t list;
  original_to_extracted_nodes : (Node_ref.t * Node_ref.t) list;
  original_to_extracted_tensors : (Tensor_ref.t * Tensor_ref.t) list;
  graph_payload_digest : Digest.Extracted_graph_payload.t;
}

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let tensors = W.list Tensor_ref.decode in
      let* source_graph = W.Obj.required o "source_graph" Graph_key.decode in
      let* extracted_key = W.Obj.required o "extracted_key" Graph_key.decode in
      let* unit = W.Obj.required o "unit" Unit_id.decode in
      let* inputs = W.Obj.required o "inputs" tensors in
      let* outputs = W.Obj.required o "outputs" tensors in
      let* original_to_extracted_nodes =
        W.Obj.required o "original_to_extracted_nodes"
          (W.list (W.pair Node_ref.decode Node_ref.decode))
      in
      let* original_to_extracted_tensors =
        W.Obj.required o "original_to_extracted_tensors"
          (W.list (W.pair Tensor_ref.decode Tensor_ref.decode))
      in
      let* graph_payload_digest =
        W.Obj.required o "graph_payload_digest"
          Wire_digest.extracted_graph_payload
      in
      Ok
        {
          source_graph;
          extracted_key;
          unit;
          inputs;
          outputs;
          original_to_extracted_nodes;
          original_to_extracted_tensors;
          graph_payload_digest;
        })

let encode g =
  let tensors = W.enc_list Tensor_ref.encode in
  W.obj
    [
      W.field "source_graph" Graph_key.encode g.source_graph;
      W.field "extracted_key" Graph_key.encode g.extracted_key;
      W.field "unit" Unit_id.encode g.unit;
      W.field "inputs" tensors g.inputs;
      W.field "outputs" tensors g.outputs;
      W.field "original_to_extracted_nodes"
        (W.enc_list (W.enc_pair Node_ref.encode Node_ref.encode))
        g.original_to_extracted_nodes;
      W.field "original_to_extracted_tensors"
        (W.enc_list (W.enc_pair Tensor_ref.encode Tensor_ref.encode))
        g.original_to_extracted_tensors;
      W.field "graph_payload_digest" Wire_digest.encode_extracted_graph_payload
        g.graph_payload_digest;
    ]
