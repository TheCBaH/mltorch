module W = Wire

type t = {
  nodes : Node_ref.t list;
  inputs : Tensor_ref.t list;
  outputs : Tensor_ref.t list;
}

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* nodes = W.Obj.required o "nodes" (W.list Node_ref.decode) in
      let* inputs = W.Obj.required o "inputs" (W.list Tensor_ref.decode) in
      let* outputs = W.Obj.required o "outputs" (W.list Tensor_ref.decode) in
      Ok { nodes; inputs; outputs })

let encode c =
  W.obj
    [
      W.field "nodes" (W.enc_list Node_ref.encode) c.nodes;
      W.field "inputs" (W.enc_list Tensor_ref.encode) c.inputs;
      W.field "outputs" (W.enc_list Tensor_ref.encode) c.outputs;
    ]
