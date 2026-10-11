module W = Wire
module Canonical = W.Text (Id.Canonical_value)
module Node_id = W.Text (Id.Conversion_node)
module Recipe_id = W.Text (Id.Recipe)

type argument =
  | Boundary of Tensor_ref.t
  | Canonical of Id.Canonical_value.t
  | Conversion_result of Id.Conversion_node.t

type node = {
  id : Id.Conversion_node.t;
  operation : Conversion_step.t;
  arguments : argument list;
}

type t = {
  id : Id.Recipe.t;
  nodes : node list;
  result : argument;
  output_spec : Tensor_spec.t;
  lossless : bool;
}

let decode_argument : argument W.decoder =
  W.variant (fun tag o ->
      let ( let* ) = W.( let* ) in
      match tag with
      | "boundary" ->
          let* r = W.Obj.required o "tensor" Tensor_ref.decode in
          Ok (Boundary r)
      | "canonical" ->
          let* c = W.Obj.required o "key" Canonical.decode in
          Ok (Canonical c)
      | "conversion_result" ->
          let* n = W.Obj.required o "node" Node_id.decode in
          Ok (Conversion_result n)
      | tag -> W.unknown_tag ~kind:"recipe_argument" tag)

let encode_argument = function
  | Boundary r -> W.tagged "boundary" [ W.field "tensor" Tensor_ref.encode r ]
  | Canonical c -> W.tagged "canonical" [ W.field "key" Canonical.encode c ]
  | Conversion_result n ->
      W.tagged "conversion_result" [ W.field "node" Node_id.encode n ]

let decode_node : node W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* id = W.Obj.required o "id" Node_id.decode in
      let* operation = W.Obj.required o "operation" Conversion_step.decode in
      let* arguments = W.Obj.required o "arguments" (W.list decode_argument) in
      Ok { id; operation; arguments })

let encode_node (n : node) =
  W.obj
    [
      W.field "id" Node_id.encode n.id;
      W.field "operation" Conversion_step.encode n.operation;
      W.field "arguments" (W.enc_list encode_argument) n.arguments;
    ]

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* id = W.Obj.required o "id" Recipe_id.decode in
      let* nodes =
        W.Obj.required o "nodes"
          (W.unique
             ~key:(fun (n : node) -> Id.Conversion_node.to_string n.id)
             (W.list decode_node))
      in
      let* result = W.Obj.required o "result" decode_argument in
      let* output_spec = W.Obj.required o "output_spec" Tensor_spec.decode in
      let* lossless = W.Obj.required o "lossless" W.bool in
      Ok { id; nodes; result; output_spec; lossless })

let encode r =
  W.obj
    [
      W.field "id" Recipe_id.encode r.id;
      W.field "nodes" (W.enc_list encode_node) r.nodes;
      W.field "result" encode_argument r.result;
      W.field "output_spec" Tensor_spec.encode r.output_spec;
      W.field "lossless" W.enc_bool r.lossless;
    ]
