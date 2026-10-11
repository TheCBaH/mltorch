module W = Wire

type effect =
  | Metadata_assertion
  | Pure
  | Unsupported of Prose.Description.t

type t = {
  node : Node_ref.t;
  operator : Operator_ref.t;
  operands : Tensor_ref.t list;
  results : Tensor_ref.t list;
  control_predecessors : Node_ref.t list;
  effect : effect;
}

module Description = W.Text (Prose.Description)

let decode_effect : effect W.decoder =
  W.variant (fun tag o ->
      let ( let* ) = W.( let* ) in
      match tag with
      | "metadata_assertion" -> Ok Metadata_assertion
      | "pure" -> Ok Pure
      | "unsupported" ->
          let* d = W.Obj.required o "description" Description.decode in
          Ok (Unsupported d)
      | tag -> W.unknown_tag ~kind:"effect" tag)

let encode_effect = function
  | Metadata_assertion -> W.tagged "metadata_assertion" []
  | Pure -> W.tagged "pure" []
  | Unsupported d ->
      W.tagged "unsupported" [ W.field "description" Description.encode d ]

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* node = W.Obj.required o "node" Node_ref.decode in
      let* operator = W.Obj.required o "operator" Operator_ref.decode in
      let* operands = W.Obj.required o "operands" (W.list Tensor_ref.decode) in
      let* results = W.Obj.required o "results" (W.list Tensor_ref.decode) in
      let* control_predecessors =
        W.Obj.required o "control_predecessors" (W.list Node_ref.decode)
      in
      let* effect = W.Obj.required o "effect" decode_effect in
      Ok { node; operator; operands; results; control_predecessors; effect })

let encode n =
  W.obj
    [
      W.field "node" Node_ref.encode n.node;
      W.field "operator" Operator_ref.encode n.operator;
      W.field "operands" (W.enc_list Tensor_ref.encode) n.operands;
      W.field "results" (W.enc_list Tensor_ref.encode) n.results;
      W.field "control_predecessors" (W.enc_list Node_ref.encode)
        n.control_predecessors;
      W.field "effect" encode_effect n.effect;
    ]
