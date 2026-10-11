module W = Wire
module Unit_id = W.Text (Id.Unit)
module Description = W.Text (Prose.Description)

type t =
  | Created of { node : Node_ref.t; reason : Prose.Description.t }
  | Eliminated of { node : Node_ref.t; reason : Prose.Description.t }
  | Internal_helper of { node : Node_ref.t; owner_unit : Id.Unit.t }
  | Tensor_not_compared of {
      tensor : Tensor_ref.t;
      reason : Prose.Description.t;
    }

let decode : t W.decoder =
  W.variant (fun tag o ->
      let ( let* ) = W.( let* ) in
      match tag with
      | "created" ->
          let* node = W.Obj.required o "node" Node_ref.decode in
          let* reason = W.Obj.required o "reason" Description.decode in
          Ok (Created { node; reason })
      | "eliminated" ->
          let* node = W.Obj.required o "node" Node_ref.decode in
          let* reason = W.Obj.required o "reason" Description.decode in
          Ok (Eliminated { node; reason })
      | "internal_helper" ->
          let* node = W.Obj.required o "node" Node_ref.decode in
          let* owner_unit = W.Obj.required o "owner_unit" Unit_id.decode in
          Ok (Internal_helper { node; owner_unit })
      | "tensor_not_compared" ->
          let* tensor = W.Obj.required o "tensor" Tensor_ref.decode in
          let* reason = W.Obj.required o "reason" Description.decode in
          Ok (Tensor_not_compared { tensor; reason })
      | tag -> W.unknown_tag ~kind:"member_disposition" tag)

let encode = function
  | Created { node; reason } ->
      W.tagged "created"
        [
          W.field "node" Node_ref.encode node;
          W.field "reason" Description.encode reason;
        ]
  | Eliminated { node; reason } ->
      W.tagged "eliminated"
        [
          W.field "node" Node_ref.encode node;
          W.field "reason" Description.encode reason;
        ]
  | Internal_helper { node; owner_unit } ->
      W.tagged "internal_helper"
        [
          W.field "node" Node_ref.encode node;
          W.field "owner_unit" Unit_id.encode owner_unit;
        ]
  | Tensor_not_compared { tensor; reason } ->
      W.tagged "tensor_not_compared"
        [
          W.field "tensor" Tensor_ref.encode tensor;
          W.field "reason" Description.encode reason;
        ]
