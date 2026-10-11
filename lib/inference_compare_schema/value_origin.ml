module W = Wire
module Canonical = W.Text (Id.Canonical_value)
module Unit_id = W.Text (Id.Unit)
module Instance = W.Text (Id.Value_instance)
module Recipe_id = W.Text (Id.Recipe)

type t =
  | Computed of { side : Side.t; unit : Id.Unit.t; tensor : Tensor_ref.t }
  | Converted of { parents : Id.Value_instance.t list; recipe : Id.Recipe.t }
  | Initial of { payload_digest : Digest.Payload.t; key : Id.Canonical_value.t }

let decode : t W.decoder =
  W.variant (fun tag o ->
      let ( let* ) = W.( let* ) in
      match tag with
      | "computed" ->
          let* side = W.Obj.required o "side" Side.decode in
          let* unit = W.Obj.required o "unit" Unit_id.decode in
          let* tensor = W.Obj.required o "tensor" Tensor_ref.decode in
          Ok (Computed { side; unit; tensor })
      | "converted" ->
          let* parents = W.Obj.required o "parents" (W.list Instance.decode) in
          let* recipe = W.Obj.required o "recipe" Recipe_id.decode in
          Ok (Converted { parents; recipe })
      | "initial" ->
          let* payload_digest =
            W.Obj.required o "payload_digest" Wire_digest.payload
          in
          let* key = W.Obj.required o "key" Canonical.decode in
          Ok (Initial { payload_digest; key })
      | tag -> W.unknown_tag ~kind:"value_origin" tag)

let encode = function
  | Computed { side; unit; tensor } ->
      W.tagged "computed"
        [
          W.field "side" Side.encode side;
          W.field "unit" Unit_id.encode unit;
          W.field "tensor" Tensor_ref.encode tensor;
        ]
  | Converted { parents; recipe } ->
      W.tagged "converted"
        [
          W.field "parents" (W.enc_list Instance.encode) parents;
          W.field "recipe" Recipe_id.encode recipe;
        ]
  | Initial { payload_digest; key } ->
      W.tagged "initial"
        [
          W.field "payload_digest" Wire_digest.encode_payload payload_digest;
          W.field "key" Canonical.encode key;
        ]
