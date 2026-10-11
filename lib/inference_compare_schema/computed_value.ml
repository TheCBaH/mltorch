module W = Wire
module Instance = W.Text (Id.Value_instance)

type t = {
  instance_id : Id.Value_instance.t;
  origin : Value_origin.t;
  spec : Tensor_spec.t;
  logical_digest : Digest.Logical_value.t;
  retained_payload_digest : Digest.Payload.t option;
}

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* instance_id = W.Obj.required o "instance_id" Instance.decode in
      let* origin = W.Obj.required o "origin" Value_origin.decode in
      let* spec = W.Obj.required o "spec" Tensor_spec.decode in
      let* logical_digest =
        W.Obj.required o "logical_digest" Wire_digest.logical_value
      in
      let* retained_payload_digest =
        W.Obj.optional o "retained_payload_digest" Wire_digest.payload
      in
      Ok { instance_id; origin; spec; logical_digest; retained_payload_digest })

let encode v =
  W.obj
    [
      W.field "instance_id" Instance.encode v.instance_id;
      W.field "origin" Value_origin.encode v.origin;
      W.field "spec" Tensor_spec.encode v.spec;
      W.field "logical_digest" Wire_digest.encode_logical_value v.logical_digest;
      W.field "retained_payload_digest"
        (W.enc_option Wire_digest.encode_payload)
        v.retained_payload_digest;
    ]
