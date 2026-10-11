module W = Wire
module Key = W.Text (Id.Canonical_value)
module Payload_id = W.Text (Id.Payload)

type t = {
  key : Id.Canonical_value.t;
  payload_id : Id.Payload.t;
  payload_digest : Digest.Payload.t;
}

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* key = W.Obj.required o "key" Key.decode in
      let* payload_id = W.Obj.required o "payload_id" Payload_id.decode in
      let* payload_digest =
        W.Obj.required o "payload_digest" Wire_digest.payload
      in
      Ok { key; payload_id; payload_digest })

let encode b =
  W.obj
    [
      W.field "key" Key.encode b.key;
      W.field "payload_id" Payload_id.encode b.payload_id;
      W.field "payload_digest" Wire_digest.encode_payload b.payload_digest;
    ]
