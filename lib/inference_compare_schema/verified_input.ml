module W = Wire
module Canonical = W.Text (Id.Canonical_value)
module Instance = W.Text (Id.Value_instance)

type t = {
  key : Id.Canonical_value.t;
  common_instance : Id.Value_instance.t;
  left_normalized_digest : Digest.Logical_value.t;
  right_normalized_digest : Digest.Logical_value.t;
}

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* key = W.Obj.required o "key" Canonical.decode in
      let* common_instance =
        W.Obj.required o "common_instance" Instance.decode
      in
      let* left_normalized_digest =
        W.Obj.required o "left_normalized_digest" Wire_digest.logical_value
      in
      let* right_normalized_digest =
        W.Obj.required o "right_normalized_digest" Wire_digest.logical_value
      in
      Ok
        {
          key;
          common_instance;
          left_normalized_digest;
          right_normalized_digest;
        })

let encode v =
  W.obj
    [
      W.field "key" Canonical.encode v.key;
      W.field "common_instance" Instance.encode v.common_instance;
      W.field "left_normalized_digest" Wire_digest.encode_logical_value
        v.left_normalized_digest;
      W.field "right_normalized_digest" Wire_digest.encode_logical_value
        v.right_normalized_digest;
    ]
