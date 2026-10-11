module W = Wire
module Support_id = W.Text (Id.Metric_support)

type t = {
  id : Id.Metric_support.t;
  payload_digest : Digest.Metric_support.t;
  positions : int64;
}

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* id = W.Obj.required o "id" Support_id.decode in
      let* payload_digest =
        W.Obj.required o "payload_digest" Wire_digest.metric_support
      in
      let* positions = W.Obj.required o "positions" W.nonneg_int64 in
      Ok { id; payload_digest; positions })

let encode s =
  W.obj
    [
      W.field "id" Support_id.encode s.id;
      W.field "payload_digest" Wire_digest.encode_metric_support
        s.payload_digest;
      W.field "positions" W.enc_int64 s.positions;
    ]
