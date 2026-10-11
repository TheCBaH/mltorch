module W = Wire

type t = {
  selection_digest : Digest.Policy_selection.t;
  bindings_digest : Digest.Policy_bindings.t;
  bindings : Policy_binding.t list;
}

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* selection_digest =
        W.Obj.required o "selection_digest" Wire_digest.policy_selection
      in
      let* bindings_digest =
        W.Obj.required o "bindings_digest" Wire_digest.policy_bindings
      in
      let* bindings =
        W.Obj.required o "bindings" (W.list Policy_binding.decode)
      in
      Ok { selection_digest; bindings_digest; bindings })

let encode b =
  W.obj
    [
      W.field "selection_digest" Wire_digest.encode_policy_selection
        b.selection_digest;
      W.field "bindings_digest" Wire_digest.encode_policy_bindings
        b.bindings_digest;
      W.field "bindings" (W.enc_list Policy_binding.encode) b.bindings;
    ]
