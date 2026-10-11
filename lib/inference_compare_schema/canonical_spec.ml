module W = Wire
module Key = W.Text (Id.Canonical_value)

type t = { key : Id.Canonical_value.t; spec : Tensor_spec.t }

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* key = W.Obj.required o "key" Key.decode in
      let* spec = W.Obj.required o "spec" Tensor_spec.decode in
      Ok { key; spec })

let encode s =
  W.obj
    [ W.field "key" Key.encode s.key; W.field "spec" Tensor_spec.encode s.spec ]
