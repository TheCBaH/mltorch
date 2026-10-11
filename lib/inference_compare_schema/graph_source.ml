module W = Wire

type t = { tensor : Tensor_ref.t; role : Input_role.t; spec : Tensor_spec.t }

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* tensor = W.Obj.required o "tensor" Tensor_ref.decode in
      let* role = W.Obj.required o "role" Input_role.decode in
      let* spec = W.Obj.required o "spec" Tensor_spec.decode in
      Ok { tensor; role; spec })

let encode s =
  W.obj
    [
      W.field "tensor" Tensor_ref.encode s.tensor;
      W.field "role" Input_role.encode s.role;
      W.field "spec" Tensor_spec.encode s.spec;
    ]
