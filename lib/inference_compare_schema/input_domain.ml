module W = Wire
module Description = W.Text (Prose.Description)

type t = {
  description : Prose.Description.t;
  specification_digest : Digest.Input_domain.t;
}

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* description = W.Obj.required o "description" Description.decode in
      let* specification_digest =
        W.Obj.required o "specification_digest" Wire_digest.input_domain
      in
      Ok { description; specification_digest })

let encode d =
  W.obj
    [
      W.field "description" Description.encode d.description;
      W.field "specification_digest" Wire_digest.encode_input_domain
        d.specification_digest;
    ]
