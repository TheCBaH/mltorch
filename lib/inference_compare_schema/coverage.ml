module W = Wire
module Description = W.Text (Prose.Description)

type t = Complete | Partial of Prose.Description.t list

let decode : t W.decoder =
  W.variant (fun tag o ->
      let ( let* ) = W.( let* ) in
      match tag with
      | "complete" -> Ok Complete
      | "partial" ->
          let* reasons =
            W.Obj.required o "reasons" (W.list Description.decode)
          in
          Ok (Partial reasons)
      | tag -> W.unknown_tag ~kind:"coverage" tag)

let encode = function
  | Complete -> W.tagged "complete" []
  | Partial reasons ->
      W.tagged "partial"
        [ W.field "reasons" (W.enc_list Description.encode) reasons ]
