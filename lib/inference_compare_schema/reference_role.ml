module W = Wire
module Oracle = W.Text (Id.Oracle)

type t = Left_reference | Oracle of Id.Oracle.t | Peers | Right_reference

let decode : t W.decoder =
  W.variant (fun tag o ->
      let ( let* ) = W.( let* ) in
      match tag with
      | "left_reference" -> Ok Left_reference
      | "oracle" ->
          let* id = W.Obj.required o "oracle" Oracle.decode in
          Ok (Oracle id)
      | "peers" -> Ok Peers
      | "right_reference" -> Ok Right_reference
      | tag -> W.unknown_tag ~kind:"reference_role" tag)

let encode = function
  | Left_reference -> W.tagged "left_reference" []
  | Oracle id -> W.tagged "oracle" [ W.field "oracle" Oracle.encode id ]
  | Peers -> W.tagged "peers" []
  | Right_reference -> W.tagged "right_reference" []
