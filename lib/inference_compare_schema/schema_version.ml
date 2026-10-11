module W = Wire

type t = V1

let to_string V1 = "1"

let decode : t W.decoder =
 fun j ->
  let ( let* ) = W.( let* ) in
  let* s = W.string j in
  match s with "1" -> Ok V1 | v -> W.fail (W.Unsupported_schema_version v)

let encode v = W.enc_string (to_string v)
