(* The element type of a synthesized tensor. A deliberately small set: the
   verification path compares f32 (and exact-matches integer) outputs, so input
   synthesis only needs these. Mapped to Aten_dtype in the runner. *)

type t = F32 | F64 | I64 | I32 | Bool

let to_string = function
  | F32 -> "f32"
  | F64 -> "f64"
  | I64 -> "i64"
  | I32 -> "i32"
  | Bool -> "bool"

let of_string = function
  | "f32" -> Some F32
  | "f64" -> Some F64
  | "i64" -> Some I64
  | "i32" -> Some I32
  | "bool" -> Some Bool
  | _ -> None

let jsont : t Jsont.t =
  Jsont.map ~kind:"dtype"
    ~dec:(fun s ->
      match of_string s with
      | Some d -> d
      | None -> Jsont.Error.msgf Jsont.Meta.none "unknown dtype %S" s)
    ~enc:to_string Jsont.string
