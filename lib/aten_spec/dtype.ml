(* The element type of a synthesized tensor. A deliberately small set: the
   verification path compares f32 (and exact-matches integer) outputs, so input
   synthesis only needs these. Mapped to Aten_dtype in the runner. *)

type t = Bool | F32 | F64 | I32 | I64

let to_string = function
  | Bool -> "bool"
  | F32 -> "f32"
  | F64 -> "f64"
  | I32 -> "i32"
  | I64 -> "i64"

let of_string = function
  | "bool" -> Some Bool
  | "f32" -> Some F32
  | "f64" -> Some F64
  | "i32" -> Some I32
  | "i64" -> Some I64
  | _ -> None

let jsont : t Jsont.t =
  Jsont.map ~kind:"dtype"
    ~dec:(fun s ->
      match of_string s with
      | None -> Jsont.Error.msgf Jsont.Meta.none "unknown dtype %S" s
      | Some d -> d)
    ~enc:to_string Jsont.string
