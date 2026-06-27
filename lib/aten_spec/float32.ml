(* A reproducible float32 value.

   JSON floats are decimal text and lose precision on round-trip, so a tensor
   value is stored as the IEEE-754 *single* bit pattern in hex ("0x3f800000").
   [Int32.bits_of_float] / [Int32.float_of_bits] move between an OCaml float and
   that 32-bit pattern (same primitive used by lib/native/half.ml).

   [to_f32] rounds a float64 to its float32-canonical value — the value an f32
   tensor would actually store. Synthesis routes every generated/derived float
   through it so the spec's claimed value equals the stored one exactly. *)

type t = float

let to_f32 x = Int32.float_of_bits (Int32.bits_of_float x)
let to_hex (f : t) = Printf.sprintf "0x%08lx" (Int32.bits_of_float f)
let of_hex s = Int32.float_of_bits (Int32.of_string s)

(* Decodes an int32 hex string (canonical, lossless) or a plain JSON number
   (convenience, rounded to f32); encodes back to the canonical hex string. *)
let jsont : t Jsont.t =
  Jsont.map ~kind:"f32"
    ~dec:(function
      | Jsont.String (s, _) -> (
          match Int32.of_string_opt s with
          | Some b -> Int32.float_of_bits b
          | None -> Jsont.Error.msgf Jsont.Meta.none "invalid float32 hex: %S" s
          )
      | Jsont.Number (n, _) -> to_f32 n
      | _ ->
          Jsont.Error.msgf Jsont.Meta.none
            "float32: expected an int32 hex string or a number")
    ~enc:(fun f -> Jsont.String (to_hex f, Jsont.Meta.none))
    Jsont.json
