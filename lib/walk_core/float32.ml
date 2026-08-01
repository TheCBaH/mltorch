(* A reproducible float32 value.

   Most finite f32 values are readable as JSON numbers. Values with no portable
   JSON number spelling, or whose bit identity is easy to lose (NaN, infinities,
   and negative zero), are stored as the IEEE-754 *single* bit pattern in hex
   ("0x3f800000").
   [Int32.bits_of_float] / [Int32.float_of_bits] move between an OCaml float and
   that 32-bit pattern (same primitive used by lib/native/half.ml).

   [to_f32] rounds a float64 to its float32-canonical value — the value an f32
   tensor would actually store. Synthesis routes every generated/derived float
   through it so the spec's claimed value equals the stored one exactly. *)

type t = float

let to_f32 x = Int32.float_of_bits (Int32.bits_of_float x)
let to_hex (f : t) = Printf.sprintf "0x%08lx" (Int32.bits_of_float f)
let of_hex s = Int32.float_of_bits (Int32.of_string s)

let number_round_trips bits f =
  float_of_string_opt (Printf.sprintf "%.17g" f)
  |> Option.fold ~none:false ~some:(fun back ->
      Int32.equal (Int32.bits_of_float (to_f32 back)) bits)

let enc_json f =
  let f = to_f32 f in
  let bits = if Float.is_nan f then 0x7fc00000l else Int32.bits_of_float f in
  if Float.is_finite f && bits <> Int32.min_int && number_round_trips bits f
  then Jsont.Number (f, Jsont.Meta.none)
  else Jsont.String (to_hex f, Jsont.Meta.none)

(* Decodes an int32 hex string (canonical, lossless) or a plain JSON number
   (convenience, rounded to f32); encodes readable finite values as numbers and
   falls back to the raw int32 hex string for exceptional bit patterns. *)
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
    ~enc:enc_json Jsont.json
