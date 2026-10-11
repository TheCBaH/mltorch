open Inference_compare_schema

type verdict = Lossless | Lossy | Unsupported

module E = Element_codec

let classify (s : E.t) (t : E.t) =
  if Id.Codec.equal s.id t.id then Lossless
  else
    match (s.element, t.element) with
    | Boolean, (Unsigned_integer | Signed_integer) -> Lossless
    | Boolean, (Float_binary32 | Float_binary64) -> Lossless
    | Unsigned_integer, Signed_integer ->
        if t.byte_width > s.byte_width then Lossless else Lossy
    | Unsigned_integer, (Float_binary32 | Float_binary64) ->
        (* An unsigned byte fits every float's significand; wider unsigned
           codecs would not fit binary32. *)
        if s.byte_width = 1 then Lossless else Unsupported
    | Signed_integer, Signed_integer ->
        if t.byte_width >= s.byte_width then Lossless else Lossy
    | Signed_integer, Unsigned_integer -> Lossy
    | Signed_integer, Float_binary64 ->
        if s.byte_width <= 4 then Lossless else Lossy
    | Signed_integer, Float_binary32 ->
        (* Exact only up to 2^24; admitted for narrow sources, whose
           conversion avoids double rounding, and always lossy. *)
        if s.byte_width <= 4 then Lossy else Unsupported
    | Float_binary32, Float_binary64 -> Lossless
    | Float_binary64, Float_binary32 -> Lossy
    | (Float_binary32 | Float_binary64), (Signed_integer | Unsigned_integer) ->
        Lossy
    | _, Boolean -> Unsupported
    | Unsigned_integer, Unsigned_integer ->
        if t.byte_width >= s.byte_width then Lossless else Lossy
    | (Float_binary32 | Float_binary64), (Float_binary32 | Float_binary64) ->
        Lossless

let rounding_admitted ~(source : E.t) ~(target : E.t)
    (r : Conversion_step.rounding) =
  match (E.is_float source, E.is_float target) with
  | false, true -> r = Nearest_even
  | _ -> true

let overflow_admitted ~(source : E.t) ~(target : E.t)
    (o : Conversion_step.overflow) =
  match o with
  | Reject | Saturate -> true
  | Wrap -> E.is_integer source && E.is_integer target
