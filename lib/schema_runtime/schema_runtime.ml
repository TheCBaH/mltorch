module String_map = Map.Make (String)

(* Float codec matching PyTorch's export serializer (torch/_export/serde/
   serialize.py): finite floats serialize as JSON numbers, but the non-finite
   ones serialize as the JSON *strings* "Infinity", "-Infinity" and "NaN"
   (json.dumps is called with allow_nan=False). Jsont's stock [Jsont.number]
   only accepts JSON numbers, so model files carrying e.g. {"as_float":
   "-Infinity"} (clamp/attention masks in the ViT models) fail to decode.

   This codec accepts a number or one of the three special strings on decode,
   and round-trips them back to the same strings on encode. The generated
   decoders reach it unqualified via their [open Schema_runtime]. *)
let float_jsont : float Jsont.t =
  let dec = function
    | Jsont.Number (n, _) -> n
    | Jsont.String ("Infinity", _) -> Float.infinity
    | Jsont.String ("-Infinity", _) -> Float.neg_infinity
    | Jsont.String ("NaN", _) -> Float.nan
    | Jsont.String (s, _) ->
        Jsont.Error.msgf Jsont.Meta.none "invalid float string: %S" s
    | _ ->
        Jsont.Error.msgf Jsont.Meta.none
          "expected a JSON number or a float string (Infinity/-Infinity/NaN)"
  in
  let enc v =
    if Float.is_nan v then Jsont.String ("NaN", Jsont.Meta.none)
    else if v = Float.infinity then Jsont.String ("Infinity", Jsont.Meta.none)
    else if v = Float.neg_infinity then
      Jsont.String ("-Infinity", Jsont.Meta.none)
    else Jsont.Number (v, Jsont.Meta.none)
  in
  Jsont.map ~kind:"float" ~dec ~enc Jsont.json
