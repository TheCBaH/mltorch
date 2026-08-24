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

(* Union "as_int" arms (Argument/ConstantValue/SymInt/SymExprHint) carry an
   unconstrained Python [int], unlike a struct's plain [int] field (a
   count/index/version the schema itself keeps small). PyTorch's exporter
   puts one specific out-of-range value there as a real sentinel:
   [sys.maxsize] (2^63-1), written as e.g. aten.slice.Tensor's [end]
   whenever a slice is unbounded on the right ([x[a:]]). That exceeds even a
   native OCaml [int] (63 bits, [Sys.int_size = 63]), so stock [Jsont.int]
   rejects it outright -- but no real tensor axis ever reaches anywhere near
   [max_int] elements, and libtorch's own slice kernel already clamps an
   out-of-range [end] down to the dimension size, so decoding the sentinel
   (or anything else past [max_int]/[min_int]) AS [max_int]/[min_int]
   preserves its "unbounded" meaning losslessly for every real model.
   Ordinary struct [int] fields stay on stock [Jsont.int] via [pp_jsont_expr]
   -- rejecting a genuinely out-of-range count/index/version as invalid is
   the behavior we want there. The generated decoders reach this unqualified
   via their [open Schema_runtime]. *)
let python_int_jsont : int Jsont.t =
  let dec = function
    | Jsont.Number (n, _) ->
        if n > Float.of_int Stdlib.max_int then Stdlib.max_int
        else if n < Float.of_int Stdlib.min_int then Stdlib.min_int
        else int_of_float n
    | _ -> Jsont.Error.msgf Jsont.Meta.none "expected a JSON number"
  in
  let enc v = Jsont.Number (float_of_int v, Jsont.Meta.none) in
  Jsont.map ~kind:"int" ~dec ~enc Jsont.json
