(* The one place text meets the schema: parse a document into the generic JSON
   value, then hand it to a decoder. Duplicate keys survive Jsont's own parser
   as repeated members, which {!Wire.record} then rejects. *)

let parse s : (Wire.json, Wire.error) Err.t =
  match Jsont_bytesrw.decode_string Jsont.json s with
  | Ok j -> Ok j
  | Error _ -> Wire.fail (Wire.Constraint Malformed_document)

let decode_string d s : ('a, Wire.error) Err.t =
  match parse s with Error e -> Error e | Ok j -> d j

let to_string j = Wire.canonical_string j
