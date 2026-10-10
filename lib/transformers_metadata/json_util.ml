open Err.Syntax
module J = Jsont.Json

let ( >>= ) = Err.bind
let obj fields = J.object' (List.map (fun (k, v) -> J.mem (J.name k) v) fields)
let invalid s = Err.fail (`Metadata_invalid s)
let missing s = Err.fail (`Metadata_missing s)

let members = function
  | Jsont.Object (ms, _) -> Ok (List.map (fun ((k, _), v) -> (k, v)) ms)
  | _ -> invalid "expected object"

let member key value =
  let* ms = members value in
  Err.of_option (`Metadata_missing key) (List.assoc_opt key ms)

let path keys value = Err.List.fold_left (fun j k -> member k j) value keys

let string = function
  | Jsont.String (s, _) -> Ok s
  | _ -> invalid "expected string"

let array = function
  | Jsont.Array (xs, _) -> Ok xs
  | _ -> invalid "expected array"

let integer = function
  | Jsont.Number (n, _)
    when Float.is_finite n && n >= 0. && n <= 9007199254740991.
         && Float.floor n = n ->
      Ok (Int64.of_float n)
  | _ -> invalid "expected checked nonnegative integer"

let bool = function
  | Jsont.Bool (b, _) -> Ok b
  | _ -> invalid "expected boolean"

let unique names =
  let rec go = function
    | a :: (b :: _ as rest) ->
        if a = b then Err.fail (`Metadata_duplicate a) else go rest
    | _ -> Ok ()
  in
  go (List.sort String.compare names)

let filter predicate xs =
  let+ rows =
    Err.List.map
      (fun x ->
        let+ keep = predicate x in
        if keep then Some x else None)
      xs
  in
  List.filter_map Fun.id rows

let rec keys = function
  | Jsont.Object (ms, _) ->
      let* () = unique (List.map (fun ((k, _), _) -> k) ms) in
      Err.List.iter (fun (_, j) -> keys j) ms
  | Jsont.Array (xs, _) -> Err.List.iter keys xs
  | _ -> Ok ()

let parse bytes =
  let* j =
    Err.import ~pos:__POS__
      (fun e -> `Metadata_invalid e)
      (Jsont_bytesrw.decode_string Jsont.json bytes)
  in
  let* () = keys j in
  Ok j

let rec sorted = function
  | Jsont.Object (ms, meta) ->
      Jsont.Object
        ( List.sort
            (fun ((a, _), _) ((b, _), _) -> String.compare a b)
            (List.map (fun (k, v) -> (k, sorted v)) ms),
          meta )
  | Jsont.Array (xs, meta) -> Jsont.Array (List.map sorted xs, meta)
  | j -> j

let text j =
  Err.import ~pos:__POS__
    (fun e -> `Metadata_invalid e)
    (Jsont_bytesrw.encode_string ~format:Jsont.Indent Jsont.json (sorted j))

let write file j =
  let* s = text j in
  Out_channel.with_open_bin file (fun oc ->
      output_string oc s;
      output_char oc '\n');
  Ok ()

let read file =
  let* bytes = Pt2_fixture_unix.Fetch.read file in
  parse bytes

let mismatch ~identity ~field ~actual ~expected =
  Err.fail
    (`Metadata_mismatch { Fault.Mismatch.actual; expected; field; identity })

let same ~identity ~field actual expected =
  if actual = expected then Ok ()
  else mismatch ~identity ~field ~actual ~expected

let equal ~identity ~field actual expected =
  let* a = text actual in
  let* b = text expected in
  same ~identity ~field a b

let schema version j =
  let* v = member "schema_version" j >>= integer in
  if v = Int64.of_int version then Ok ()
  else invalid "unsupported schema_version"

let field k j = member k j >>= string

let pin (p : Pt2_checkpoint_map.Document.Pin.t) =
  obj
    [
      ("name", J.string p.name);
      ("sha256", J.string (Pt2_sha256.Digest.to_hex p.sha256));
      ("size", J.number (Int64.to_float p.size));
      ("url", J.string p.url);
    ]

let pin_of j =
  let* name = field "name" j in
  let* sha256 = field "sha256" j in
  let* size = member "size" j >>= integer in
  let* url = field "url" j in
  Pt2_checkpoint_map.Document.pin_of_fields ~name ~sha256 ~size ~url
