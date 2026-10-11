open Err.Syntax

type error = [ `Timm_fixture of string ]

let pp_error ppf (`Timm_fixture why) = Fmt.string ppf why
let bad why = Err.fail (`Timm_fixture why)
let ( >>= ) = Err.bind

let members = function
  | Jsont.Object (xs, _) -> Ok (List.map (fun ((k, _), v) -> (k, v)) xs)
  | _ -> bad "expected object"

let member key value =
  let* xs = members value in
  Err.of_option (`Timm_fixture ("missing field " ^ key)) (List.assoc_opt key xs)

let path keys value = Err.List.fold_left (fun j k -> member k j) value keys
let string = function Jsont.String (s, _) -> Ok s | _ -> bad "expected string"
let array = function Jsont.Array (xs, _) -> Ok xs | _ -> bad "expected array"
let field key j = member key j >>= string

let fields names j =
  let* xs = members j in
  let actual = List.map fst xs in
  if List.sort String.compare names = List.sort String.compare actual then Ok ()
  else bad "unexpected or missing object field"

let number = function
  | Jsont.Number (n, _) when Float.is_finite n -> Ok n
  | _ -> bad "expected finite number"

let integer j =
  let* n = number j in
  if n < 0. || n > 536870912. || Float.floor n <> n then
    bad "integer exceeds supported limit"
  else Ok (int_of_float n)

let same what a b = if a = b then Ok () else bad (what ^ " disagrees")

let unique names =
  let sorted = List.sort String.compare names in
  let rec go = function
    | a :: (b :: _ as rest) -> if a = b then bad ("duplicate " ^ a) else go rest
    | _ -> Ok ()
  in
  go sorted

let rec keys = function
  | Jsont.Object (xs, _) ->
      let* () = unique (List.map (fun ((k, _), _) -> k) xs) in
      Err.List.iter (fun (_, j) -> keys j) xs
  | Jsont.Array (xs, _) -> Err.List.iter keys xs
  | _ -> Ok ()

let parse text =
  if String.length text > 8 * 1024 * 1024 then bad "JSON exceeds byte limit"
  else
    let* j =
      Jsont_bytesrw.decode_string Jsont.json text
      |> Err.import ~pos:__POS__ (fun e -> `Timm_fixture e)
    in
    let+ () = keys j in
    j

let version n j =
  member "schema_version" j >>= integer >>= same "schema_version" n

let shape j =
  let* xs = array j in
  if List.length xs > 8 then bad "unsupported rank"
  else
    let* dims = Err.List.map integer xs in
    if List.exists (( >= ) 0) dims then bad "nonpositive dimension" else Ok dims

let digest s =
  Err.of_option (`Timm_fixture "invalid sha256") (Pt2_sha256.Digest.of_hex s)
