type json = Jsont.json

module Constraint = struct
  type t =
    | Count_overflow
    | Duplicate_element
    | Empty_list
    | Malformed_document
    | Negative_value
    | Nonfinite_number
    | Out_of_range
end

type sort = [ `Array | `Bool | `Null | `Number | `Object | `String ]

type reason =
  | Constraint of Constraint.t
  | Duplicate_key of string
  | Expected of sort
  | Invalid_decimal of string
  | Invalid_digest of Digest_error.t
  | Invalid_text of Text_error.t
  | Missing_member of string
  | Unknown_member of string
  | Unknown_tag of { variant : string; tag : string }
  | Unsupported_schema_version of string

type error = { path : string list; reason : reason }

let pp_sort ppf s =
  Fmt.string ppf
    (match s with
    | `Array -> "array"
    | `Bool -> "bool"
    | `Null -> "null"
    | `Number -> "number"
    | `Object -> "object"
    | `String -> "string")

let pp_constraint ppf (c : Constraint.t) =
  Fmt.string ppf
    (match c with
    | Count_overflow -> "count overflow"
    | Duplicate_element -> "duplicate element"
    | Empty_list -> "empty list"
    | Malformed_document -> "malformed document"
    | Negative_value -> "negative value"
    | Nonfinite_number -> "nonfinite number"
    | Out_of_range -> "out of range")

let pp_reason ppf = function
  | Constraint c -> pp_constraint ppf c
  | Duplicate_key k -> Fmt.pf ppf "duplicate key %S" k
  | Expected s -> Fmt.pf ppf "expected %a" pp_sort s
  | Invalid_decimal s -> Fmt.pf ppf "invalid decimal %S" s
  | Invalid_digest e -> Fmt.pf ppf "invalid digest (%a)" Digest_error.pp e
  | Invalid_text e -> Text_error.pp ppf e
  | Missing_member k -> Fmt.pf ppf "missing member %S" k
  | Unknown_member k -> Fmt.pf ppf "unknown member %S" k
  | Unknown_tag { variant; tag } -> Fmt.pf ppf "unknown %s tag %S" variant tag
  | Unsupported_schema_version v -> Fmt.pf ppf "unsupported schema version %S" v

let pp_error ppf { path; reason } =
  Fmt.pf ppf "%a at /%a" pp_reason reason Fmt.(list ~sep:(any "/") string) path

let fail reason = Err.fail ~pp_error { path = []; reason }

let at component r =
  Err.map_error (fun e -> { e with path = component :: e.path }) r

let ( let* ) = Err.bind
let return = Err.return

type 'a decoder = json -> ('a, error) Err.t
type 'a encoder = 'a -> json

let string : string decoder = function
  | Jsont.String (s, _) -> Ok s
  | _ -> fail (Expected `String)

let bool : bool decoder = function
  | Jsont.Bool (b, _) -> Ok b
  | _ -> fail (Expected `Bool)

let float : float decoder = function
  | Jsont.Number (f, _) ->
      if Float.is_finite f then Ok f else fail (Constraint Nonfinite_number)
  | _ -> fail (Expected `Number)

(* A decimal integer: optional '-', then digits with no leading zero unless the
   value is exactly "0". [Int64.of_string] alone would accept "0x10", "1_0"
   and "+5", none of which is the wire form. *)
let is_decimal s =
  let n = String.length s in
  let start = if n > 0 && s.[0] = '-' then 1 else 0 in
  let digits = n - start in
  digits > 0
  && String.for_all (fun c -> c >= '0' && c <= '9') (String.sub s start digits)
  && (digits = 1 || s.[start] <> '0')
  && s <> "-0"

let int64 : int64 decoder = function
  | Jsont.String (s, _) -> (
      if not (is_decimal s) then fail (Invalid_decimal s)
      else
        match Int64.of_string_opt s with
        | Some v -> Ok v
        | None -> fail (Invalid_decimal s))
  | _ -> fail (Expected `String)

(* [Sys.int_size] is 31 on js_of_ocaml; bound to 2^30 - 1 so a value decoded on
   one backend means the same on all. *)
let max_int31 = 0x3FFF_FFFF

let int : int decoder =
 fun j ->
  let* v = int64 j in
  if
    Int64.compare v (Int64.of_int (-max_int31)) < 0
    || Int64.compare v (Int64.of_int max_int31) > 0
  then fail (Constraint Out_of_range)
  else Ok (Int64.to_int v)

let enc_string s = Jsont.Json.string s
let enc_bool b = Jsont.Json.bool b
let enc_int64 v = Jsont.Json.string (Int64.to_string v)
let enc_int v = enc_int64 (Int64.of_int v)

let enc_float f =
  if Float.is_finite f then Jsont.Json.number f
  else invalid_arg "Wire.enc_float: nonfinite value must be tagged"

let enc_null = Jsont.Json.null ()

module Text (T : Text_value.S) = struct
  let decode : T.t decoder =
   fun j ->
    let* s = string j in
    Err.map_error
      (fun e -> { path = []; reason = Invalid_text e })
      (T.of_string s)

  let encode v = enc_string (T.to_string v)
end

let list d : 'a list decoder = function
  | Jsont.Array (items, _) ->
      let rec go i acc = function
        | [] -> Ok (List.rev acc)
        | x :: rest ->
            let* v = at (string_of_int i) (d x) in
            go (i + 1) (v :: acc) rest
      in
      go 0 [] items
  | _ -> fail (Expected `Array)

let enc_list e xs = Jsont.Json.list (List.map e xs)

let option d : 'a option decoder = function
  | Jsont.Null _ -> Ok None
  | j -> Err.map Option.some (d j)

let enc_option e = function None -> enc_null | Some v -> e v

let pair da db : ('a * 'b) decoder = function
  | Jsont.Array ([ a; b ], _) ->
      let* a = at "0" (da a) in
      let* b = at "1" (db b) in
      Ok (a, b)
  | Jsont.Array _ -> fail (Constraint Out_of_range)
  | _ -> fail (Expected `Array)

let enc_pair ea eb (a, b) = Jsont.Json.list [ ea a; eb b ]

module Obj = struct
  type t = { members : (string * json) list; mutable consumed : string list }

  let of_json : json -> (t, error) Err.t = function
    | Jsont.Object (mems, _) ->
        let rec go seen = function
          | [] -> Ok ()
          | ((name, _), _) :: rest ->
              if List.mem name seen then fail (Duplicate_key name)
              else go (name :: seen) rest
        in
        let* () = go [] mems in
        Ok
          { members = List.map (fun ((n, _), v) -> (n, v)) mems; consumed = [] }
    | _ -> fail (Expected `Object)

  let find o name =
    o.consumed <- name :: o.consumed;
    List.assoc_opt name o.members

  let required o name d =
    match find o name with
    | None -> fail (Missing_member name)
    | Some v -> at name (d v)

  let optional o name d =
    match find o name with
    | None -> fail (Missing_member name)
    | Some v -> at name (option d v)

  let tag o = required o "tag" string

  let finish o =
    match
      List.find_opt (fun (n, _) -> not (List.mem n o.consumed)) o.members
    with
    | None -> Ok ()
    | Some (n, _) -> fail (Unknown_member n)
end

let record reader : 'a decoder =
 fun j ->
  let* o = Obj.of_json j in
  let* v = reader o in
  let* () = Obj.finish o in
  Ok v

let unknown_tag ~kind tag = fail (Unknown_tag { variant = kind; tag })

let variant f : 'a decoder =
  record (fun o ->
      let* tag = Obj.tag o in
      f tag o)

let obj members =
  Jsont.Json.object'
    (List.map (fun (n, v) -> Jsont.Json.mem (Jsont.Json.name n) v) members)

let tagged tag members = obj (("tag", enc_string tag) :: members)
let field name enc v = (name, enc v)

let check_version o ~expected =
  let* v = Obj.required o "schema_version" string in
  if String.equal v expected then Ok () else fail (Unsupported_schema_version v)

(* Canonical text. Strings escape the quote, the backslash and every control
   character; everything else is emitted as the UTF-8 bytes it already is. *)
let add_string b s =
  Buffer.add_char b '"';
  String.iter
    (fun c ->
      match c with
      | '"' -> Buffer.add_string b "\\\""
      | '\\' -> Buffer.add_string b "\\\\"
      | c when Char.code c < 0x20 ->
          Buffer.add_string b (Printf.sprintf "\\u%04x" (Char.code c))
      | c -> Buffer.add_char b c)
    s;
  Buffer.add_char b '"'

let rec add_json b : json -> unit = function
  | Jsont.Null _ -> Buffer.add_string b "null"
  | Bool (v, _) -> Buffer.add_string b (if v then "true" else "false")
  | Number (f, _) ->
      if Float.is_finite f then Buffer.add_string b (Printf.sprintf "%.17g" f)
      else Buffer.add_string b "null"
  | String (s, _) -> add_string b s
  | Array (items, _) ->
      Buffer.add_char b '[';
      List.iteri
        (fun i x ->
          if i > 0 then Buffer.add_char b ',';
          add_json b x)
        items;
      Buffer.add_char b ']'
  | Object (mems, _) ->
      let mems =
        List.sort (fun ((a, _), _) ((c, _), _) -> String.compare a c) mems
      in
      Buffer.add_char b '{';
      List.iteri
        (fun i ((n, _), v) ->
          if i > 0 then Buffer.add_char b ',';
          add_string b n;
          Buffer.add_char b ':';
          add_json b v)
        mems;
      Buffer.add_char b '}'

let canonical_string j =
  let b = Buffer.create 256 in
  add_json b j;
  Buffer.contents b

let enum ~kind table =
  let decode : _ decoder =
   fun j ->
    let* s = string j in
    match List.find_opt (fun (_, name) -> String.equal name s) table with
    | Some (v, _) -> Ok v
    | None -> unknown_tag ~kind s
  in
  let encode v =
    match List.find_opt (fun (x, _) -> x = v) table with
    | Some (_, name) -> enc_string name
    | None -> invalid_arg ("Wire.enum: unlisted constructor of " ^ kind)
  in
  (decode, encode)

let nonneg_float : float decoder =
 fun j ->
  let* f = float j in
  if f < 0.0 then fail (Constraint Negative_value) else Ok f

let nonneg_int : int decoder =
 fun j ->
  let* v = int j in
  if v < 0 then fail (Constraint Negative_value) else Ok v

let nonneg_int64 : int64 decoder =
 fun j ->
  let* v = int64 j in
  if Int64.compare v 0L < 0 then fail (Constraint Negative_value) else Ok v

let unique ~key d : 'a list decoder =
 fun j ->
  let* items = d j in
  let rec go i seen = function
    | [] -> Ok items
    | x :: rest ->
        let k = key x in
        if List.mem k seen then
          at (string_of_int i) (fail (Constraint Duplicate_element))
        else go (i + 1) (k :: seen) rest
  in
  go 0 [] items
