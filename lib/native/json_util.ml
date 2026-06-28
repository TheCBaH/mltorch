(* Shared jsont helpers for the native library. Mirrors the role of
   lib/aten_spec/spec_util.ml: JSON value constructors (encode direction),
   object-member lookup helpers (decode direction), tagged-union dispatch,
   and the float32 codec with hex fallback.  All other native modules that
   define a [jsont] value reference this module rather than re-defining the
   boilerplate. *)

(* ---- JSON value constructors ---------------------------------------------- *)

let meta = Jsont.Meta.none
let jstr s = Jsont.String (s, meta)
let jint i = Jsont.Number (float_of_int i, meta)
let jnull = Jsont.Null ((), meta)
let jarr xs = Jsont.Array (xs, meta)
let jobj kvs = Jsont.Object (List.map (fun (k, v) -> ((k, meta), v)) kvs, meta)

(* A single-key (tagged) object — the encoding of every union variant. *)
let single ~case v = jobj [ (case, v) ]

(* ---- Decode/encode via a codec -------------------------------------------- *)

let dec codec value =
  match Jsont.Json.decode codec value with
  | Ok v -> v
  | Error s -> Jsont.Error.msg meta s

let enc codec v =
  match Jsont.Json.encode codec v with
  | Ok j -> j
  | Error s -> Jsont.Error.msg meta s

(* ---- Object helpers ------------------------------------------------------- *)

let members = function Jsont.Object (ms, _) -> Some ms | _ -> None

let find_member ms key =
  List.find_map
    (fun ((k, _), v) -> if String.equal k key then Some v else None)
    ms

(* Require a field by name; error if absent. *)
let req_field ms key codec ctx =
  match find_member ms key with
  | Some v -> dec codec v
  | None -> Jsont.Error.msgf meta "%s: missing %S" ctx key

(* Extract the members list from an object, error if not an object. *)
let req_obj json ctx =
  match members json with
  | Some ms -> ms
  | None -> Jsont.Error.msgf meta "%s: expected object" ctx

(* Optional field: [None] when the key is absent. *)
let opt_field ms key codec =
  match find_member ms key with Some v -> Some (dec codec v) | None -> None

(* Dispatch a single-key object { "<case>": payload } through [cases]. *)
let union ~kind cases = function
  | Jsont.Object ([ ((key, _), value) ], _) -> (
      match List.assoc_opt key cases with
      | Some f -> f value
      | None -> Jsont.Error.msgf meta "%s: unknown variant %S" kind key)
  | Jsont.Object _ ->
      Jsont.Error.msgf meta "%s: expected a single-key object" kind
  | _ -> Jsont.Error.msgf meta "%s: expected an object" kind

(* ---- Float32 codec -------------------------------------------------------- *)
(* Mirrors lib/aten_spec/float32.ml (not shared: adding aten_spec as a dep of
   native would create a circular dependency).  Finite values that survive a
   %.17g round-trip are stored as JSON numbers; special patterns (NaN, ±inf,
   -0, and ambiguous decimals) fall back to the 32-bit hex "0x3f800000". *)

let f32_to_f32 x = Int32.float_of_bits (Int32.bits_of_float x)

let enc_f32 f =
  let f = f32_to_f32 f in
  let bits = Int32.bits_of_float f in
  let round_trips =
    match float_of_string_opt (Printf.sprintf "%.17g" f) with
    | Some back -> Int32.equal (Int32.bits_of_float (f32_to_f32 back)) bits
    | None -> false
  in
  if Float.is_finite f && bits <> Int32.min_int && round_trips then
    Jsont.Number (f, meta)
  else Jsont.String (Printf.sprintf "0x%08lx" bits, meta)

let f32_jsont : float Jsont.t =
  Jsont.map ~kind:"f32"
    ~dec:(function
      | Jsont.String (s, _) -> (
          match Int32.of_string_opt s with
          | Some b -> Int32.float_of_bits b
          | None -> Jsont.Error.msgf meta "f32: invalid hex %S" s)
      | Jsont.Number (n, _) -> f32_to_f32 n
      | _ -> Jsont.Error.msgf meta "f32: expected number or hex string")
    ~enc:enc_f32 Jsont.json
