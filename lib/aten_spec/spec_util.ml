(* Small jsont helpers shared by the spec codecs.

   The format is read *and* written, so codecs are bidirectional. The unions
   ({"uniform": ...}, {"values": ...}) are hand-rolled over generic JSON in both
   directions: [union] dispatches a single-key object by its key on decode, and
   the constructors below ([jobj]/[single]/...) build one on encode.
   [find_member] looks a key up in a decoded object so a codec can pick among
   several optional keys (e.g. a tensor's values/random/sequence payload). *)

let meta = Jsont.Meta.none

(* Generic-JSON constructors for the encode direction. *)
let jstr s = Jsont.String (s, meta)
let jnum n = Jsont.Number (n, meta)
let jbool b = Jsont.Bool (b, meta)
let jint i = Jsont.Number (float_of_int i, meta)
let jnull = Jsont.Null ((), meta)
let jarr xs = Jsont.Array (xs, meta)
let jobj kvs = Jsont.Object (List.map (fun (k, v) -> ((k, meta), v)) kvs, meta)

(* A single-key (tagged) object, the encoding of our unions. *)
let single ~case v = jobj [ (case, v) ]

(* Decode [value] with [codec], raising a jsont decode error on failure. *)
let dec codec value =
  match Jsont.Json.decode codec value with
  | Error s -> Jsont.Error.msg meta s
  | Ok v -> v

(* Encode [v] with [codec] to generic JSON, for nesting under a union key. *)
let enc codec v =
  match Jsont.Json.encode codec v with
  | Error s -> Jsont.Error.msg meta s
  | Ok j -> j

let members = function Jsont.Object (ms, _) -> Some ms | _ -> None
let is_null = function Jsont.Null _ -> true | _ -> false

let find_member ms key =
  List.find_map
    (fun ((k, _), v) -> if String.equal k key then Some v else None)
    ms

(* Dispatch a single-key object [{ "<case>": payload }] through [cases]. *)
let union ~kind cases = function
  | Jsont.Object ([ ((key, _), value) ], _) -> (
      match List.assoc_opt key cases with
      | None ->
          Jsont.Error.msgf Jsont.Meta.none "%s: unknown variant %S" kind key
      | Some f -> f value)
  | Jsont.Object _ ->
      Jsont.Error.msgf Jsont.Meta.none "%s: expected a single-key object" kind
  | _ -> Jsont.Error.msgf Jsont.Meta.none "%s: expected an object" kind
