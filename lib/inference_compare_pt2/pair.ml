(* The ATen / Native pair for one exported model and one case: both endpoint
   models, the importer's correspondence as a mapping artifact, the pinned case,
   and the two coordinator endpoints. Everything is built from the model file
   and the user tensor; nothing is matched by name or value. *)

open Inference_compare_schema
module Codecs = Inference_compare_codecs
module L = Codecs.Logical_value
module Core = Inference_compare_core
module Ex = Inference_compare_extract

type t = {
  aten : Aten_model.t;
  native : Native_model.t;
  import : Import_map.t;
  case : Case.t;
  case_ctx : L.Context.t;
  hashers : Codecs.Payload_hasher.Registry.t;
  algorithm : Id.Digest_algorithm.t;
}

let ( let* ) = Result.bind

let of_aten ~hashers ~algorithm ~(inputs : (string * Pt2_tensor.t) list)
    (aten : Aten_model.t) : (t, string) result =
  let* lowered, capture =
    Native_interp.lower_archive_captured aten.archive
    |> Result.map_error (fun e ->
        Format.asprintf "lowering: %a" Native_interp.pp_error (Err.Error.kind e))
  in
  let* native = Native_model.of_lowered ~hashers ~algorithm lowered capture in
  let case_ctx =
    {
      L.Context.codecs = Codecs.Element_codec.Registry.builtin;
      hashers;
      algorithm;
      ledger = Codecs.Owned_blob.Ledger.create ();
    }
  in
  let* case = Case.load case_ctx aten ~inputs in
  let payload_digest name =
    match List.assoc_opt name case.bytes with
    | None -> Error ("no value for " ^ name)
    | Some bytes -> (
        match Codecs.Payload_hasher.digest_string hashers algorithm bytes with
        | Ok d -> Ok d
        | Error _ -> Error "digest backend unavailable")
  in
  let* import = Import_map.make ~aten ~native ~payload_digest in
  Ok { aten; native; import; case; case_ctx; hashers; algorithm }

let load ~hashers ~algorithm ~path ~inputs =
  let* aten = Aten_model.of_file ~hashers ~algorithm path in
  of_aten ~hashers ~algorithm ~inputs aten

(* The value of an exporter source of the case, by its name. *)
let source_bytes t name =
  match List.assoc_opt name t.case.bytes with
  | Some b -> Ok b
  | None -> Error ("no value for the source " ^ name)

let mapping t = t.import.mapping
let left_graph t = t.aten.desc
let right_graph t = t.native.desc

let initial t =
  List.map
    (fun (name, v) ->
      match Id.Canonical_value.of_string name with
      | Ok k -> (k, v)
      | Error _ -> invalid_arg "Pair.initial")
    t.case.values

let node_attributes t side (r : Node_ref.t) =
  match (side : Side.t) with
  | Left -> Aten_model.attributes_of t.aten r
  | Right -> Native_model.attributes_of t.native r

let aten_endpoint t : Core.Side_exec.endpoint =
  Endpoint
    {
      runner = Aten_runner.create ();
      build =
        (fun ex -> Ex.Dialect.realize (module Aten_dialect.Dialect) t.aten ex);
    }

let native_endpoint ?dots t : Core.Side_exec.endpoint =
  Endpoint
    {
      runner = Native_runner.create ?dots ();
      build =
        (fun ex ->
          Ex.Dialect.realize (module Native_dialect.Dialect) t.native ex);
    }

let aten_identity () = Aten_runner.identity ()
let native_identity ?dots () = Native_runner.identity ?dots ()
