(* One side of the pair on its own: a pinned model file and the coordinator
   endpoint made from it. This is what an experiment's engine configuration
   names; the pair of models and their mapping come from the exporter. *)

open Inference_compare_schema
module Codecs = Inference_compare_codecs
module Core = Inference_compare_core
module Ex = Inference_compare_extract

type kind = Aten | Native | Native4d

type t = {
  endpoint : Core.Side_exec.endpoint;
  identity : Engine_identity.t;
  graph : Graph_key.t;  (** The model's own graph key. *)
  attributes : Node_ref.t -> string;
}

let ( let* ) = Result.bind

(* The pin is always SHA-256 of the file's bytes, whatever algorithm the
   experiment's own digests use. *)
let file_pin ~hashers contents =
  match
    Codecs.Payload_hasher.digest_string hashers
      Inference_compare_sha256.algorithm contents
  with
  | Ok d -> Ok (Digest.Payload.hex d)
  | Error _ -> Error "digest backend unavailable"

let read_pinned ~hashers ~path ~pin =
  let* contents =
    match Pt2_archive.read_file path with
    | Ok c -> Ok c
    | Error e ->
        Error
          (Format.asprintf "%s: %a" path Pt2_archive.pp_error (Err.Error.kind e))
  in
  let* hex = file_pin ~hashers contents in
  if String.equal hex pin then Ok ()
  else
    Error
      (Printf.sprintf "%s: sha256 is %s, the experiment pins %s" path hex pin)

let load kind ~hashers ~algorithm ~path ~pin : (t, string) result =
  let* () = read_pinned ~hashers ~path ~pin in
  let* aten = Aten_model.of_file ~hashers ~algorithm path in
  let* artifact =
    Digest.Engine_artifact.make Inference_compare_sha256.algorithm ~hex:pin
    |> Result.map_error (fun _ -> "invalid model pin")
  in
  match kind with
  | Aten ->
      Ok
        {
          endpoint =
            Endpoint
              {
                runner = Aten_runner.create ~artifact ();
                build =
                  (fun ex ->
                    Ex.Dialect.realize (module Aten_dialect.Dialect) aten ex);
              };
          identity = Aten_runner.identity ~artifact ();
          graph = aten.key;
          attributes = Aten_model.attributes_of aten;
        }
  | Native | Native4d -> (
      let* lowered, capture =
        Native_interp.lower_archive_captured aten.archive
        |> Result.map_error (fun e ->
            Format.asprintf "lowering: %a" Native_interp.pp_error
              (Err.Error.kind e))
      in
      let* native =
        Native_model.of_lowered ~hashers ~algorithm lowered capture
      in
      match kind with
      | Native4d ->
          let ctx =
            {
              Codecs.Logical_value.Context.codecs =
                Codecs.Element_codec.Registry.builtin;
              hashers;
              algorithm;
              ledger = Codecs.Owned_blob.Ledger.create ();
            }
          in
          let* four =
            Native4d_model.make ~hashers ~algorithm ~ctx ~archive:aten.archive
              ~native ~source_bytes:(fun name ->
                Error ("no case value for " ^ name))
          in
          Ok
            {
              endpoint =
                Endpoint
                  {
                    runner = Native4d_runner.create ~artifact ();
                    build =
                      (fun ex ->
                        Ex.Dialect.realize
                          (module Native4d_dialect.Dialect)
                          four.n4 ex);
                  };
              identity = Native4d_runner.identity ~artifact ();
              graph = four.n4.key;
              attributes = Graph_model.attributes_of four.n4;
            }
      | _ ->
          Ok
            {
              endpoint =
                Endpoint
                  {
                    runner = Native_runner.create ~artifact ();
                    build =
                      (fun ex ->
                        Ex.Dialect.realize
                          (module Native_dialect.Dialect)
                          native ex);
                  };
              identity = Native_runner.identity ~artifact ();
              graph = native.key;
              attributes = Native_model.attributes_of native;
            })
