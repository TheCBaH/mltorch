(* The identity under which a prepared (e.g. compiled) graph may be reused:
   the extracted graph's payload digest -- which covers the body, its embedded
   constants and both id maps -- together with every identity field of the
   engine that prepares it. A change to any of them is a different key. *)

open Inference_compare_schema

let make (ex : Inference_compare_extract.Extraction.t) (e : Engine_identity.t) =
  String.concat "|"
    [
      Digest.Extracted_graph_payload.to_string
        ex.descriptor.graph_payload_digest;
      Id.Engine.to_string e.engine;
      Id.Engine_route.to_string e.route;
      Digest.Build.to_string e.build_digest;
      Digest.Configuration.to_string e.configuration_digest;
      (match e.artifact_digest with
      | Some d -> Digest.Engine_artifact.to_string d
      | None -> "-");
      Digest.Environment.to_string e.environment_digest;
    ]
