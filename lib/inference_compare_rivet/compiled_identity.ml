(* The identity of a compiled route: its engine and route names, the
   configuration it was built with, and the pinned components it is made of.
   Every field is an argument, so no route's identity is another's with a
   different label. *)

open Inference_compare_schema
module P = Inference_compare_pt2

type component = { name : string; pin : string }

let make ~engine ~route ~configuration ~environment ~components :
    Engine_identity.t =
  let pins =
    String.concat "\n" (List.map (fun c -> c.name ^ " = " ^ c.pin) components)
  in
  {
    engine = Id.Engine.of_literal engine;
    route = Id.Engine_route.of_literal route;
    build_digest =
      P.Aten_runner.sha_digest Digest.Build.make
        ("inference-compare-compiled/" ^ engine ^ "/v1\n" ^ pins);
    configuration_digest =
      P.Aten_runner.sha_digest Digest.Configuration.make
        (configuration ^ "\n" ^ pins);
    artifact_digest = None;
    environment_digest =
      P.Aten_runner.sha_digest Digest.Environment.make environment;
  }
