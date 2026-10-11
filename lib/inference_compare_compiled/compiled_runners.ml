(* The two compiled routes as ordinary graph runners over Native bodies, each
   with an identity naming its compiler and loader. *)

open Inference_compare_schema
module P = Inference_compare_pt2

let identity ~engine ~route ~components () : Engine_identity.t =
  let id mk s =
    match mk s with Ok v -> v | Error _ -> invalid_arg "identity"
  in
  let pins =
    String.concat "\n"
      (List.map
         (fun (c : Ancestry.component) -> c.name ^ " = " ^ c.pin)
         components)
  in
  {
    engine = id Id.Engine.of_string engine;
    route = id Id.Engine_route.of_string route;
    build_digest =
      P.Aten_runner.sha_digest Digest.Build.make
        ("inference-compare-compiled/" ^ engine ^ "/v1\n" ^ pins);
    configuration_digest =
      P.Aten_runner.sha_digest Digest.Configuration.make
        ("scalar binary64; isa=aarch64\n" ^ pins);
    artifact_digest = None;
    environment_digest =
      P.Aten_runner.sha_digest Digest.Environment.make
        (Sys.os_type ^ ":" ^ string_of_int Sys.word_size ^ ":aarch64");
  }

let compcert_identity =
  identity ~engine:"compcert-embedded" ~route:"embedded-compcert-aarch64"
    ~components:
      (List.filter
         (fun (c : Ancestry.component) -> c.routes <> `Assembly)
         Ancestry.components)

let assembly_identity =
  identity ~engine:"rivet-assembly" ~route:"direct-assembly-aarch64"
    ~components:
      (List.filter
         (fun (c : Ancestry.component) -> c.routes <> `Compcert)
         Ancestry.components)

module Compcert = P.Graph_runner.Make (Compcert_engine)
module Assembly = P.Graph_runner.Make (Assembly_engine)

let compcert () = Compcert.create ~identity:compcert_identity
let assembly () = Assembly.create ~identity:assembly_identity
