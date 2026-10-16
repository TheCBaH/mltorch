(* The generated-code routes over Native bodies. Each engine prepares a body by
   building its Loop IR bundle and the artifact (a compiled binary, or a Wasm
   module), and runs it on the inputs bound for the body. The artifact lives in
   a directory of its own, removed when the body is released. Every
   prerequisite and configuration is resolved when the runner is made: the
   identity names the compiler or runtime as found, and a body never prepares
   against a different one. *)

open Inference_compare_schema
module P = Inference_compare_pt2
module Tid = Graph_ir.Tensor_id

let text pp e = Format.asprintf "%a" pp e

let identity ~engine ~route ~configuration ~environment : Engine_identity.t =
  let id mk s =
    match mk s with Ok v -> v | Error _ -> invalid_arg "identity"
  in
  {
    engine = id Id.Engine.of_string engine;
    route = id Id.Engine_route.of_string route;
    build_digest =
      P.Aten_runner.sha_digest Digest.Build.make
        ("inference-compare-generated/" ^ engine ^ "/v1");
    configuration_digest =
      P.Aten_runner.sha_digest Digest.Configuration.make configuration;
    artifact_digest = None;
    environment_digest =
      P.Aten_runner.sha_digest Digest.Environment.make environment;
  }

(* The first line a tool prints for [--version], or why it cannot be had. *)
let banner argv =
  match Loop_c_exec.Proc.run argv with
  | Ok (Loop_c_exec.Proc.Exited 0, out) ->
      Ok
        (match String.index_opt out '\n' with
        | Some i -> String.sub out 0 i
        | None -> out)
  | Ok (st, _) ->
      Error
        (Format.asprintf "%s %a" (String.concat " " argv)
           Loop_c_exec.Proc.pp_status st)
  | Error m -> Error m

let ssa_pipeline = Ssa_backends.Pipeline.Exact

(* The numerics of an engine: the binary64 reference, or [Fp32_ordered], under
   which every kernel is binary32 and the SSA kernels are the planned scalar
   ones. *)
let ssa_pipeline_of = function
  | Loop_ir.Loop_numerics.Fp32_ordered ->
      Ssa_backends.Pipeline.Planned
        {
          numerics = Ssa_ir.Ssa_numerics.Fp32_ordered;
          target = Ssa_ir.Ssa_target.scalar;
        }
  | Loop_ir.Loop_numerics.Reference_f64
  | Loop_ir.Loop_numerics.Simd_fp32_ordered
  | Loop_ir.Loop_numerics.Simd_fp32_relaxed ->
      ssa_pipeline

module type BACKEND = sig
  type prepared

  val prepare : dir:string -> Graph_ir.graph -> (prepared, string) result

  val run :
    prepared ->
    inputs:(Tid.t * Tensor.packed) list ->
    (Tensor.packed list, string) result
end

(* A backend as a graph engine: the artifact directory is the compiled body. *)
module Engine (B : BACKEND) = struct
  type op = Graph_ir.op
  type compiled = { dir : string; prepared : B.prepared }

  let compile g =
    let dir = Loop_c_exec.Proc.temp_dir "compare_generated" in
    match B.prepare ~dir g with
    | Ok prepared -> Ok { dir; prepared }
    | Error m ->
        Loop_c_exec.Proc.remove_tree dir;
        Error m

  let execute c (_ : Graph_ir.graph) ~inputs = B.run c.prepared ~inputs
  let release c = Loop_c_exec.Proc.remove_tree c.dir
end

(* Extracted bodies read every external operand as an input, so a bundle with a
   constant of its own is not one this route can run. *)
let no_constants (_ : Tid.t) = None
let bind inputs id = List.assoc_opt id inputs

module C (X : sig
  val pipeline : Ssa_backends.Pipeline.t option
  val numerics : Loop_ir.Loop_numerics.t
end) =
Engine (struct
  module H = Loop_c_exec.Host

  type prepared = H.prepared

  let prepare ~dir g =
    match
      Loop_ir.Loop_bundle.build ~config:Loop_ir.Loop_bundle_c.default_config g
    with
    | Error e -> Error (text Loop_ir.Loop_bundle.pp_error (Err.Error.kind e))
    | Ok bundle -> (
        let kernel =
          Option.map
            (fun pipeline ~name inv -> Ssa_backends.c ~pipeline ~name inv)
            X.pipeline
        in
        match
          Err.payload
            (H.prepare ~numerics:X.numerics ?kernel ~dir bundle
               ~constants:no_constants)
        with
        | Ok p -> Ok p
        | Error e -> Error (text H.pp_error e))

  let run p ~inputs =
    match Err.payload (H.run p ~bind:(bind inputs)) with
    | Ok o -> Ok o
    | Error e -> Error (text H.pp_error e)
end)

module Wasm (X : sig
  val pipeline : Ssa_backends.Pipeline.t option
  val numerics : Loop_ir.Loop_numerics.t
end) =
Engine (struct
  module H = Loop_wasm_exec.Host

  type prepared = H.prepared

  let prepare ~dir g =
    match
      Loop_ir.Loop_bundle.build ~config:Loop_ir.Loop_bundle_wasm.default_config
        g
    with
    | Error e -> Error (text Loop_ir.Loop_bundle.pp_error (Err.Error.kind e))
    | Ok bundle -> (
        let kernel =
          Option.map
            (fun pipeline ~table_alloc inv ->
              Ssa_backends.wasm ~pipeline ~table_alloc inv)
            X.pipeline
        in
        match
          Err.payload
            (H.prepare ~numerics:X.numerics ?kernel ~dir bundle
               ~constants:no_constants)
        with
        | Ok p -> Ok p
        | Error e -> Error (text H.pp_error e))

  let run p ~inputs =
    match Err.payload (H.run p ~bind:(bind inputs)) with
    | Ok o -> Ok o
    | Error e -> Error (text H.pp_error e)
end)

let reference = Loop_ir.Loop_numerics.Reference_f64
let fp32 = Loop_ir.Loop_numerics.Fp32_ordered

module Loop_c = C (struct
  let pipeline = None
  let numerics = reference
end)

module Ssa_c = C (struct
  let pipeline = Some ssa_pipeline
  let numerics = reference
end)

module Loop_wasm = Wasm (struct
  let pipeline = None
  let numerics = reference
end)

module Ssa_wasm = Wasm (struct
  let pipeline = Some ssa_pipeline
  let numerics = reference
end)

module Loop_c_fp32 = C (struct
  let pipeline = None
  let numerics = fp32
end)

module Ssa_c_fp32 = C (struct
  let pipeline = Some (ssa_pipeline_of fp32)
  let numerics = fp32
end)

module Loop_wasm_fp32 = Wasm (struct
  let pipeline = None
  let numerics = fp32
end)

module Ssa_wasm_fp32 = Wasm (struct
  let pipeline = Some (ssa_pipeline_of fp32)
  let numerics = fp32
end)

let compiler = List.hd Loop_c_exec.Host.default_compiler

let c_available () =
  Result.map ignore (banner [ compiler; "--version" ])
  |> Result.map_error (fun m -> "host C compiler: " ^ m)

let node_command () = !Loop_wasm_exec.Node.node

let wasm_available () =
  Result.map ignore (banner (node_command () @ [ "--version" ]))
  |> Result.map_error (fun m -> "node: " ^ m)

let pipeline_name = function
  | None -> "loop"
  | Some p -> "ssa-" ^ Ssa_backends.Pipeline.name p

let c_instance ?(numerics = reference) ~engine ~pipeline create () =
  let ( let* ) = Result.bind in
  let* version = banner [ compiler; "--version" ] in
  let id =
    identity ~engine ~route:"host-c-subprocess"
      ~configuration:
        (String.concat ";"
           [
             "compiler=" ^ String.concat " " Loop_c_exec.Host.default_compiler;
             "kernels=" ^ pipeline_name pipeline;
             "numerics=" ^ Loop_ir.Loop_numerics.identity numerics;
           ])
      ~environment:(String.concat ":" [ Sys.os_type; version ])
  in
  Ok { P.Provider.runner = create ~identity:(fun () -> id); identity = id }

let wasm_instance ?(numerics = reference) ~engine ~pipeline create () =
  let ( let* ) = Result.bind in
  let command = node_command () in
  let* version = banner (command @ [ "--version" ]) in
  let id =
    identity ~engine ~route:"wasm-node-subprocess"
      ~configuration:
        (String.concat ";"
           [
             "node=" ^ String.concat " " command;
             "kernels=" ^ pipeline_name pipeline;
             "numerics=" ^ Loop_ir.Loop_numerics.identity numerics;
           ])
      ~environment:(String.concat ":" [ Sys.os_type; version ])
  in
  Ok { P.Provider.runner = create ~identity:(fun () -> id); identity = id }

module Loop_c_runner = P.Graph_runner.Make (Loop_c)
module Ssa_c_runner = P.Graph_runner.Make (Ssa_c)
module Loop_wasm_runner = P.Graph_runner.Make (Loop_wasm)
module Ssa_wasm_runner = P.Graph_runner.Make (Ssa_wasm)
module Loop_c_fp32_runner = P.Graph_runner.Make (Loop_c_fp32)
module Ssa_c_fp32_runner = P.Graph_runner.Make (Ssa_c_fp32)
module Loop_wasm_fp32_runner = P.Graph_runner.Make (Loop_wasm_fp32)
module Ssa_wasm_fp32_runner = P.Graph_runner.Make (Ssa_wasm_fp32)

let providers : P.Provider.t list =
  [
    {
      name = "loop-c";
      available = c_available;
      instantiate =
        c_instance ~engine:"loop-c" ~pipeline:None Loop_c_runner.create;
    };
    {
      name = "loop-wasm";
      available = wasm_available;
      instantiate =
        wasm_instance ~engine:"loop-wasm" ~pipeline:None Loop_wasm_runner.create;
    };
    {
      name = "ssa-c";
      available = c_available;
      instantiate =
        c_instance ~engine:"ssa-c" ~pipeline:(Some ssa_pipeline)
          Ssa_c_runner.create;
    };
    {
      name = "ssa-wasm";
      available = wasm_available;
      instantiate =
        wasm_instance ~engine:"ssa-wasm" ~pipeline:(Some ssa_pipeline)
          Ssa_wasm_runner.create;
    };
    {
      name = "loop-c-fp32";
      available = c_available;
      instantiate =
        c_instance ~numerics:fp32 ~engine:"loop-c-fp32" ~pipeline:None
          Loop_c_fp32_runner.create;
    };
    {
      name = "loop-wasm-fp32";
      available = wasm_available;
      instantiate =
        wasm_instance ~numerics:fp32 ~engine:"loop-wasm-fp32" ~pipeline:None
          Loop_wasm_fp32_runner.create;
    };
    {
      name = "ssa-c-fp32";
      available = c_available;
      instantiate =
        c_instance ~numerics:fp32 ~engine:"ssa-c-fp32"
          ~pipeline:(Some (ssa_pipeline_of fp32))
          Ssa_c_fp32_runner.create;
    };
    {
      name = "ssa-wasm-fp32";
      available = wasm_available;
      instantiate =
        wasm_instance ~numerics:fp32 ~engine:"ssa-wasm-fp32"
          ~pipeline:(Some (ssa_pipeline_of fp32))
          Ssa_wasm_fp32_runner.create;
    };
  ]

let register () = List.iter P.Provider.register providers

module Engines = struct
  module Loop_c = Loop_c
  module Loop_wasm = Loop_wasm
  module Ssa_c = Ssa_c
  module Ssa_wasm = Ssa_wasm
end
