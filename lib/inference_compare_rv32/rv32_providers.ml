(* The embedded riscv32 CompCert as variant words. Under qemu-riscv32:
   [compcert-rv32] is CompCert's ilp32d profile on qemu's default CPU (the
   reference binary64 numerics), [compcert-rv32-fp32] its F-only profile with D
   disabled. On the Coral NPU simulators, one word per configuration, binary32
   only: [compcert-coral-<configuration>-fp32]; the word without [fp32] refuses,
   since the Coral cores have no D extension. A Coral word needs the extracted
   emulators archive, named by CORALNPU_DATA; its identity names the
   configuration ID, the simulator's digest and command, and the deadline. *)

open Inference_compare_schema
module P = Inference_compare_pt2
module Q = Loop_c_embed_compcert_rv32.Compcert_rv32_loader
module Profile = Machine_rivet_riscv32.Rv32_profile

let id mk s = match mk s with Ok v -> v | Error _ -> invalid_arg "identity"

(* The pinned components, as the CompCert embedding's lock and the submodules
   record them. *)
let pins =
  String.concat "\n"
    [
      "CompCert riscv32 and riscv32_fp32 = v3.17-4 \
       export-riscv32=f7a943ce3eee7381e98ac2c33ce662f22055a3ed13fd59ba70ce7590e87a22ff \
       export-riscv32_fp32=22e0345ee536d135a90bcae4e5c0d1b8b9ded4f5a947ff57f85bdc25abb4b771 \
       (vendored/rivet-compcert@fccfeaad9bf2489e47da26608d64b412741c838b)";
      "Rivet assembler and image planner = \
       vendored/rivet@4a0ba8d9c6bfdb979e80246c12e3d34629e2af83";
      "Loop IR bundle to C (Compcert_ilp32 dialect) = this repository";
    ]

let identity ?artifact ~engine ~route ~configuration ~environment () :
    Engine_identity.t =
  {
    engine = id Id.Engine.of_string engine;
    route = id Id.Engine_route.of_string route;
    build_digest =
      P.Aten_runner.sha_digest Digest.Build.make
        ("inference-compare-rv32/" ^ engine ^ "/v1\n" ^ pins);
    configuration_digest =
      P.Aten_runner.sha_digest Digest.Configuration.make
        (configuration ^ "\n" ^ pins);
    artifact_digest =
      Option.map
        (fun hex ->
          id
            (fun hex ->
              Err.payload
                (Digest.Engine_artifact.make Inference_compare_sha256.algorithm
                   ~hex))
            hex)
        artifact;
    environment_digest =
      P.Aten_runner.sha_digest Digest.Environment.make environment;
  }

let fp32 = Loop_ir.Loop_numerics.Fp32_ordered
let reference = Loop_ir.Loop_numerics.Reference_f64
let host = Sys.os_type ^ ":" ^ string_of_int Sys.word_size

(* {1 qemu-riscv32} *)

let qemu = Machine_rivet_riscv32.Rv32_process.qemu

(* The first line of [qemu-riscv32 --version], or why it cannot be had. *)
let qemu_banner () =
  match Loop_c_exec.Proc.run [ qemu; "--version" ] with
  | Ok (Loop_c_exec.Proc.Exited 0, out) ->
      Ok
        (match String.index_opt out '\n' with
        | Some i -> String.sub out 0 i
        | None -> out)
  | Ok _ | Error _ -> Error (qemu ^ " --version failed (is it on the PATH?)")

let qemu_available () = Result.map ignore (qemu_banner ())

let qemu_provider ~name ~numerics ~profile (module L : Loop_c_embed.LOADER) :
    P.Provider.t =
  let instantiate () =
    Result.map
      (fun banner ->
        let module R =
          P.Graph_runner.Make
            (Rv32_engine.Make
               (L)
               (struct
                 let numerics = numerics
               end))
        in
        let identity =
          identity ~engine:name ~route:"embedded-compcert-riscv32-qemu"
            ~configuration:
              (String.concat "; "
                 [
                   "numerics=" ^ Loop_ir.Loop_numerics.identity numerics;
                   "profile=" ^ Profile.name profile;
                   "qemu-cpu="
                   ^ Option.value (Profile.qemu_cpu profile) ~default:"default";
                 ])
            ~environment:(host ^ ":" ^ banner)
            ()
        in
        {
          P.Provider.runner = R.create ~identity:(fun () -> identity);
          identity;
        })
      (qemu_banner ())
  in
  { name; available = qemu_available; instantiate }

(* {1 Coral NPU simulators} *)

let coral_configurations = [ "core-mini-highmem-axi"; "mpact-v2" ]
let deadline = 600.
let coral_word c = "compcert-coral-" ^ c

let archive () =
  match Sys.getenv_opt "CORALNPU_DATA" with
  | None | Some "" ->
      Error
        "CORALNPU_DATA does not name the extracted emulators archive (make \
         coralnpu.download)"
  | Some dir -> Ok dir

let profile_of ~dir c =
  Result.map_error
    (Fmt.str "%a" Coral_profile.pp_error)
    (Err.payload (Coral_profile.load ~dir c))

(* What a Coral run is: the configuration (its ID covers the manifest record),
   the simulator binary's digest, the command it is launched with, and the
   deadline of one call. *)
let coral_identity ~dir (p : Coral_profile.t) =
  identity ~artifact:p.artifact_sha256
    ~engine:(coral_word p.name ^ "-fp32")
    ~route:"embedded-compcert-riscv32-fp32-coral"
    ~configuration:
      (String.concat "; "
         [
           "numerics=" ^ Loop_ir.Loop_numerics.identity fp32;
           "profile=" ^ Profile.name Profile.Rv32imf_ilp32f;
           "configuration=" ^ p.configuration_id;
           "command=" ^ String.concat " " p.command;
           Printf.sprintf "deadline=%gs" deadline;
         ])
    ~environment:
      (String.concat ":"
         [
           host; Filename.basename (Filename.dirname dir); Filename.basename dir;
         ])
    ()

let coral_provider c : P.Provider.t =
  let available () =
    Result.bind (archive ()) (fun dir -> Result.map ignore (profile_of ~dir c))
  in
  let instantiate () =
    Result.bind (archive ()) (fun dir ->
        Result.map
          (fun p ->
            let module L =
            Loop_c_embed_compcert_coral.Compcert_coral_loader.Make (struct
              let dir = dir
              let configuration = c
              let deadline = deadline
            end) in
            let module R =
              P.Graph_runner.Make
                (Rv32_engine.Make
                   (L)
                   (struct
                     let numerics = fp32
                   end))
            in
            let identity = coral_identity ~dir p in
            {
              P.Provider.runner = R.create ~identity:(fun () -> identity);
              identity;
            })
          (profile_of ~dir c))
  in
  { name = coral_word c ^ "-fp32"; available; instantiate }

(* Binary64 cannot run on a Coral core: the word parses and refuses. *)
let coral_binary64 c =
  let reason =
    "the Coral cores have no D extension, so binary64 does not run there; add \
     fp32"
  in
  {
    P.Provider.name = coral_word c;
    available = (fun () -> Error reason);
    instantiate = (fun () -> Error reason);
  }

let providers () =
  [
    qemu_provider ~name:"compcert-rv32" ~numerics:reference
      ~profile:Profile.Rv32imafd_ilp32d
      (module Q.Ilp32d);
    qemu_provider ~name:"compcert-rv32-fp32" ~numerics:fp32
      ~profile:Profile.Rv32imf_ilp32f
      (module Q.Ilp32f);
  ]
  @ List.concat_map
      (fun c -> [ coral_binary64 c; coral_provider c ])
      coral_configurations

let register () = List.iter P.Provider.register (providers ())
