(* The qemu engines against the host C engines of the same numerics, bit for
   bit, over the chain fixture as an extracted body (every source an input);
   the Coral words' refusals; and what the Coral identity is made of. *)

open Inference_compare_schema
module V = Inference_compare_rv32.Rv32_providers
module Q = Loop_c_embed_compcert_rv32.Compcert_rv32_loader
module Gen = Inference_compare_generated.Generated.Engines
module Tid = Graph_ir.Tensor_id

let body () =
  let g = Native_test.Graph_fixtures.chain () in
  { g with Graph_ir.Graph.input_kinds = Tid.Map.empty }

let inputs ~salt (g : Graph_ir.graph) =
  List.map
    (fun id ->
      let sg = Tid.Map.find id g.tensors in
      ( id,
        Tensor.materialize sg.Tensor_sig.shape (fun c ->
            let k = (Vec6.offset sg.Tensor_sig.shape c :> int) + salt in
            0.25 +. (float_of_int (k mod 9) /. 8.)) ))
    g.inputs

let bits (Tensor.Tensor t as packed) =
  let acc = ref [] in
  Vec6.iter t.Tensor.shape (fun c ->
      acc := Int64.bits_of_float (Tensor.read packed c) :: !acc);
  List.rev !acc

module type ENGINE =
  Inference_compare_pt2.Graph_runner.ENGINE with type op = Graph_ir.op

let outputs (module E : ENGINE) g ~salt =
  match E.compile g with
  | Error m -> Error m
  | Ok c ->
      let r = E.execute c g ~inputs:(inputs ~salt g) in
      E.release c;
      r

let compare name a b =
  let g = body () in
  List.iter
    (fun salt ->
      match (outputs a g ~salt, outputs b g ~salt) with
      | Ok x, Ok y ->
          Printf.printf "%s, salt %d: %s\n" name salt
            (if List.map bits x = List.map bits y then "bitwise" else "DIFFERS")
      | Error m, _ | _, Error m -> Printf.printf "%s: %s\n" name m)
    [ 0; 1; 2 ]

module Rv32_fp32 =
  Inference_compare_rv32.Rv32_engine.Make
    (Q.Ilp32f)
    (struct
      let numerics = Loop_ir.Loop_numerics.Fp32_ordered
    end)

let%expect_test "the F-only qemu engine equals the host binary32 C engine" =
  compare "compcert-rv32-fp32 vs loop-c-fp32"
    (module Rv32_fp32)
    (module Gen.Loop_c_fp32);
  [%expect
    {|
    compcert-rv32-fp32 vs loop-c-fp32, salt 0: bitwise
    compcert-rv32-fp32 vs loop-c-fp32, salt 1: bitwise
    compcert-rv32-fp32 vs loop-c-fp32, salt 2: bitwise |}]

let provider name =
  List.find
    (fun (p : Inference_compare_pt2.Provider.t) -> p.name = name)
    (V.providers ())

let%expect_test "the RISC-V binary64 word refuses before compiling" =
  let p = provider "compcert-rv32" in
  (match p.available () with
  | Error m -> Printf.printf "available: %s\n" m
  | Ok () -> print_endline "unexpectedly available");
  (match p.instantiate () with
  | Error m -> Printf.printf "instantiate: %s\n" m
  | Ok _ -> print_endline "unexpectedly instantiated");
  [%expect
    {|
    available: the RISC-V CompCert route uses only the F32 ABI; add fp32
    instantiate: the RISC-V CompCert route uses only the F32 ABI; add fp32 |}]

let%expect_test "the Coral words refuse binary64 and a missing archive" =
  let show name =
    match (provider name).available () with
    | Ok () -> Printf.printf "%s: available\n" name
    | Error m -> Printf.printf "%s: %s\n" name m
  in
  Unix.putenv "CORALNPU_DATA" "/no/such/archive";
  List.iter show
    [
      "compcert-coral-mpact-v2";
      "compcert-coral-core-mini-highmem-axi";
      "compcert-coral-mpact-v2-fp32";
    ];
  Unix.putenv "CORALNPU_DATA" "";
  show "compcert-coral-mpact-v2-fp32";
  [%expect
    {|
    compcert-coral-mpact-v2: the Coral cores have no D extension, so binary64 does not run there; add fp32
    compcert-coral-core-mini-highmem-axi: the Coral cores have no D extension, so binary64 does not run there; add fp32
    compcert-coral-mpact-v2-fp32: /no/such/archive/hardware-configurations.json: No such file or directory
    compcert-coral-mpact-v2-fp32: CORALNPU_DATA does not name the extracted emulators archive (make coralnpu.download) |}]

(* The identity names the configuration, the simulator's digest and its
   command: changing either changes it. *)
let%expect_test "the Coral identity follows the artifact and the command" =
  let manifest =
    Err.payload
      (Coral_manifest.of_string
         (In_channel.with_open_bin "../coralnpu/fragment.json"
            In_channel.input_all))
  in
  let p =
    match
      Result.bind manifest (fun m ->
          Err.payload (Coral_profile.of_manifest m "tiny-mpact"))
    with
    | Ok p -> p
    | Error _ -> failwith "fragment"
  in
  let id (p : Coral_profile.t) = V.coral_identity ~dir:"/a/v0/b" p in
  let base = id p in
  let digests (i : Engine_identity.t) =
    ( Digest.Configuration.to_string i.configuration_digest,
      Option.map Digest.Engine_artifact.to_string i.artifact_digest )
  in
  let differs what q =
    Printf.printf "%s: configuration %b, artifact %b\n" what
      (fst (digests base) <> fst (digests (id q)))
      (snd (digests base) <> snd (digests (id q)))
  in
  Printf.printf "artifact recorded: %b\n" (base.artifact_digest <> None);
  differs "another artifact" { p with artifact_sha256 = String.make 64 'f' };
  differs "another command" { p with command = p.command @ [ "--trace" ] };
  differs "the same profile" p;
  [%expect
    {|
    artifact recorded: true
    another artifact: configuration false, artifact true
    another command: configuration true, artifact false
    the same profile: configuration false, artifact false |}]

let%expect_test "the RV32 Machine IR engine equals the binary32 C engines" =
  compare "rivet-rv32-fp32 vs loop-c-fp32"
    (module Inference_compare_rv32.Mir_rv32_engine)
    (module Gen.Loop_c_fp32);
  compare "rivet-rv32-fp32 vs compcert-rv32-fp32"
    (module Inference_compare_rv32.Mir_rv32_engine)
    (module Rv32_fp32);
  let p = provider "rivet-rv32" in
  (match p.available () with
  | Error m -> Printf.printf "rivet-rv32: %s\n" m
  | Ok () -> print_endline "unexpectedly available");
  [%expect
    {|
    rivet-rv32-fp32 vs loop-c-fp32, salt 0: bitwise
    rivet-rv32-fp32 vs loop-c-fp32, salt 1: bitwise
    rivet-rv32-fp32 vs loop-c-fp32, salt 2: bitwise
    rivet-rv32-fp32 vs compcert-rv32-fp32, salt 0: bitwise
    rivet-rv32-fp32 vs compcert-rv32-fp32, salt 1: bitwise
    rivet-rv32-fp32 vs compcert-rv32-fp32, salt 2: bitwise
    rivet-rv32: the RV32 Machine IR route has no D extension; add fp32 |}]
