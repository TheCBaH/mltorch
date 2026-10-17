(* Each Coral engine against loop-c-fp32 over the chain fixture as an extracted
   body (every source an input), bit for bit. *)

module Gen = Inference_compare_generated.Generated.Engines
module Tid = Graph_ir.Tensor_id

module type ENGINE =
  Inference_compare_pt2.Graph_runner.ENGINE with type op = Graph_ir.op

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

let outputs (module E : ENGINE) g ~salt =
  match E.compile g with
  | Error m -> Error m
  | Ok c ->
      let r = E.execute c g ~inputs:(inputs ~salt g) in
      E.release c;
      r

let coral configuration : (module ENGINE) =
  let module L = Loop_c_embed_compcert_coral.Compcert_coral_loader.Make (struct
    let dir = Sys.getenv "CORALNPU_DATA"
    let configuration = configuration
    let deadline = 120.
  end) in
  (module Inference_compare_rv32.Rv32_engine.Make
            (L)
            (struct
              let numerics = Loop_ir.Loop_numerics.Fp32_ordered
            end))

let%expect_test "the Coral engines equal loop-c-fp32 bit for bit" =
  let g = body () in
  List.iter
    (fun c ->
      List.iter
        (fun salt ->
          match
            (outputs (coral c) g ~salt, outputs (module Gen.Loop_c_fp32) g ~salt)
          with
          | Ok x, Ok y ->
              Printf.printf "compcert-coral-%s-fp32, salt %d: %s\n" c salt
                (if List.map bits x = List.map bits y then "bitwise"
                 else "DIFFERS")
          | Error m, _ | _, Error m -> Printf.printf "%s: %s\n" c m)
        [ 0; 1 ])
    [ "mpact-v2"; "core-mini-highmem-axi" ];
  [%expect
    {|
    compcert-coral-mpact-v2-fp32, salt 0: bitwise
    compcert-coral-mpact-v2-fp32, salt 1: bitwise
    compcert-coral-core-mini-highmem-axi-fp32, salt 0: bitwise
    compcert-coral-core-mini-highmem-axi-fp32, salt 1: bitwise |}]
