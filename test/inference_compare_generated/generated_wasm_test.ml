module E = Inference_compare_generated.Generated.Engines
module Body = Inference_compare_generated_test.Generated_body

let%expect_test "Loop Wasm gives the direct evaluator's bits" =
  let module C = Body.Check (E.Loop_wasm) in
  C.run ();
  [%expect {|
    seed 0: true
    seed 3: true |}]

let%expect_test "SSA Wasm gives the direct evaluator's bits" =
  let module C = Body.Check (E.Ssa_wasm) in
  C.run ();
  [%expect {|
    seed 0: true
    seed 3: true |}]
