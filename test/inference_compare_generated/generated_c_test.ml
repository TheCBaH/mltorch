module E = Inference_compare_generated.Generated.Engines

let%expect_test "Loop C gives the direct evaluator's bits" =
  let module C = Generated_body.Check (E.Loop_c) in
  C.run ();
  [%expect {|
    seed 0: true
    seed 3: true |}]

let%expect_test "SSA C gives the direct evaluator's bits" =
  let module C = Generated_body.Check (E.Ssa_c) in
  C.run ();
  [%expect {|
    seed 0: true
    seed 3: true |}]

module P = Inference_compare_pt2
module V = P.Variant

let identity name =
  match P.Provider.instantiate name with
  | Ok i -> i.identity
  | Error m -> failwith m

let%expect_test "the words are variants and each engine has its own identity" =
  Inference_compare_generated.Generated.register ();
  List.iter
    (fun s ->
      match V.of_string s with
      | Ok v -> print_endline (V.to_string v)
      | Error m -> print_endline m)
    [
      "native:loop-c";
      "native.canonical:ssa-c";
      "native:loop-wasm";
      "native4d:loop-c";
      "aten:loop-c";
      "native:loop-c,arena";
      "native:loop-c,symbolic";
    ];
  Printf.printf "loop-c = ssa-c: %b\n" (identity "loop-c" = identity "ssa-c");
  Printf.printf "loop-c = loop-c: %b\n" (identity "loop-c" = identity "loop-c");
  [%expect
    {|
    native:loop-c
    native.canonical:ssa-c
    native:loop-wasm
    variant "native4d:loop-c": no configuration "loop-c"
    variant "aten:loop-c": no configuration "loop-c"
    variant "native:loop-c,arena": loop-c takes no other word
    variant "native:loop-c,symbolic": no configuration "symbolic"
    loop-c = ssa-c: false
    loop-c = loop-c: true |}]

let%expect_test "a missing prerequisite is a refusal naming it" =
  Inference_compare_generated.Generated.register ();
  let saved = !Loop_wasm_exec.Node.node in
  Loop_wasm_exec.Node.node := [ "no-such-node" ];
  (match P.Provider.instantiate "loop-wasm" with
  | Ok _ -> print_endline "instantiated"
  | Error m -> print_endline m);
  Loop_wasm_exec.Node.node := saved;
  [%expect
    {| loop-wasm is not available: node: no-such-node: No such file or directory |}]
