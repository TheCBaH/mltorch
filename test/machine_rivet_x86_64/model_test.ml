(* Whole bundles as x86-64 processes: every invocation of a bundle is selected,
   allocated, framed, published and run as a static process (on an x86-64 CPU,
   under qemu-user elsewhere), its results copied back into the context's own
   storage for the next invocation. Each call is compared with the reference
   evaluator, or, under [Fp32_ordered], with the binary32 bundle oracle. *)

module F = Native_test.Graph_fixtures
module T = Machine_model_test.Model_test
module Rt = Machine_rivet_x86_64.Rivet_x64_route

let route = Rt.route ()

let%expect_test "conv, batch norm, relu as processes" =
  T.check ~route ~constant:T.positive ~calls:2 "chain" F.chain;
  T.check ~route ~constant:T.positive "wide chain" T.wide_chain;
  [%expect
    {|
    chain (3 invocations): bitwise, bitwise
    wide chain (3 invocations): bitwise, bitwise |}]

let%expect_test "Fp32_ordered: whole bundles as processes equal the oracle" =
  List.iter
    (fun allocation ->
      let route = Rt.route ~allocation () in
      let name n = Fmt.str "%s %s" n (Rt.Allocation.name allocation) in
      T.check ~route ~pipeline:T.fp32_ordered ~oracle:T.Fp32
        ~constant:T.positive ~calls:2 (name "chain") F.chain;
      T.check ~route ~pipeline:T.fp32_ordered ~oracle:T.Fp32
        ~constant:T.positive (name "wide chain") T.wide_chain;
      T.check ~route ~pipeline:T.fp32_ordered ~oracle:T.Fp32
        (name "layer_norm over W, C") (fun () ->
          F.build "layer_norm"
            Graph_builder.(
              let* x = input ~shape:(F.s 1 1 1 2 4 5) () in
              layer_norm
                { Norm.LayerNorm.dims = [ Axis.W; Axis.C ]; eps = 1e-5 }
                ~x ()));
      T.check ~route ~pipeline:T.fp32_ordered ~oracle:T.Fp32
        (name "softmax over C") (fun () ->
          F.build "softmax"
            Graph_builder.(
              let* x = input ~shape:(F.s 1 1 2 3 4 5) () in
              softmax { Reduce.Softmax.axis = Axis.C } x)))
    Rt.Allocation.[ Reference; Scanned ];
  [%expect
    {|
    chain reference (3 invocations): bitwise, bitwise
    wide chain reference (3 invocations): bitwise, bitwise
    layer_norm over W, C reference (1 invocations): bitwise, bitwise
    softmax over C reference (1 invocations): bitwise, bitwise
    chain scanned (3 invocations): bitwise, bitwise
    wide chain scanned (3 invocations): bitwise, bitwise
    layer_norm over W, C scanned (1 invocations): bitwise, bitwise
    softmax over C scanned (1 invocations): bitwise, bitwise |}]

(* The check can fail: the binary64 reference is not the binary32 answer. *)
let%expect_test "Fp32_ordered is not the binary64 reference" =
  T.check ~route ~pipeline:T.fp32_ordered ~constant:T.positive "chain" F.chain;
  [%expect {| chain (3 invocations): DIFFERS, DIFFERS |}]

(* The policy planned for neon128's 128-bit description, so loops vectorize on
   the exact binary32 form as SSE2 packed singles: whole bundles as processes,
   both allocators, bitwise the binary32 oracle. *)
let%expect_test "Fp32_ordered vectors: whole bundles as processes" =
  List.iter
    (fun allocation ->
      let route = Rt.route ~allocation () in
      List.iter
        (fun (name, g) ->
          T.check ~route ~pipeline:T.fp32_vector ~oracle:T.Fp32
            ~constant:T.positive
            (Fmt.str "%s %s" name (Rt.Allocation.name allocation))
            g)
        (T.vector_fixtures @ T.transformer_fixtures
       @ T.transformer_bf16_fixtures
       @ Machine_model_test.Transformer_fixtures.bf16_out))
    Rt.Allocation.[ Reference; Scanned ];
  [%expect
    {|
    relu then add over C=32 reference (2 invocations): bitwise, bitwise
    softmax over H, W=32 reference (1 invocations): bitwise, bitwise
    layer_norm over H, W=32 reference (1 invocations): bitwise, bitwise
    conv, batch norm, relu to C=32 reference (3 invocations): bitwise, bitwise
    residual add 2x160 reference (1 invocations): bitwise, bitwise
    residual add 1x67 reference (1 invocations): bitwise, bitwise
    residual add 1x33 reference (1 invocations): bitwise, bitwise
    rms norm 2x160 reference (1 invocations): bitwise, bitwise
    rms norm 1x67 reference (1 invocations): bitwise, bitwise
    rms norm 1x33 reference (1 invocations): bitwise, bitwise
    tanh gelu mul 2x160 reference (2 invocations): bitwise, bitwise
    tanh gelu mul 1x67 reference (2 invocations): bitwise, bitwise
    tanh gelu mul 1x33 reference (2 invocations): bitwise, bitwise
    residual add bf16 2x160 reference (1 invocations): bitwise, bitwise
    residual add bf16 1x67 reference (1 invocations): bitwise, bitwise
    residual add bf16 1x33 reference (1 invocations): bitwise, bitwise
    rms norm bf16 2x160 reference (1 invocations): bitwise, bitwise
    rms norm bf16 1x67 reference (1 invocations): bitwise, bitwise
    rms norm bf16 1x33 reference (1 invocations): bitwise, bitwise
    tanh gelu mul bf16 2x160 reference (2 invocations): bitwise, bitwise
    tanh gelu mul bf16 1x67 reference (2 invocations): bitwise, bitwise
    tanh gelu mul bf16 1x33 reference (2 invocations): bitwise, bitwise
    residual add bf16->bf16 2x160 reference (2 invocations): bitwise, bitwise
    residual add bf16->bf16 1x67 reference (2 invocations): bitwise, bitwise
    residual add bf16->bf16 1x33 reference (2 invocations): bitwise, bitwise
    rms norm bf16->bf16 2x160 reference (2 invocations): bitwise, bitwise
    rms norm bf16->bf16 1x67 reference (2 invocations): bitwise, bitwise
    rms norm bf16->bf16 1x33 reference (2 invocations): bitwise, bitwise
    tanh gelu mul bf16->bf16 2x160 reference (3 invocations): bitwise, bitwise
    tanh gelu mul bf16->bf16 1x67 reference (3 invocations): bitwise, bitwise
    tanh gelu mul bf16->bf16 1x33 reference (3 invocations): bitwise, bitwise
    relu then add over C=32 scanned (2 invocations): bitwise, bitwise
    softmax over H, W=32 scanned (1 invocations): bitwise, bitwise
    layer_norm over H, W=32 scanned (1 invocations): bitwise, bitwise
    conv, batch norm, relu to C=32 scanned (3 invocations): bitwise, bitwise
    residual add 2x160 scanned (1 invocations): bitwise, bitwise
    residual add 1x67 scanned (1 invocations): bitwise, bitwise
    residual add 1x33 scanned (1 invocations): bitwise, bitwise
    rms norm 2x160 scanned (1 invocations): bitwise, bitwise
    rms norm 1x67 scanned (1 invocations): bitwise, bitwise
    rms norm 1x33 scanned (1 invocations): bitwise, bitwise
    tanh gelu mul 2x160 scanned (2 invocations): bitwise, bitwise
    tanh gelu mul 1x67 scanned (2 invocations): bitwise, bitwise
    tanh gelu mul 1x33 scanned (2 invocations): bitwise, bitwise
    residual add bf16 2x160 scanned (1 invocations): bitwise, bitwise
    residual add bf16 1x67 scanned (1 invocations): bitwise, bitwise
    residual add bf16 1x33 scanned (1 invocations): bitwise, bitwise
    rms norm bf16 2x160 scanned (1 invocations): bitwise, bitwise
    rms norm bf16 1x67 scanned (1 invocations): bitwise, bitwise
    rms norm bf16 1x33 scanned (1 invocations): bitwise, bitwise
    tanh gelu mul bf16 2x160 scanned (2 invocations): bitwise, bitwise
    tanh gelu mul bf16 1x67 scanned (2 invocations): bitwise, bitwise
    tanh gelu mul bf16 1x33 scanned (2 invocations): bitwise, bitwise
    residual add bf16->bf16 2x160 scanned (2 invocations): bitwise, bitwise
    residual add bf16->bf16 1x67 scanned (2 invocations): bitwise, bitwise
    residual add bf16->bf16 1x33 scanned (2 invocations): bitwise, bitwise
    rms norm bf16->bf16 2x160 scanned (2 invocations): bitwise, bitwise
    rms norm bf16->bf16 1x67 scanned (2 invocations): bitwise, bitwise
    rms norm bf16->bf16 1x33 scanned (2 invocations): bitwise, bitwise
    tanh gelu mul bf16->bf16 2x160 scanned (3 invocations): bitwise, bitwise
    tanh gelu mul bf16->bf16 1x67 scanned (3 invocations): bitwise, bitwise
    tanh gelu mul bf16->bf16 1x33 scanned (3 invocations): bitwise, bitwise |}]

(* The binary32 transcendentals as SSE2 packed sequences, over the F32_math
   sample set, as processes against the binary32 oracle. A NaN compares equal
   to any NaN: x86-64 makes its default NaN with the sign set, where the oracle
   and AArch64 clear it, and payloads are not modelled. *)
let%expect_test "binary32 transcendentals as SSE2 vectors" =
  List.iter
    (fun (name, g) ->
      T.check ~route ~pipeline:T.fp32_vector_forced ~oracle:T.Fp32 ~calls:1
        ~same:T.same_portable ~input:T.sample_input name g)
    T.transcendental_fixtures;
  [%expect
    {|
    exp (1 invocations): bitwise
    log (1 invocations): bitwise
    sin (1 invocations): bitwise
    cos (1 invocations): bitwise
    gelu (erf) (1 invocations): bitwise |}]

(* The binary32 sequences as x86_64 processes (SSE2: the truncations are
   the integer round trip), both allocators, scalar and vector. *)
let%expect_test "binary32 sequences as processes" =
  let module S = Machine_model_test.Sequence_test in
  List.iter
    (fun allocation ->
      let route = Rt.route ~allocation () in
      let a = Rt.Allocation.name allocation in
      S.each ~route ~pipeline:T.fp32_ordered ("scalar " ^ a);
      S.each ~route ~pipeline:T.fp32_vector ("vector " ^ a))
    Rt.Allocation.[ Reference; Scanned ];
  [%expect
    {|
    exp, scalar reference (1 invocations): bitwise
    exp, around each k step, scalar reference (1 invocations): bitwise
    erf, scalar reference (1 invocations): bitwise
    exact gelu, scalar reference (1 invocations): bitwise
    softmax over H, W=32, scalar reference (1 invocations): bitwise
    tanh gelu mul 1x67, scalar reference (2 invocations): bitwise
    tanh gelu mul bf16 1x33, scalar reference (2 invocations): bitwise
    exp, vector reference (1 invocations): bitwise
    exp, around each k step, vector reference (1 invocations): bitwise
    erf, vector reference (1 invocations): bitwise
    exact gelu, vector reference (1 invocations): bitwise
    softmax over H, W=32, vector reference (1 invocations): bitwise
    tanh gelu mul 1x67, vector reference (2 invocations): bitwise
    tanh gelu mul bf16 1x33, vector reference (2 invocations): bitwise
    exp, scalar scanned (1 invocations): bitwise
    exp, around each k step, scalar scanned (1 invocations): bitwise
    erf, scalar scanned (1 invocations): bitwise
    exact gelu, scalar scanned (1 invocations): bitwise
    softmax over H, W=32, scalar scanned (1 invocations): bitwise
    tanh gelu mul 1x67, scalar scanned (2 invocations): bitwise
    tanh gelu mul bf16 1x33, scalar scanned (2 invocations): bitwise
    exp, vector scanned (1 invocations): bitwise
    exp, around each k step, vector scanned (1 invocations): bitwise
    erf, vector scanned (1 invocations): bitwise
    exact gelu, vector scanned (1 invocations): bitwise
    softmax over H, W=32, vector scanned (1 invocations): bitwise
    tanh gelu mul 1x67, vector scanned (2 invocations): bitwise
    tanh gelu mul bf16 1x33, vector scanned (2 invocations): bitwise |}]
