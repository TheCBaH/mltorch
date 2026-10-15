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
