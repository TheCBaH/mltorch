module F = Native_test.Graph_fixtures
open Loop_ir

(* The binary64 census of whole-model Wasm modules from both emitters, the Loop
   emitter and the structured SSA emitter, on the same bundles, under the
   binary64 reference and under the policy that exists today. The baseline the
   binary32 policy is measured against: each row is one built module. *)

let bundle g =
  Err.or_raise ~pp_error:Loop_bundle.pp_error
    (Loop_bundle.build ~config:Loop_bundle_wasm.default_config g)

let census (m : Wasm.Module.t) =
  Fmt.str "%a" Wasm_census.pp (Wasm_census.of_module m)

let show name = function
  | Ok (w : Loop_bundle_wasm.t) ->
      Fmt.pr "%s: %s@." name (census w.Loop_bundle_wasm.module_)
  | Error e -> Fmt.pr "%s: refused: %a@." name Loop_bundle_wasm.pp_error e

let fixtures =
  [
    ("chain", F.chain);
    ( "softmax",
      fun () ->
        F.build "softmax"
          Graph_builder.(
            let* x = input ~shape:(F.s 1 1 2 3 4 5) () in
            softmax { Reduce.Softmax.axis = Axis.C } x) );
  ]

let%expect_test "baseline census of both emitters' bundles" =
  List.iter
    (fun (name, g) ->
      let b = bundle (g ()) in
      let ssa pipeline =
        Err.payload
          (Loop_bundle_wasm.build
             ~kernel:(fun ~table_alloc inv ->
               Ssa_backends.wasm ~pipeline ~table_alloc inv)
             b)
      in
      show (name ^ " loop reference") (Err.payload (Loop_bundle_wasm.build b));
      show
        (name ^ " loop simd_fp32_ordered")
        (Err.payload
           (Loop_bundle_wasm.build ~vector:Loop_target.wasm128
              ~numerics:Loop_numerics.Simd_fp32_ordered b));
      show
        (name ^ " loop fp32_ordered")
        (Err.payload
           (Loop_bundle_wasm.build ~numerics:Loop_numerics.Fp32_ordered b));
      show (name ^ " ssa exact") (ssa Ssa_backends.Pipeline.Exact);
      show
        (name ^ " ssa fp32_ordered scalar")
        (ssa
           (Ssa_backends.Pipeline.Planned
              {
                numerics = Ssa_ir.Ssa_numerics.Fp32_ordered;
                target = Ssa_ir.Ssa_target.scalar;
              }));
      show
        (name ^ " ssa simd_fp32_ordered")
        (ssa
           (Ssa_backends.Pipeline.Planned
              {
                numerics = Ssa_ir.Ssa_numerics.Simd_fp32_ordered;
                target = Ssa_ir.Ssa_target.wasm128;
              })))
    fixtures;
  [%expect
    {|
    chain loop reference: arithmetic 13, conversion 12, helper_call 0, memory_move 22; imports 0
    chain loop simd_fp32_ordered: arithmetic 26, conversion 30, helper_call 0, memory_move 37; imports 0
    chain loop fp32_ordered: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    chain ssa exact: arithmetic 13, conversion 18, helper_call 0, memory_move 73; imports 0
    chain ssa fp32_ordered scalar: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    chain ssa simd_fp32_ordered: arithmetic 13, conversion 18, helper_call 0, memory_move 72; imports 0
    softmax loop reference: arithmetic 5, conversion 4, helper_call 2, memory_move 17; imports 1
    softmax loop simd_fp32_ordered: arithmetic 9, conversion 12, helper_call 6, memory_move 29; imports 1
    softmax loop fp32_ordered: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    softmax ssa exact: arithmetic 5, conversion 6, helper_call 2, memory_move 47; imports 1
    softmax ssa fp32_ordered scalar: arithmetic 0, conversion 0, helper_call 0, memory_move 0; imports 0
    softmax ssa simd_fp32_ordered: arithmetic 5, conversion 6, helper_call 2, memory_move 47; imports 1 |}]

(* The gate: under Fp32_ordered a bundle's module holds no binary64 instruction
   of any role and imports nothing, from either emitter. A module that failed
   this would print its census instead of "free". *)
let%expect_test "Fp32_ordered Wasm modules are free of binary64" =
  List.iter
    (fun (name, g) ->
      let b = bundle (g ()) in
      let verdict = function
        | Ok (w : Loop_bundle_wasm.t) ->
            let c = Wasm_census.of_module w.Loop_bundle_wasm.module_ in
            if Wasm_census.total c = 0 && Wasm_census.imports c = [] then "free"
            else Fmt.str "%a" Wasm_census.pp c
        | Error e -> Fmt.str "refused: %a" Loop_bundle_wasm.pp_error e
      in
      Fmt.pr "%s loop: %s@." name
        (verdict
           (Err.payload
              (Loop_bundle_wasm.build ~numerics:Loop_numerics.Fp32_ordered b)));
      Fmt.pr "%s ssa: %s@." name
        (verdict
           (Err.payload
              (Loop_bundle_wasm.build
                 ~kernel:(fun ~table_alloc inv ->
                   Ssa_backends.wasm
                     ~pipeline:
                       (Ssa_backends.Pipeline.Planned
                          {
                            numerics = Ssa_ir.Ssa_numerics.Fp32_ordered;
                            target = Ssa_ir.Ssa_target.scalar;
                          })
                     ~table_alloc inv)
                 b)));
      Fmt.pr "%s ssa vectors: %s@." name
        (verdict
           (Err.payload
              (Loop_bundle_wasm.build
                 ~kernel:(fun ~table_alloc inv ->
                   Ssa_backends.wasm
                     ~pipeline:
                       (Ssa_backends.Pipeline.Planned
                          {
                            numerics = Ssa_ir.Ssa_numerics.Fp32_ordered;
                            target = Ssa_ir.Ssa_target.wasm128;
                          })
                     ~table_alloc inv)
                 b))))
    [
      ("chain", Native_test.Graph_fixtures.chain);
      ( "relu then add over C=32",
        fun () ->
          Native_test.Graph_fixtures.build "pointwise"
            Graph_builder.(
              let* x =
                input ~shape:(Native_test.Graph_fixtures.s 1 1 1 2 3 32) ()
              in
              let* y = relu x in
              add y x) );
      ( "softmax",
        fun () ->
          Native_test.Graph_fixtures.build "softmax"
            Graph_builder.(
              let* x =
                input ~shape:(Native_test.Graph_fixtures.s 1 1 2 3 4 5) ()
              in
              softmax { Reduce.Softmax.axis = Axis.C } x) );
    ];
  [%expect
    {|
    chain loop: free
    chain ssa: free
    chain ssa vectors: free
    relu then add over C=32 loop: free
    relu then add over C=32 ssa: free
    relu then add over C=32 ssa vectors: free
    softmax loop: free
    softmax ssa: free
    softmax ssa vectors: free |}]
