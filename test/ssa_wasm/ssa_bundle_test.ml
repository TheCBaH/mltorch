open Graph_ir
open Loop_ir

(* The whole-model Wasm bundle built through the structured SSA backend, run
   under node against the per-node reference evaluator. *)

let tensor_of ~salt (sg : Tensor_sig.t) =
  let v c =
    float_of_int
      ((((Vec6.offset sg.Tensor_sig.shape c :> int) + salt) mod 7) - 3)
    /. 4.
  in
  Tensor.materialize sg.Tensor_sig.shape v

let positive (sg : Tensor_sig.t) =
  Tensor.materialize sg.Tensor_sig.shape (fun c ->
      0.25
      +. 0.05
         *. float_of_int ((Vec6.offset sg.Tensor_sig.shape c :> int) mod 11))

let bits t =
  let (Tensor.Tensor tt) = t in
  let acc = ref [] in
  Vec6.iter tt.Tensor.shape (fun c ->
      acc := Int32.bits_of_float (Tensor.read_at t (Vec6.get c)) :: !acc);
  List.rev !acc

let max_diff a b =
  let (Tensor.Tensor ta) = a in
  let m = ref 0. in
  Vec6.iter ta.Tensor.shape (fun c ->
      m :=
        Float.max !m
          (Float.abs
             (Tensor.read_at a (Vec6.get c) -. Tensor.read_at b (Vec6.get c))));
  !m

let sig_of_edge (b : Loop_bundle.t) id =
  Tensor_id.Map.find id b.Loop_bundle.graph.Graph.tensors

let map_of l id = List.assoc_opt id l

let with_dir f =
  let dir = Loop_c_exec.Proc.temp_dir "ssa_wasm_bundle_test" in
  Fun.protect
    ~finally:(fun () -> Loop_c_exec.Proc.remove_tree dir)
    (fun () -> f dir)

module H = Loop_wasm_exec.Host

let wide_chain () =
  let module F = Native_test.Graph_fixtures in
  F.build "wide_chain"
    Graph_builder.(
      let* x = input ~shape:(F.nhwc ~h:10 ~w:10 ~c:8) () in
      let* w =
        constant ~shape:(F.weight_shape ~out_channels:16 ~in_channels:8) ()
      in
      let* bias = constant ~shape:(F.s1c 16) () in
      let* gamma = constant ~shape:(F.s1c 16) () in
      let* beta = constant ~shape:(F.s1c 16) () in
      let* mean = constant ~shape:(F.s1c 16) () in
      let* var = constant ~shape:(F.s1c 16) () in
      let* y = conv2d (F.conv_params ~in_channels:8) ~x ~weight:w ~bias () in
      let* n =
        batch_norm F.bn_params ~x:y ~weight:gamma ~bias:beta ~running_mean:mean
          ~running_var:var ()
      in
      relu n)

let run_graph ?kernel ?numerics ~constant g =
  with_dir (fun dir ->
      let b =
        Err.or_raise ~pp_error:Loop_bundle.pp_error
          (Loop_bundle.build ~config:Loop_bundle_wasm.default_config g)
      in
      let constants =
        List.map
          (fun id -> (id, constant (sig_of_edge b id)))
          b.Loop_bundle.constants
      in
      let p =
        Err.or_raise ~pp_error:H.pp_error
          (H.prepare ?numerics ?kernel ~dir b ~constants:(map_of constants))
      in
      let inputs =
        List.map
          (fun id -> (id, tensor_of ~salt:1 (sig_of_edge b id)))
          b.Loop_bundle.inputs
      in
      let reference =
        Err.or_raise ~pp_error:Eval_direct.pp_error
          (Eval_direct.run g ~constants ~inputs)
      in
      let outs =
        Err.or_raise ~pp_error:H.pp_error (H.run p ~bind:(map_of inputs))
      in
      let st = (H.bundle_wasm p).Loop_bundle_wasm.stats in
      ( List.map2
          (fun id t -> (t, Tensor_id.Map.find id reference))
          g.Graph.outputs outs,
        st ))

let target ~numerics t = Ssa_backends.Pipeline.Planned { numerics; target = t }

let%expect_test "the SSA Wasm bundle is bitwise the reference" =
  List.iter
    (fun (name, g) ->
      let g = g () in
      List.iter
        (fun pipeline ->
          let ssa, st =
            run_graph
              ~kernel:(fun ~table_alloc inv ->
                Ssa_backends.wasm ~pipeline ~table_alloc inv)
              ~constant:(tensor_of ~salt:3) g
          in
          Fmt.pr "%s %s: identical: %b (%d invocations, %d kernels)@." name
            (Ssa_backends.Pipeline.name pipeline)
            (List.for_all (fun (t, r) -> bits t = bits r) ssa)
            st.Loop_bundle_wasm.invocations st.Loop_bundle_wasm.distinct_kernels)
        [
          Ssa_backends.Pipeline.Representation;
          Ssa_backends.Pipeline.Exact;
          target ~numerics:Ssa_ir.Ssa_numerics.Reference_f64
            Ssa_ir.Ssa_target.wasm128;
        ])
    [ ("chain", Native_test.Graph_fixtures.chain); ("wide_chain", wide_chain) ];
  [%expect
    {|
    chain representation: identical: true (3 invocations, 3 kernels)
    chain exact: identical: true (3 invocations, 3 kernels)
    chain planned:reference_f64:wasm128: identical: true (3 invocations, 3 kernels)
    wide_chain representation: identical: true (3 invocations, 3 kernels)
    wide_chain exact: identical: true (3 invocations, 3 kernels)
    wide_chain planned:reference_f64:wasm128: identical: true (3 invocations, 3 kernels) |}]

let%expect_test
    "binary32 policies through the SSA Wasm bundle stay within tolerance" =
  List.iter
    (fun (name, loop_numerics, numerics, t) ->
      let g = wide_chain () in
      let ssa, st =
        run_graph ~numerics:loop_numerics
          ~kernel:(fun ~table_alloc inv ->
            Ssa_backends.wasm ~pipeline:(target ~numerics t) ~table_alloc inv)
          ~constant:positive g
      in
      Fmt.pr "%s: %d of %d invocations binary32; within 1e-4: %b@." name
        st.Loop_bundle_wasm.f32_invocations st.Loop_bundle_wasm.invocations
        (List.for_all (fun (t, r) -> max_diff t r < 1e-4) ssa))
    [
      ( "ordered",
        Loop_numerics.Simd_fp32_ordered,
        Ssa_ir.Ssa_numerics.Simd_fp32_ordered,
        Ssa_ir.Ssa_target.wasm128 );
      ( "relaxed",
        Loop_numerics.Simd_fp32_relaxed,
        Ssa_ir.Ssa_numerics.Simd_fp32_relaxed,
        Ssa_ir.Ssa_target.wasm128_relaxed );
    ];
  [%expect
    {|
    ordered: 3 of 3 invocations binary32; within 1e-4: true
    relaxed: 3 of 3 invocations binary32; within 1e-4: true |}]

(* ---- Fp32_ordered: both Wasm emitters equal the binary32 oracle -------------- *)

(* The oracle is the Loop interpreter at binary32 over the bundle; the binary64
   reference is printed beside it, and must differ, or the check would pass on
   any engine. *)
let run_f32 ?kernel ~numerics ~constant g =
  with_dir (fun dir ->
      let b =
        Err.or_raise ~pp_error:Loop_bundle.pp_error
          (Loop_bundle.build ~config:Loop_bundle_wasm.default_config g)
      in
      let constants =
        List.map
          (fun id -> (id, constant (sig_of_edge b id)))
          b.Loop_bundle.constants
      in
      let p =
        Err.or_raise ~pp_error:H.pp_error
          (H.prepare ~numerics ?kernel ~dir b ~constants:(map_of constants))
      in
      let inputs =
        List.map
          (fun id -> (id, tensor_of ~salt:1 (sig_of_edge b id)))
          b.Loop_bundle.inputs
      in
      let oracle =
        Err.or_raise ~pp_error:Loop_bundle_oracle.pp_error
          (Loop_bundle_oracle.run b ~constants:(map_of constants)
             ~inputs:(map_of inputs))
      in
      let reference =
        Err.or_raise ~pp_error:Eval_direct.pp_error
          (Eval_direct.run g ~constants ~inputs)
      in
      let outs =
        Err.or_raise ~pp_error:H.pp_error (H.run p ~bind:(map_of inputs))
      in
      let st = (H.bundle_wasm p).Loop_bundle_wasm.stats in
      ( List.for_all2 (fun t o -> bits t = bits o) outs oracle,
        List.for_all2
          (fun id t -> bits t = bits (Tensor_id.Map.find id reference))
          g.Graph.outputs outs,
        st ))

let%expect_test "Fp32_ordered through both Wasm emitters is the binary32 oracle"
    =
  List.iter
    (fun (name, g) ->
      let g = g () in
      let report engine (same_oracle, same_reference, st) =
        Fmt.pr
          "%s %s: equals the oracle %b, equals the f64 reference %b (%d of %d \
           binary32)@."
          name engine same_oracle same_reference
          st.Loop_bundle_wasm.f32_invocations st.Loop_bundle_wasm.invocations
      in
      report "loop"
        (run_f32 ~numerics:Loop_numerics.Fp32_ordered ~constant:positive g);
      report "ssa"
        (run_f32 ~numerics:Loop_numerics.Fp32_ordered
           ~kernel:(fun ~table_alloc inv ->
             Ssa_backends.wasm
               ~pipeline:
                 (target ~numerics:Ssa_ir.Ssa_numerics.Fp32_ordered
                    Ssa_ir.Ssa_target.scalar)
               ~table_alloc inv)
           ~constant:positive g))
    [ ("chain", Native_test.Graph_fixtures.chain); ("wide_chain", wide_chain) ];
  [%expect
    {|
    chain loop: equals the oracle true, equals the f64 reference false (3 of 3 binary32)
    chain ssa: equals the oracle true, equals the f64 reference false (3 of 3 binary32)
    wide_chain loop: equals the oracle true, equals the f64 reference false (3 of 3 binary32)
    wide_chain ssa: equals the oracle true, equals the f64 reference false (3 of 3 binary32) |}]

(* Mutation: one invocation's kernel left binary64 turns the check red. *)
let%expect_test "a binary64 kernel among binary32 ones is caught" =
  let g = wide_chain () in
  let calls = ref 0 in
  let same_oracle, _, st =
    run_f32 ~numerics:Loop_numerics.Fp32_ordered
      ~kernel:(fun ~table_alloc inv ->
        incr calls;
        let pipeline =
          if !calls = 1 then Ssa_backends.Pipeline.Exact
          else
            target ~numerics:Ssa_ir.Ssa_numerics.Fp32_ordered
              Ssa_ir.Ssa_target.scalar
        in
        Ssa_backends.wasm ~pipeline ~table_alloc inv)
      ~constant:positive g
  in
  Fmt.pr "equals the oracle %b (%d of %d binary32)@." same_oracle
    st.Loop_bundle_wasm.f32_invocations st.Loop_bundle_wasm.invocations;
  [%expect {| equals the oracle false (2 of 3 binary32) |}]
