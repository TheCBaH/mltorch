module F = Native_test.Graph_fixtures

(* The converter route against the direct lowering over whole-graph bundles:
   the capability, numerical and structure checks of the route gate. Every row
   is one graph's tally; a refusal is a gap, named by its typed cause. *)

let wide_chain () =
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

let graphs =
  [
    ("chain", F.chain);
    ("diamond", F.diamond);
    ("residual", F.residual);
    ("sink_permute_binary", F.sink_permute_binary);
    ("sink_permute_layer_norm", F.sink_permute_layer_norm);
    ("sink_permute_mean_basic", F.sink_permute_mean_basic);
    ("sink_permute_pad", F.sink_permute_pad);
    ("sink_permute_sdpa", F.sink_permute_sdpa);
    ("sink_permute_slice", F.sink_permute_slice);
    ("wide_chain", wide_chain);
  ]

let%expect_test "the structured route against the direct lowering, per graph" =
  let clean =
    List.map
      (fun (name, g) ->
        let t = Route_gate.bundle (g ()) in
        Fmt.pr "%s: %a" name Route_gate.Tally.pp t;
        Route_gate.Tally.clean t)
      graphs
  in
  Fmt.pr "every comparison agrees: %b@." (List.for_all Fun.id clean);
  [%expect
    {|
    chain: invocations=3 converted=3 refused=0
      f64   compared=3 bitwise=3 oracle=3 contractions=3 vectors=3
      neon  compared=3 bitwise=3 oracle=3 contractions=3 vectors=3
      wasm  compared=3 bitwise=3 oracle=3 contractions=3 vectors=3
      coral compared=3 bitwise=3 oracle=3 contractions=3 vectors=3
    diamond: invocations=3 converted=3 refused=0
      f64   compared=3 bitwise=3 oracle=3 contractions=3 vectors=3
      neon  compared=3 bitwise=3 oracle=3 contractions=3 vectors=3
      wasm  compared=3 bitwise=3 oracle=3 contractions=3 vectors=3
      coral compared=3 bitwise=3 oracle=3 contractions=3 vectors=3
    residual: invocations=3 converted=3 refused=0
      f64   compared=3 bitwise=3 oracle=3 contractions=3 vectors=3
      neon  compared=3 bitwise=3 oracle=3 contractions=3 vectors=3
      wasm  compared=3 bitwise=3 oracle=3 contractions=3 vectors=3
      coral compared=3 bitwise=3 oracle=3 contractions=3 vectors=3
    sink_permute_binary: invocations=5 converted=5 refused=0
      f64   compared=5 bitwise=5 oracle=5 contractions=5 vectors=5
      neon  compared=5 bitwise=5 oracle=5 contractions=5 vectors=5
      wasm  compared=5 bitwise=5 oracle=5 contractions=5 vectors=5
      coral compared=5 bitwise=5 oracle=5 contractions=5 vectors=5
    sink_permute_layer_norm: invocations=2 converted=2 refused=0
      f64   compared=2 bitwise=2 oracle=2 contractions=2 vectors=2
      neon  compared=2 bitwise=2 oracle=2 contractions=2 vectors=2
      wasm  compared=2 bitwise=2 oracle=2 contractions=2 vectors=2
      coral compared=2 bitwise=2 oracle=2 contractions=2 vectors=2
    sink_permute_mean_basic: invocations=2 converted=2 refused=0
      f64   compared=2 bitwise=2 oracle=2 contractions=2 vectors=2
      neon  compared=2 bitwise=2 oracle=2 contractions=2 vectors=2
      wasm  compared=2 bitwise=2 oracle=2 contractions=2 vectors=2
      coral compared=2 bitwise=2 oracle=2 contractions=2 vectors=2
    sink_permute_pad: invocations=2 converted=2 refused=0
      f64   compared=2 bitwise=2 oracle=2 contractions=2 vectors=2
      neon  compared=2 bitwise=2 oracle=2 contractions=2 vectors=2
      wasm  compared=2 bitwise=2 oracle=2 contractions=2 vectors=2
      coral compared=2 bitwise=2 oracle=2 contractions=2 vectors=2
    sink_permute_sdpa: invocations=2 converted=2 refused=0
      f64   compared=2 bitwise=2 oracle=2 contractions=2 vectors=2
      neon  compared=2 bitwise=2 oracle=2 contractions=2 vectors=2
      wasm  compared=2 bitwise=2 oracle=2 contractions=2 vectors=2
      coral compared=2 bitwise=2 oracle=2 contractions=2 vectors=2
    sink_permute_slice: invocations=2 converted=2 refused=0
      f64   compared=2 bitwise=2 oracle=2 contractions=2 vectors=2
      neon  compared=2 bitwise=2 oracle=2 contractions=2 vectors=2
      wasm  compared=2 bitwise=2 oracle=2 contractions=2 vectors=2
      coral compared=2 bitwise=2 oracle=2 contractions=2 vectors=2
    wide_chain: invocations=3 converted=3 refused=0
      f64   compared=3 bitwise=3 oracle=3 contractions=3 vectors=3
      neon  compared=3 bitwise=3 oracle=3 contractions=3 vectors=3
      wasm  compared=3 bitwise=3 oracle=3 contractions=3 vectors=3
      coral compared=3 bitwise=3 oracle=3 contractions=3 vectors=3
    every comparison agrees: true |}]

(* Failure behavior: the same injected failures as the direct lowering's parity
   suite, through the structured route, against the Loop interpreter. A guard
   outside the recognised [Fail_if] pair is a converter refusal: the gap list
   for failures, not a pass. *)
module Failure_cases = struct
  open Loop_ir
  open Loop_ir_test
  open Ssa_bridge_test.Ssa_fixtures

  let structured plan =
    match Err.payload (Loop_lower.lower_structured plan) with
    | Error e -> Error (Fmt.str "Loop_lower: %a" Loop_lower.pp_error e)
    | Ok loop -> (
        match Err.payload (Ssa_of_loop.convert loop) with
        | Ok p -> Ok p
        | Error (`Unsupported u) ->
            Error (Fmt.str "refused: %a" Ssa_of_loop_unsupported.pp u))

  let direct plan =
    Err.or_raise ~pp_error:Ssa_lower.Ssa_lower_plan.pp_error
      (Ssa_lower.Ssa_lower_plan.lower plan)

  let route name plan ~bind =
    let reference =
      match Err.payload (Loop_lower.lower plan) with
      | Error _ -> Fmt.failwith "%s: Loop refused" name
      | Ok loop ->
          Err.map_error
            (fun e -> (e : Loop_interp.error :> Kernel_eval.error))
            (Loop_interp.run loop ~bind)
    in
    let verdict program =
      Fmt.str "%a" Ssa_bridge.Ssa_check.pp_verdict
        (Ssa_bridge.Ssa_check.compare ~reference
           ~ssa:(Ssa_lower.Ssa_exec.run plan program ~bind))
    in
    let table lower =
      match Err.payload (lower plan) with
      | Ok loop -> Loop_js_failure.sites loop
      | Error _ -> [||]
    in
    let optimized = table Loop_lower.lower
    and unoptimized = table Loop_lower.lower_structured in
    let kind (f : Loop_failure.t) =
      match f with
      | Loop_failure.Gather_out_of_range _ -> "gather"
      | Loop_failure.I64_division_by_zero -> "i64_div0"
      | Loop_failure.I64_division_overflow -> "i64_ovf"
      | Loop_failure.I64_from_float _ -> "i64_from_float"
      | Loop_failure.Index_overflow _ -> "index_overflow"
      | Loop_failure.Load_out_of_range _ -> "load"
      | Loop_failure.Local_out_of_range _ -> "local"
      | Loop_failure.Scan_lane_out_of_range _ -> "scan_lane"
      | Loop_failure.Scan_row_out_of_range _ -> "scan_row"
    in
    let kinds a = String.concat "," (List.map kind (Array.to_list a)) in
    let same_table = String.equal (kinds optimized) (kinds unoptimized) in
    Fmt.pr
      "%s:@.  direct:     %s@.  structured: %s@.  sites: %d, failure kinds in \
       the optimized program's order: %b@."
      name
      (verdict (direct plan))
      (match structured plan with Ok p -> verdict p | Error e -> e)
      (Array.length unoptimized) same_table

  let w = Expr.Index.output Expr.Axis.W
  let max_index = Expr.Index.const 0x7FFF_FFFF

  let shifted axis =
    Expr.Index.assume_position
      (Expr.Index.add
         (Expr.Index.of_position (Expr.Index.output axis))
         (Expr.Index.const 1))

  let run () =
    route "w + max"
      (Fusion_plan.default
         (Loop_fixtures.pixel_kernel
            (Expr.Value.value_of_index
               (Expr.Index.add (Expr.Index.of_position w) max_index))))
      ~bind:(bind_data ~shape:(Loop_fixtures.shape_w 4) [| 0.; 0.; 0.; 0. |]);
    let coord =
      Expr.Coord.set
        (Expr.Coord.set Loop_programs.out_coord Expr.Axis.H
           (shifted Expr.Axis.H))
        Expr.Axis.W (shifted Expr.Axis.W)
    in
    route "shifted read"
      (Fusion_plan.default
         (Loop_fixtures.pixel_kernel ~shape:(hw 2 2) (Loop_programs.ld coord)))
      ~bind:(bind_data ~shape:(hw 2 2) [| 1.; 2.; 3.; 4. |]);
    route "pointwise"
      (Fusion_plan.default Loop_programs.kernel)
      ~bind:(bind_data ~shape:(Loop_fixtures.shape_w 4) [| 1.; 2.; 3.; 4. |]);
    route "gather past the source"
      (Fusion_plan.default Loop_programs.gather_kernel)
      ~bind:
        (Loop_programs.i64_bind ~floats:[| 10.; 20.; 30.; 40. |]
           ~cells:[| 4L; 0L; 0L; 0L |]);
    route "int64 x / 0"
      (Fusion_plan.default
         (Loop_programs.i64_body_kernel
            (Expr.Value.i64_div Loop_programs.i64_here (Expr.Value.i64_const 0L))))
      ~bind:
        (Loop_programs.i64_bind ~floats:[| 2.9; -2.9; 0.; 0. |]
           ~cells:[| -7L; 7L; -7L; 7L |])
end

let%expect_test "an injected failure through the direct and structured routes" =
  Failure_cases.run ();
  [%expect
    {|
    w + max:
      direct:     agree on failure: index_overflow
      structured: agree on failure: index_overflow
      sites: 1, failure kinds in the optimized program's order: true
    shifted read:
      direct:     agree on failure: coord_out_of_range
      structured: agree on failure: coord_out_of_range
      sites: 1, failure kinds in the optimized program's order: true
    pointwise:
      direct:     agree
      structured: agree
      sites: 0, failure kinds in the optimized program's order: true
    gather past the source:
      direct:     agree on failure: gather_index_out_of_range
      structured: agree on failure: gather_index_out_of_range
      sites: 1, failure kinds in the optimized program's order: true
    int64 x / 0:
      direct:     agree on failure: i64_division_by_zero
      structured: agree on failure: i64_division_by_zero
      sites: 2, failure kinds in the optimized program's order: true |}]

(* The comparison can fail: one structured store rewritten to a constant is seen
   by the oracle and the direct comparison alike. *)
let mutate_store p =
  let module S = Ssa_ir in
  let changed = ref false in
  let rule state stmt =
    match stmt with
    | S.Ssa_stmt.Instr ({ op = S.Ssa_op.Store store; _ } as instruction)
      when not !changed ->
        changed := true;
        let value = S.Ssa_rewrite.fresh state store.value.S.Ssa_value.ty in
        let constant =
          match value.S.Ssa_value.ty with
          | S.Ssa_type.Scalar S.Ssa_type.F64 -> S.Ssa_const.F64 1.
          | S.Ssa_type.Scalar S.Ssa_type.F32 -> S.Ssa_const.F32 1.
          | _ -> failwith "a float store is required"
        in
        S.Ssa_rewrite.mark_changed state;
        [
          S.Ssa_stmt.Instr
            {
              S.Ssa_instr.results = [ value ];
              op = S.Ssa_op.Const constant;
              token = None;
              origin = S.Ssa_origin.Unknown;
            };
          S.Ssa_stmt.Instr
            { instruction with op = S.Ssa_op.Store { store with value } };
        ]
    | _ -> [ stmt ]
  in
  fst (S.Ssa_rewrite.program rule p)

let%expect_test "a rewritten structured store is seen by the gate" =
  let t = Route_gate.bundle ~rewrite:mutate_store (F.chain ()) in
  Fmt.pr "%a" Route_gate.Tally.pp t;
  Fmt.pr "clean: %b@." (Route_gate.Tally.clean t);
  [%expect
    {|
    invocations=3 converted=3 refused=0
      f64   compared=3 bitwise=0 oracle=0 contractions=3 vectors=3
      neon  compared=3 bitwise=0 oracle=0 contractions=3 vectors=3
      wasm  compared=3 bitwise=0 oracle=0 contractions=3 vectors=3
      coral compared=3 bitwise=0 oracle=0 contractions=3 vectors=3
    clean: false |}]

(* The emitters read the invocation's own failure-site table, made from its
   optimized Loop program; the converted program comes from the unoptimized
   one. Every invocation must still emit, in all three emitters, with the same
   verdict as the direct route. *)
let%expect_test
    "the three emitters take the structured route as they take the direct one" =
  let emitters =
    let open Ssa_backends in
    let pipeline = Pipeline.Exact in
    [
      ( "c",
        fun frontend inv ->
          Result.map ignore (c_via ~frontend ~pipeline ~name:"k" inv) );
      ( "js",
        fun frontend inv -> Result.map ignore (js_via ~frontend ~pipeline inv)
      );
      ( "wasm",
        fun frontend inv ->
          Result.map ignore
            (wasm_via ~frontend ~pipeline
               ~table_alloc:(fun ~bytes:_ ->
                 Loop_ir.Loop_wasm_layout.Address.of_int64 0L)
               inv) );
    ]
  in
  List.iter
    (fun (name, g) ->
      let b =
        Err.or_raise ~pp_error:Loop_ir.Loop_bundle.pp_error
          (Loop_ir.Loop_bundle.build (g ()))
      in
      let sites =
        List.fold_left
          (fun n (inv : Loop_ir.Loop_bundle.invocation) ->
            n
            + Array.length
                (Loop_ir.Loop_js_failure.sites inv.Loop_ir.Loop_bundle.program))
          0 b.Loop_ir.Loop_bundle.invocations
      in
      let agreeing =
        List.map
          (fun (emitter, emit) ->
            let n =
              List.length
                (List.filter
                   (fun inv ->
                     Result.is_ok (emit Ssa_backends.Frontend.Direct inv)
                     = Result.is_ok (emit Ssa_backends.Frontend.Structured inv))
                   b.Loop_ir.Loop_bundle.invocations)
            in
            Fmt.str "%s %d" emitter n)
          emitters
      in
      Fmt.pr "%s: invocations=%d failure_sites=%d same verdict: %s@." name
        (List.length b.Loop_ir.Loop_bundle.invocations)
        sites
        (String.concat ", " agreeing))
    graphs;
  [%expect
    {|
    chain: invocations=3 failure_sites=0 same verdict: c 3, js 3, wasm 3
    diamond: invocations=3 failure_sites=0 same verdict: c 3, js 3, wasm 3
    residual: invocations=3 failure_sites=0 same verdict: c 3, js 3, wasm 3
    sink_permute_binary: invocations=5 failure_sites=0 same verdict: c 5, js 5, wasm 5
    sink_permute_layer_norm: invocations=2 failure_sites=0 same verdict: c 2, js 2, wasm 2
    sink_permute_mean_basic: invocations=2 failure_sites=0 same verdict: c 2, js 2, wasm 2
    sink_permute_pad: invocations=2 failure_sites=0 same verdict: c 2, js 2, wasm 2
    sink_permute_sdpa: invocations=2 failure_sites=0 same verdict: c 2, js 2, wasm 2
    sink_permute_slice: invocations=2 failure_sites=0 same verdict: c 2, js 2, wasm 2
    wide_chain: invocations=3 failure_sites=0 same verdict: c 3, js 3, wasm 3 |}]
