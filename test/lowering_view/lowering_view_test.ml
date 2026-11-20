(* Every family of the lowering view on fixture nodes: a pointwise op, the
   two-output max pool (each output is its own invocation), a Region node, a
   granted node, and an SSA-refused one. *)

module A = Lowering_view.Artifact
module F = Native_test.Graph_fixtures
module K = Lowering_key
module Lv = Lowering_view
module Pass = Loop_ir.Loop_opt.Pass

let build ?granted g =
  Err.or_raise ~pp_error:Loop_ir.Loop_bundle.pp_error
    (Loop_ir.Loop_bundle.build ?granted g)

let relu () =
  F.build "relu"
    Graph_builder.(
      let* x = input ~shape:(F.nhwc ~h:2 ~w:3 ~c:2) () in
      relu x)

let pool () =
  F.build2 "pool"
    Graph_builder.(
      let* x = input ~shape:(F.nhwc ~h:4 ~w:4 ~c:2) () in
      let params : Pool.MaxPool2dWithIndices.params =
        {
          ceil_mode = false;
          kernel = { h = Dim.extent 2; w = Dim.extent 2 };
          stride = { h = Op_config.Pos.of_int 2; w = Op_config.Pos.of_int 2 };
          pad = { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
        }
      in
      max_pool2d_with_indices params x)

let rms_norm () =
  F.build "rms_norm"
    Graph_builder.(
      let* x = input ~shape:(F.s 1 1 1 3 4 5) () in
      rms_norm { Norm.RmsNorm.dims = [ Axis.C ]; eps = 1e-5 } ~x ())

let nobody _ = false
let invocations g = (build g).Loop_ir.Loop_bundle.invocations
let only g = List.hd (invocations g)
let spec ?numerics pipeline = K.Ssa_spec.{ default with pipeline; numerics }
let ssa_route ?numerics pipeline = K.Route.Ssa (spec ?numerics pipeline)

(* The text, or the refusal, then the notes. *)
let show ?(lines = max_int) inv variant =
  let a = Lv.of_invocation ~granted:nobody inv variant in
  Fmt.pr "== %s@." (K.name variant);
  (match a.A.body with
  | A.Refused r -> Fmt.pr "refused: %a@." A.Refusal.pp r
  | A.Text { language; text; truncated } ->
      let ls = String.split_on_char '\n' text in
      Fmt.pr "[%s%s, %d lines]@." (A.Language.name language)
        (if truncated then ", truncated" else "")
        (List.length ls);
      List.iteri (fun i l -> if i < lines then print_endline l) ls);
  List.iter (fun n -> Fmt.pr "note: %a@." A.Note.pp n) a.A.notes

(* One line per variant: emits (lines) or refuses. *)
let status inv variants =
  List.iter
    (fun v ->
      let a = Lv.of_invocation ~granted:nobody inv v in
      match a.A.body with
      | A.Text { text; _ } ->
          Fmt.pr "%-28s emits %d lines@." (K.name v)
            (List.length (String.split_on_char '\n' text))
      | A.Refused r -> Fmt.pr "%-28s refused: %a@." (K.name v) A.Refusal.pp r)
    variants

let every_family =
  [
    K.default K.Loop;
    K.default K.Js;
    K.default K.C;
    K.default K.Wat;
    K.default K.Ssa;
    K.Js (K.Route.Ssa K.Ssa_spec.default);
    K.C (K.Route.Ssa K.Ssa_spec.default);
    K.Wat (K.Route.Ssa K.Ssa_spec.default);
  ]

let%expect_test
    "a pointwise node: Loop, JavaScript and C, as the drivers run them" =
  let inv = only (relu ()) in
  show inv (K.default K.Loop);
  show inv (K.default K.Js);
  show inv (K.default K.C);
  [%expect
    {|
    == loop
    [loop, 5 lines]
    in t0 f32 [H=2 W=3 C=2]
    out t1 f32 [H=2 W=3 C=2]
    for i0 in [0, 12):
      x0 = load t0[@i0]
      store t1[@i0] = f32(round_f32((x0 < 0 ? 0 : x0)))
    == js
    [js, 10 lines]
    "use strict";
    function loop_kernel(b0, b1) {
      let x0 = 0;
      for (let i0 = 0; i0 < 12; i0++) {
        x0 = b0[i0];
        b1[i0] = x0 < 0 ? 0 : x0;
      }
      return null;
    }

    == c
    [c, 22 lines]
    static int kernel(struct model_error *err, double *local, const float *b0, float *b1) {
      (void)err; (void)local;
      (void)b0;
      (void)b1;
      double x0 = (0x0p+0);
      (void)x0;
      {
        for (int64_t i0 = ((int64_t)0); i0 < ((int64_t)12); i0 += 4) {
          v4df vt0 = vf_splat((0x0p+0)); (void)vt0;
          vt0 = vf_load_f32((b0 + i0));
          vf_store_f32(b1 + i0, vf_round_f32(vf_sel((vt0 < vf_splat((0x0p+0))), vf_splat((0x0p+0)), vt0)));
        }
      {
        for (int64_t i0 = ((int64_t)12); i0 < ((int64_t)12); i0++) {
          x0 = (double)b0[i0];
          b1[i0] = (float)((x0 < (0x0p+0)) ? (0x0p+0) : x0);
        }
      }
      }
      return 0;
    }

    note: numerics: simd_fp32_relaxed
    note: vector target: neon128
    note: working precision: f64
    note: helpers: erf_approx, float_max, vector_prelude |}]

let%expect_test "a pointwise node: WebAssembly and the SSA program" =
  let inv = only (relu ()) in
  show ~lines:12 inv (K.default K.Wat);
  show inv (K.default K.Ssa);
  [%expect
    {|
    == wat
    [wat, 112 lines]
    (module
      (memory 1)
      (func 0 (param i32 i32 i32) (result i32)
        (local v128 v128 i32 f64)
        i32.const 0
        local.set 5
        block
          loop
            local.get 5
            i32.const 12
            i32.ge_s
            br_if 1
    note: numerics: simd_fp32_relaxed
    note: vector target: wasm128
    note: working precision: f64
    == ssa
    [ssa, 34 lines]
    buffer b0 in f32 [1, 1, 1, 2, 1, 6]
    buffer b1 out f32 [1, 1, 1, 2, 1, 6]
    entry(%0:effect) {
      (%1:index) = const 0:index
      (%2:index) = const 2:index
      (%3:f64) = const 0x0p+0:f64
      (%4:index) = const 6:index
      (%5:index) = const 4:index
      (%6:vec<x4,f64>) = vec.splat x4 %3
      (%7:effect) = for %1 to %2 step 1 iter(%8:index, %9:effect := %0) {
        (%10:effect) = for %1 to %5 step 4 iter(%11:index, %12:effect := %9) {
          (%13:vec<x4,f64>, %14:effect) = vec.load.f32_to_f64 b0[%1, %1, %1, %8, %1, %11] step [0, 0, 0, 0, 0, 1] x4 effect %12
          (%15:mask<x4>) = lanes float.compare.lt %13, %6
          (%16:vec<x4,f64>, %17:effect) = vec.load.f32_to_f64 b0[%1, %1, %1, %8, %1, %11] step [0, 0, 0, 0, 0, 1] x4 effect %14
          (%18:vec<x4,f64>) = lanes select %15, %6, %16
          (%19:vec<x4,f32>) = lanes convert.f64_to_f32 %18
          (%20:vec<x4,f64>) = lanes convert.f32_to_f64 %19
          (%21:effect) = vec.store.f32_round b1[%1, %1, %1, %8, %1, %11] step [0, 0, 0, 0, 0, 1] x4, %20 effect %17
          yield %21
        }
        (%22:effect) = for %5 to %4 step 1 iter(%23:index, %24:effect := %10) {
          (%25:f64, %26:effect) = load.in_bounds.f32_to_f64 b0[%1, %1, %1, %8, %1, %23] effect %24
          (%27:pred) = float.compare.lt %25, %3
          (%28:f64) = select %27, %3, %25
          (%29:f32) = convert.f64_to_f32 %28
          (%30:f64) = convert.f32_to_f64 %29
          (%31:effect) = store.f32_round b1[%1, %1, %1, %8, %1, %23], %30 effect %26
          yield %31
        }
        yield %22
      }
      yield %7
    }

    note: numerics: simd_fp32_relaxed
    note: SSA target: neon128
    note: working precision: f64
    note: vectorization: vectorized: 1 loops, 12 iterations |}]

let%expect_test "a pointwise node through the SSA route" =
  let inv = only (relu ()) in
  show inv (K.Js (K.Route.Ssa K.Ssa_spec.default));
  show inv (K.C (ssa_route K.Pipeline.Exact));
  show ~lines:10 inv (K.Wat (ssa_route K.Pipeline.Exact));
  [%expect
    {|
    == js:ssa
    [js, 34 lines]
    "use strict";
    function loop_kernel(b0, b1) {
      let v0 = 0;
      let v1 = 0;
      let v2 = 0;
      let v3 = 0;
      let v4 = 0;
      let v5 = 0;
      let v6 = 0;
      let v7 = false;
      let v8 = 0;
      let v9 = 0;
      let v10 = 0;
      v0 = 0;
      v1 = 2;
      v2 = 0;
      v3 = 6;
      const t0 = Math.max(0, v1 - v0);
      for (let t1 = 0; t1 < t0; t1++) {
        v4 = v0 + t1 * 1;
        const t2 = Math.max(0, v3 - v0);
        for (let t3 = 0; t3 < t2; t3++) {
          v5 = v0 + t3 * 1;
          v6 = b0[((((v0 * 1 + v0) * 1 + v0) * 2 + v4) * 1 + v0) * 6 + v5];
          v7 = v6 < v2;
          v8 = v7 ? v2 : v6;
          v9 = Math.fround(v8);
          v10 = v9;
          b1[((((v0 * 1 + v0) * 1 + v0) * 2 + v4) * 1 + v0) * 6 + v5] = v10;
        }
      }
      return null;
    }

    note: numerics: reference_f64
    note: SSA target: scalar
    == c:ssa/exact
    [c, 56 lines]
    static int kernel(struct model_error *err, double *local, const float *b0, float *b1) {
      (void)err; (void)local;
      (void)b0;
      (void)b1;
      int64_t v0 = 0;
      int64_t v1 = 0;
      int64_t v2 = 0;
      double v3 = 0;
      int64_t v4 = 0;
      int64_t v5 = 0;
      int64_t v6 = 0;
      double v7 = 0;
      int v8 = 0;
      double v9 = 0;
      float v10 = 0;
      double v11 = 0;
      (void)v0;
      (void)v1;
      (void)v2;
      (void)v3;
      (void)v4;
      (void)v5;
      (void)v6;
      (void)v7;
      (void)v8;
      (void)v9;
      (void)v10;
      (void)v11;
      v0 = INT64_C(0);
      v1 = INT64_C(2);
      v2 = INT64_C(3);
      v3 = (0x0p+0);
      {
        const int64_t t0 = v1;
        for (v4 = v0; v4 < t0; v4 += 1) {
          {
            const int64_t t1 = v2;
            for (v5 = v0; v5 < t1; v5 += 1) {
              {
                const int64_t t2 = v1;
                for (v6 = v0; v6 < t2; v6 += 1) {
                  v7 = (double)b0[(((((v0 * INT64_C(1) + v0) * INT64_C(1) + v0) * INT64_C(2) + v4) * INT64_C(3) + v5) * INT64_C(2) + v6)];
                  v8 = (v7 < v3);
                  v9 = (v8 ? v3 : v7);
                  v10 = (float)v9;
                  v11 = (double)v10;
                  b1[(((((v0 * INT64_C(1) + v0) * INT64_C(1) + v0) * INT64_C(2) + v4) * INT64_C(3) + v5) * INT64_C(2) + v6)] = (float)v11;
                }
              }
            }
          }
        }
      }
      return 0;
    }

    note: working precision: f64
    note: helpers: f32_prelude
    == wat:ssa/exact
    [wat, 133 lines]
    (module
      (memory 1)
      (func 0 (param i32 i32 i32) (result i32)
        (local i32 i32 i32 f64 i32 i32 i32 f64 i32 f64 f32 f64)
        i32.const 0
        local.set 3
        i32.const 2
        local.set 4
        i32.const 3
        local.set 5
    note: working precision: f64 |}]

let%expect_test "the pass subset and the numerics reach the emitters" =
  let inv = only (relu ()) in
  let raw = K.Loop_spec.make [] in
  show inv (K.Loop raw);
  show inv (K.Loop (K.Loop_spec.make [ Pass.Unit_loops ]));
  let text spec =
    match (Lv.of_invocation ~granted:nobody inv (K.Loop spec)).A.body with
    | A.Text { text; _ } -> text
    | A.Refused _ -> "refused"
  in
  let texts =
    List.map text
      [
        K.Loop_spec.make [];
        K.Loop_spec.make [ Pass.Unit_loops ];
        K.Loop_spec.default;
      ]
  in
  Fmt.pr "three subsets, three texts: %b@."
    (List.length (List.sort_uniq String.compare texts) = 3);
  show inv (K.Js (K.Route.Loop raw));
  show inv
    (K.C
       (K.Route.Loop
          (K.Loop_spec.make ~numerics:Loop_ir.Loop_numerics.Reference_f64
             Pass.all)));
  show inv
    (K.Loop (K.Loop_spec.make ~numerics:Loop_ir.Loop_numerics.Reference_f64 []));
  [%expect
    {|
    == loop:raw
    [loop, 9 lines]
    in t0 f32 [H=2 W=3 C=2]
    out t1 f32 [H=2 W=3 C=2]
    for i0 in [0, 1):
      for i1 in [0, 1):
        for i2 in [0, 1):
          for i3 in [0, 2):
            for i4 in [0, 3):
              for i5 in [0, 2):
                store t1[i0,i1,i2,i3,i4,i5] = f32(round_f32((load t0[i0,i1,i2,i3,i4,i5] < 0 ? 0 : load t0[i0,i1,i2,i3,i4,i5])))
    == loop:unit_loops
    [loop, 6 lines]
    in t0 f32 [H=2 W=3 C=2]
    out t1 f32 [H=2 W=3 C=2]
    for i0 in [0, 2):
      for i1 in [0, 3):
        for i2 in [0, 2):
          store t1[0,0,0,i0,i1,i2] = f32(round_f32((load t0[0,0,0,i0,i1,i2] < 0 ? 0 : load t0[0,0,0,i0,i1,i2])))
    three subsets, three texts: true
    == js:loop/raw
    [js, 18 lines]
    "use strict";
    function loop_kernel(b0, b1) {
      for (let i0 = 0; i0 < 1; i0++) {
        for (let i1 = 0; i1 < 1; i1++) {
          for (let i2 = 0; i2 < 1; i2++) {
            for (let i3 = 0; i3 < 2; i3++) {
              for (let i4 = 0; i4 < 3; i4++) {
                for (let i5 = 0; i5 < 2; i5++) {
                  b1[2 * (3 * (2 * (i0 + i1 + i2) + i3) + i4) + i5] = b0[2 * (3 * (2 * (i0 + i1 + i2) + i3) + i4) + i5] < 0 ? 0 : b0[2 * (3 * (2 * (i0 + i1 + i2) + i3) + i4) + i5];
                }
              }
            }
          }
        }
      }
      return null;
    }

    == c:loop/reference_f64
    [c, 15 lines]
    static int kernel(struct model_error *err, double *local, const float *b0, float *b1) {
      (void)err; (void)local;
      (void)b0;
      (void)b1;
      double x0 = (0x0p+0);
      (void)x0;
      {
        for (int64_t i0 = ((int64_t)0); i0 < ((int64_t)12); i0++) {
          x0 = (double)b0[i0];
          b1[i0] = (float)((x0 < (0x0p+0)) ? (0x0p+0) : x0);
        }
      }
      return 0;
    }

    note: numerics: reference_f64
    note: working precision: f64
    == loop:raw/reference_f64
    [loop, 9 lines]
    in t0 f32 [H=2 W=3 C=2]
    out t1 f32 [H=2 W=3 C=2]
    for i0 in [0, 1):
      for i1 in [0, 1):
        for i2 in [0, 1):
          for i3 in [0, 2):
            for i4 in [0, 3):
              for i5 in [0, 2):
                store t1[i0,i1,i2,i3,i4,i5] = f32(round_f32((load t0[i0,i1,i2,i3,i4,i5] < 0 ? 0 : load t0[i0,i1,i2,i3,i4,i5])))
    note: numerics reference_f64 does not apply to this family |}]

let%expect_test "a policy a family does not run is a typed refusal" =
  let inv = only (relu ()) in
  let fp32 = Loop_ir.Loop_numerics.Simd_fp32_relaxed in
  show inv (K.Js (K.Route.Loop (K.Loop_spec.make ~numerics:fp32 Pass.all)));
  show inv (K.Js (ssa_route ~numerics:fp32 K.Pipeline.Planned));
  show inv (K.Asm K.Target.X86_64);
  [%expect
    {|
    == js:loop/simd_fp32_relaxed
    refused: the javascript backend does not run the simd_fp32_relaxed policy
    == js:ssa/simd_fp32_relaxed
    refused: the javascript backend does not run the simd_fp32_relaxed policy
    == asm
    refused: family asm is not supported |}]

let%expect_test "the two outputs of a max pool are two invocations" =
  let invs = invocations (pool ()) in
  Fmt.pr "%d invocations@." (List.length invs);
  List.iter
    (fun (inv : Loop_ir.Loop_bundle.invocation) ->
      Fmt.pr "outputs: %d@." (List.length inv.Loop_ir.Loop_bundle.outputs);
      show inv (K.default K.Loop))
    invs;
  [%expect
    {|
    2 invocations
    outputs: 1
    == loop
    [loop, 14 lines]
    in t0 f32 [H=4 W=4 C=2]
    out t1 f32 [H=2 W=2 C=2]
    for i0 in [0, 2):
      for i1 in [0, 2):
        for i2 in [0, 2):
          x0 = -inf
          o0 = 0
          for i3 in [(2 * i0), ((2 * i0) + 2)):
            for i4 in [(2 * i1), ((2 * i1) + 2)):
              x1 = load t0[0,0,0,i3,i4,i2]
              if pool_better(x0, x1):
                x0 = x1
                o0 = ((4 * i3) + i4)
          store t1[0,0,0,i0,i1,i2] = f32(round_f32(x0))
    outputs: 1
    == loop
    [loop, 16 lines]
    in t0 f32 [H=4 W=4 C=2]
    out t2 i64 [H=2 W=2 C=2]
    for i0 in [0, 2):
      for i1 in [0, 2):
        for i2 in [0, 2):
          x0 = -inf
          o0 = 0
          for i3 in [(2 * i0), ((2 * i0) + 2)):
            for i4 in [(2 * i1), ((2 * i1) + 2)):
              x1 = load t0[0,0,0,i3,i4,i2]
              if pool_better(x0, x1):
                x0 = x1
                o0 = ((4 * i3) + i4)
          x2 = float_of_index(o0)
          fail_if (!(x2 == x2) || (x2 < -9.2233720368547758e+18 || !(x2 < 9.2233720368547758e+18))) -> i64_from_float(x2)
          store t2[0,0,0,i0,i1,i2] = i64(float_to_i64(x2)) |}]

let%expect_test "a Region node emits or refuses in every family" =
  let inv = only (rms_norm ()) in
  status inv every_family;
  show ~lines:14 inv (K.default K.Loop);
  show ~lines:6 inv (K.default K.Ssa);
  [%expect
    {|
    loop                         emits 18 lines
    js                           emits 20 lines
    c                            emits 37 lines
    wat                          emits 268 lines
    ssa                          emits 60 lines
    js:ssa                       emits 73 lines
    c:ssa                        emits 163 lines
    wat:ssa                      emits 463 lines
    == loop
    [loop, 18 lines]
    in t0 f32 [H=3 W=4 C=5]
    in t4 f32 [C=5]
    out t5 f32 [H=3 W=4 C=5]
    alloc a0 : float64[2]
    for i0 in [0, 12):
      mark key
      x0 = 0
      for i1 in [0, 5):
        mark reduction
        x1 = load t0[@((5 * i0) + i1)]
        x0 = (x0 + (x1 * x1))
      a0[0] = x0
      mark local
      a0[1] = (1 / sqrt(((a0[0] / 5) + 1.0000000000000001e-05)))
    == ssa
    [ssa, 60 lines]
    buffer b0 in f32 [1, 1, 1, 1, 12, 5]
    buffer b4 in f32 [1, 1, 1, 1, 1, 5]
    buffer b5 out f32 [1, 1, 1, 1, 12, 5]
    entry(%0:effect) {
      (%1:local, %2:effect) = local.alloc 2 effect %0
      (%3:index) = const 0:index
    note: numerics: simd_fp32_relaxed
    note: SSA target: neon128
    note: working precision: f64
    note: vectorization: vectorized: 1 loops, 60 iterations |}]

let%expect_test "a granted node shows no body in any family" =
  let module L = Native_test.Attention_layer_fixtures in
  let g =
    L.node
      (L.with_norm L.L.Norm.Kind.Layer L.L.Placement.Pre
         {
           (L.all_flags L.small) with
           L.Config.rope = L.Rope_form.Shared;
           mask = true;
         })
  in
  let node = (List.hd g.Graph_ir.Graph.nodes).Graph_ir.Node.id in
  let granted = Graph_ir.Node_id.equal node in
  let b = build ~granted g in
  let inv = List.hd b.Loop_ir.Loop_bundle.invocations in
  List.iter
    (fun v ->
      let a = Lv.of_node ~granted b node v in
      match a.A.body with
      | A.Refused (A.Refusal.Granted n) ->
          Fmt.pr "%-18s granted: %b@." (K.name v)
            (Graph_ir.Node_id.equal n node)
      | A.Refused _ | A.Text _ -> Fmt.pr "%-18s NOT granted@." (K.name v))
    every_family;
  (* the same node, not granted by the caller, is an ordinary invocation *)
  Fmt.pr "ungranted by the caller: %s@."
    (match (Lv.of_invocation ~granted:nobody inv (K.default K.Loop)).A.body with
    | A.Text _ -> "emits"
    | A.Refused _ -> "refused");
  [%expect
    {|
    loop               granted: true
    js                 granted: true
    c                  granted: true
    wat                granted: true
    ssa                granted: true
    js:ssa             granted: true
    c:ssa              granted: true
    wat:ssa            granted: true
    ungranted by the caller: emits |}]

let%expect_test "an invocation SSA cannot lower is a typed refusal" =
  (* no graph fixture reaches the converter's own refusals (the route census
     converts every invocation), so the placed kernel is swapped for one the
     structured lowering already refuses *)
  let inv = only (relu ()) in
  let inv =
    {
      inv with
      Loop_ir.Loop_bundle.placed =
        Fusion_plan.default Loop_ir_test.Loop_programs.i64_load_of_f32_kernel;
    }
  in
  status inv every_family;
  let a = Lv.of_invocation ~granted:nobody inv (K.default K.Ssa) in
  (match a.A.body with
  | A.Refused (A.Refusal.Ssa_lowering (Ssa_backends.Refusal.Loop_lowering _)) ->
      Fmt.pr "typed: Ssa_lowering (Loop_lowering _)@."
  | A.Refused _ | A.Text _ -> Fmt.pr "untyped@.");
  [%expect
    {|
    loop                         emits 5 lines
    js                           emits 10 lines
    c                            emits 22 lines
    wat                          emits 112 lines
    ssa                          refused: Loop_lower: t1: load of format f32 is not lowered
    js:ssa                       refused: Loop_lower: t1: load of format f32 is not lowered
    c:ssa                        refused: Loop_lower: t1: load of format f32 is not lowered
    wat:ssa                      refused: Loop_lower: t1: load of format f32 is not lowered
    typed: Ssa_lowering (Loop_lowering _) |}]

let%expect_test "of_node finds an invocation by its node" =
  let b = build (relu ()) in
  let node =
    (List.hd b.Loop_ir.Loop_bundle.invocations).Loop_ir.Loop_bundle.node
  in
  let ok = Lv.of_node ~granted:nobody b node (K.default K.Loop) in
  let missing = Graph_ir.Node_id.of_int 4242 in
  let bad = Lv.of_node ~granted:nobody b missing (K.default K.Loop) in
  Fmt.pr "%a@.%a@." K.pp_key ok.A.key A.pp bad;
  [%expect
    {|
    native/n0/loop
    native/n4242/loop
    refused: the bundle has no invocation for n4242 |}]

(* ---- Machine IR: every target and stage of a pointwise node ---- *)

let machine_ir target stage =
  K.Machine_ir K.Mir_spec.{ target; stage }

let%expect_test "Machine IR: every target and stage" =
  let inv = only (relu ()) in
  status inv
    (List.concat_map
       (fun t -> List.map (machine_ir t) K.Mir_stage.all)
       K.Target.all);
  [%expect {|
    machine_ir:aarch64/allocated emits 160 lines
    machine_ir:aarch64/generic   emits 84 lines
    machine_ir:aarch64           emits 139 lines
    machine_ir:aarch64/scanned   emits 76 lines
    machine_ir:aarch64/selected  emits 59 lines
    machine_ir:riscv32/allocated emits 171 lines
    machine_ir:riscv32/generic   emits 84 lines
    machine_ir:riscv32           emits 141 lines
    machine_ir:riscv32/scanned   emits 70 lines
    machine_ir:riscv32/selected  emits 57 lines
    machine_ir:riscv32_grouped/allocated refused: the reference allocator does not allocate register groups (allocated)
    machine_ir:riscv32_grouped/generic emits 84 lines
    machine_ir:riscv32_grouped   refused: the reference allocator does not allocate register groups (realized)
    machine_ir:riscv32_grouped/scanned emits 70 lines
    machine_ir:riscv32_grouped/selected emits 57 lines
    machine_ir:allocated         emits 185 lines
    machine_ir:generic           emits 84 lines
    machine_ir                   emits 152 lines
    machine_ir:scanned           emits 94 lines
    machine_ir:selected          emits 65 lines |}]

let%expect_test "Machine IR: realized x86-64 head and notes; typed refusals" =
  let inv = only (relu ()) in
  show ~lines:3 inv (machine_ir K.Target.X86_64 K.Mir_stage.Realized);
  let typed v =
    match (Lv.of_invocation ~granted:nobody inv v).A.body with
    | A.Refused (A.Refusal.Mir_stage _) -> Fmt.pr "typed: Mir_stage _@."
    | A.Refused _ | A.Text _ -> Fmt.pr "untyped@."
  in
  typed (machine_ir K.Target.Riscv32_grouped K.Mir_stage.Realized);
  [%expect {|
    == machine_ir
    [machine_ir, 152 lines]
    target x86_64 [sse2]
    fn0 main func kernel() -> (eax) frame 200 {
    bb0():
    note: numerics: fp32_ordered
    note: SSA target: scalar
    note: x86_64 (Intel SDM 325462 084) features [sse2], numerics=fp32_ordered vectorized=f32 other=f32 fma=none reassociation=none/scalar/row_block=1/forbidden; symbols: 2 bound, 1 bss, 0 rodata, 0 external [], 1 function; 2 relocations
    typed: Mir_stage _ |}]

let%expect_test "Machine IR: an SSA-refused node is a typed Mir_lowering" =
  let inv = only (relu ()) in
  let inv =
    {
      inv with
      Loop_ir.Loop_bundle.placed =
        Fusion_plan.default Loop_ir_test.Loop_programs.i64_load_of_f32_kernel;
    }
  in
  let a = Lv.of_invocation ~granted:nobody inv (K.default K.Machine_ir) in
  Fmt.pr "%a@." A.pp a;
  (match a.A.body with
  | A.Refused (A.Refusal.Mir_lowering _) -> Fmt.pr "typed: Mir_lowering _@."
  | A.Refused _ | A.Text _ -> Fmt.pr "untyped@.");
  [%expect {|
    native/n0/machine_ir
    refused: invocation 0 (n0): Loop_lower: t1: load of format f32 is not lowered
    typed: Mir_lowering _ |}]
