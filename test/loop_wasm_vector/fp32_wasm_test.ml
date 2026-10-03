open Loop_ir

(* The binary32 corpus through Wasm under node, bitwise against the fp32 oracle
   ([Loop_interp.run ~precision:F32]): first every admitted program as a forced
   scalar binary32 kernel, then under the [Simd_fp32_ordered] policy with the
   128-bit vectors (four [f32x4] registers per logical vector). A program the
   policy leaves binary64 is compared with the binary64 interpreter, so the
   unvectorized half is still bitwise the reference. *)

module P = Loop_vector_programs

let extra = Loop_fp32_programs.extra

let zeroed (p : Loop_program.t) id =
  List.find_map
    (fun (b : Loop_buffer.t) ->
      if
        b.Loop_buffer.role = Loop_buffer.Output
        && Tensor_id.equal b.Loop_buffer.id id
      then Some (Loop_interp.allocate b)
      else None)
    p.Loop_program.buffers

let verdict a b =
  match (Err.payload a, Err.payload b) with
  | Ok a, Ok b ->
      if Tensor_id.Map.equal Loop_check.tensors_equal a b then "equal"
      else "DIFFERS"
  | Error _, Error _ -> "both fail"
  | _ -> "ONE FAILED"

let features ?vector ?numerics ?precision p =
  match Err.payload (Loop_wasm.lower ?vector ?numerics ?precision p) with
  | Ok l ->
      String.concat ","
        (List.map Wasm_features.name
           (Wasm_features.of_module l.Loop_wasm.module_))
  | Error e -> Fmt.failwith "%a" Loop_wasm.pp_error e

let forced () =
  List.iter
    (fun (name, p) ->
      match Loop_numerics.admit p with
      | Error r -> Fmt.pr "%-40s refused: %a@." name Loop_numerics.Refusal.pp r
      | Ok () ->
          let precision = Loop_numerics.Precision.F32 in
          let oracle =
            Loop_interp.run ~precision ~outputs:(zeroed p) p ~bind:P.bind
          in
          let w =
            Loop_wasm_exec.exec ~precision ~outputs:(zeroed p) p ~bind:P.bind
          in
          Fmt.pr "%-40s %s@." name (verdict oracle w))
    (P.all @ extra)

let%expect_test "forced binary32 scalar Wasm equals the fp32 oracle" =
  forced ();
  [%expect
    {|
    double rounding                          equal
    arith chain                              equal
    offset view                              equal
    broadcast and invariant                  equal
    strided                                  equal
    maximum                                  equal
    select and compare                       equal
    pool_better                              equal
    not and or                               equal
    index value                              equal
    temporaries                              equal
    sqrt and trunc                           equal
    transcendentals                          equal
    int32 source                             equal
    bool store                               equal
    nested loops, vector inner               equal
    matmul: reduction per output             equal
    reduction with a triangular inner bound  equal
    uniform index temporary                  equal
    extents around the width: 3              equal
    extents around the width: 4              equal
    extents around the width: 5              equal
    matvec: 21 outputs, k 4                  equal
    matmul: 3 rows, 19 outputs, k 4          equal
    matvec: exactly one vector               equal
    int64 to float, once rounded             equal
    index to float                           equal
    erf                                      equal
    log, cos, sqrt                           equal
    packed float scratch                     equal |}]

let policy ~target ~numerics =
  List.iter
    (fun (name, p) ->
      let plan = Loop_plan.resolve ~target ~numerics p in
      let precision = plan.Loop_plan.precision in
      let oracle =
        Loop_interp.run ~precision ~outputs:(zeroed p) p ~bind:P.bind
      in
      let w =
        Loop_wasm_exec.exec ~vector:target ~numerics ~outputs:(zeroed p) p
          ~bind:P.bind
      in
      Fmt.pr "%-40s %s %-8s %s@." name
        (Loop_numerics.Precision.name precision)
        (verdict oracle w)
        (features ~vector:target ~numerics p))
    (P.all @ extra)

let%expect_test "Simd_fp32_ordered on SIMD Wasm: f32x4 kernels, bitwise" =
  policy ~target:Loop_target.wasm128 ~numerics:Loop_numerics.Simd_fp32_ordered;
  [%expect
    {|
    double rounding                          f32 equal    simd128
    arith chain                              f32 equal    simd128
    offset view                              f32 equal    simd128
    broadcast and invariant                  f32 equal    simd128
    strided                                  f64 equal
    maximum                                  f32 equal    simd128
    select and compare                       f32 equal    simd128
    pool_better                              f32 equal    simd128
    not and or                               f32 equal    simd128
    index value                              f32 equal    simd128
    temporaries                              f32 equal    simd128
    sqrt and trunc                           f32 equal    simd128
    transcendentals                          f32 equal    simd128
    int32 source                             f32 equal    simd128
    bool store                               f32 equal    simd128
    nested loops, vector inner               f32 equal    simd128
    matmul: reduction per output             f64 equal    simd128
    reduction with a triangular inner bound  f64 equal    simd128
    uniform index temporary                  f64 equal    simd128
    extents around the width: 3              f64 equal
    extents around the width: 4              f64 equal    simd128
    extents around the width: 5              f64 equal    simd128
    matvec: 21 outputs, k 4                  f32 equal    simd128
    matmul: 3 rows, 19 outputs, k 4          f32 equal    simd128
    matvec: exactly one vector               f32 equal    simd128
    int64 to float, once rounded             f32 equal    simd128
    index to float                           f64 equal
    erf                                      f32 equal    simd128
    log, cos, sqrt                           f32 equal    simd128
    packed float scratch                     f64 equal |}]

let%expect_test "the same with every cost zero" =
  policy
    ~target:(Loop_target.forced Loop_target.wasm128)
    ~numerics:Loop_numerics.Simd_fp32_ordered;
  [%expect
    {|
    double rounding                          f32 equal    simd128
    arith chain                              f32 equal    simd128
    offset view                              f32 equal    simd128
    broadcast and invariant                  f32 equal    simd128
    strided                                  f32 equal    simd128
    maximum                                  f32 equal    simd128
    select and compare                       f32 equal    simd128
    pool_better                              f32 equal    simd128
    not and or                               f32 equal    simd128
    index value                              f32 equal    simd128
    temporaries                              f32 equal    simd128
    sqrt and trunc                           f32 equal    simd128
    transcendentals                          f32 equal    simd128
    int32 source                             f32 equal    simd128
    bool store                               f32 equal    simd128
    nested loops, vector inner               f32 equal    simd128
    matmul: reduction per output             f64 equal    simd128
    reduction with a triangular inner bound  f64 equal    simd128
    uniform index temporary                  f64 equal    simd128
    extents around the width: 3              f64 equal
    extents around the width: 4              f64 equal    simd128
    extents around the width: 5              f64 equal    simd128
    matvec: 21 outputs, k 4                  f32 equal    simd128
    matmul: 3 rows, 19 outputs, k 4          f32 equal    simd128
    matvec: exactly one vector               f32 equal    simd128
    int64 to float, once rounded             f32 equal    simd128
    index to float                           f64 equal
    erf                                      f32 equal    simd128
    log, cos, sqrt                           f32 equal    simd128
    packed float scratch                     f64 equal |}]

(* ---- relaxed: sums scheduled along their own axis ------------------------- *)

let reductions_of (plan : Loop_plan.t) =
  let rec count = function
    | Loop_vector.Reduction _ -> 1
    | Loop_vector.If (_, a, b) -> counts a + counts b
    | Loop_vector.Loop { body; _ } -> counts body
    | Loop_vector.Scalar _ | Loop_vector.Vector _ -> 0
  and counts ns = List.fold_left (fun n x -> n + count x) 0 ns in
  match plan.Loop_plan.vector with
  | None -> 0
  | Some vp -> counts vp.Loop_vector.body

let relaxed ?(bind = P.bind) programs =
  let target = Loop_target.wasm128 in
  let numerics = Loop_numerics.Simd_fp32_relaxed in
  List.iter
    (fun (name, p) ->
      let plan = Loop_plan.resolve ~target ~numerics p in
      let precision = plan.Loop_plan.precision in
      let oracle =
        Loop_interp.run ~precision ~outputs:(zeroed p) (Loop_plan.oracle plan p)
          ~bind
      in
      let w =
        Loop_wasm_exec.exec ~vector:target ~numerics ~outputs:(zeroed p) p ~bind
      in
      Fmt.pr "%-40s %s %-8s %d scheduled sum(s)@." name
        (Loop_numerics.Precision.name precision)
        (verdict oracle w) (reductions_of plan))
    programs

let%expect_test "Simd_fp32_relaxed on SIMD Wasm: scheduled sums, bitwise" =
  relaxed Loop_fp32_programs.dots;
  relaxed ~bind:Loop_fp32_programs.moderate_bind Loop_fp32_programs.dots;
  [%expect
    {|
    dot, 96 terms                            f32 equal    1 scheduled sum(s)
    dot, 83 terms                            f32 equal    1 scheduled sum(s)
    dot, 70 terms                            f32 equal    1 scheduled sum(s)
    dot, 64 terms                            f32 equal    1 scheduled sum(s)
    dot, 96 terms                            f32 equal    1 scheduled sum(s)
    dot, 83 terms                            f32 equal    1 scheduled sum(s)
    dot, 70 terms                            f32 equal    1 scheduled sum(s)
    dot, 64 terms                            f32 equal    1 scheduled sum(s) |}]

let%expect_test "scheduled sums on Wasm are exact on integers" =
  let bind id =
    if
      Tensor_id.equal id (Loop_fixtures.tid 0)
      || Tensor_id.equal id (Loop_fixtures.tid 2)
    then
      Some
        (Loop_fixtures.f32_tensor (Loop_fixtures.shape_w P.big) (fun c ->
             float_of_int
               ((((Vec6.offset (Loop_fixtures.shape_w P.big) c :> int) * 5) + 3)
               mod 7)))
    else None
  in
  List.iter
    (fun (name, p) ->
      let sequential =
        Loop_interp.run ~precision:Loop_numerics.Precision.F32
          ~outputs:(zeroed p) p ~bind
      in
      let w =
        Loop_wasm_exec.exec ~vector:Loop_target.wasm128
          ~numerics:Loop_numerics.Simd_fp32_relaxed ~outputs:(zeroed p) p ~bind
      in
      Fmt.pr "%-20s equals the sequential sum: %s@." name (verdict sequential w))
    Loop_fp32_programs.dots;
  [%expect
    {|
    dot, 96 terms        equals the sequential sum: equal
    dot, 83 terms        equals the sequential sum: equal
    dot, 70 terms        equals the sequential sum: equal
    dot, 64 terms        equals the sequential sum: equal |}]

(* ---- relaxed SIMD: a multiply-add the engine may fuse ---------------------- *)

let%expect_test "relaxed SIMD is found by validating a probe, not by a version"
    =
  Fmt.pr "bare node validates the probe: %b@."
    (Loop_wasm_exec.Node.supports ~flags:false Wasm_features.Relaxed_simd);
  Fmt.pr "node with the feature flag validates the probe: %b@."
    (Loop_wasm_exec.Node.supports Wasm_features.Relaxed_simd);
  [%expect
    {|
    bare node validates the probe: false
    node with the feature flag validates the probe: true |}]

(* Cell by cell, the answer must be one of the two admissible ones: every
   multiply-add fused, or none. *)
let member a ~fused ~unfused =
  Tensor_id.Map.for_all
    (fun id x ->
      let (Tensor.Tensor t) = x in
      let y = Tensor_id.Map.find id fused
      and z = Tensor_id.Map.find id unfused in
      let ok = ref true in
      Vec6.iter t.Tensor.shape (fun c ->
          let u = Tensor.read_at x (Vec6.get c)
          and v = Tensor.read_at y (Vec6.get c)
          and w = Tensor.read_at z (Vec6.get c) in
          let same p q =
            (Float.is_nan p && Float.is_nan q)
            || Int32.bits_of_float p = Int32.bits_of_float q
          in
          if not (same u v || same u w) then ok := false);
      !ok)
    a

let differs a b = not (Tensor_id.Map.equal Loop_check.tensors_equal a b)

let%expect_test "relaxed madd: the result is the fused or the unfused answer" =
  let target = Loop_target.wasm128_relaxed in
  let numerics = Loop_numerics.Simd_fp32_relaxed in
  List.iter
    (fun (name, p) ->
      let plan = Loop_plan.resolve ~target ~numerics p in
      let precision = plan.Loop_plan.precision in
      let run ~fused =
        match
          Err.payload
            (Loop_interp.run ~precision ~fused ~outputs:(zeroed p)
               (Loop_plan.oracle plan p) ~bind:Loop_fp32_programs.moderate_bind)
        with
        | Ok m -> m
        | Error _ -> failwith "oracle"
      in
      let fused = run ~fused:true and unfused = run ~fused:false in
      let w =
        Loop_wasm_exec.exec ~vector:target ~numerics ~outputs:(zeroed p) p
          ~bind:Loop_fp32_programs.moderate_bind
      in
      let features =
        match Err.payload (Loop_wasm.lower ~vector:target ~numerics p) with
        | Ok l ->
            String.concat ","
              (List.map Wasm_features.name
                 (Wasm_features.of_module l.Loop_wasm.module_))
        | Error _ -> "?"
      in
      match Err.payload w with
      | Ok got ->
          Fmt.pr "%-30s %-14s admissible: %b; fused and unfused differ: %b@."
            name features
            (member got ~fused ~unfused)
            (differs fused unfused)
      | Error e -> Fmt.pr "%-30s failed: %a@." name Loop_wasm_exec.pp_error e)
    ((Loop_fp32_programs.multiply_add :: Loop_fp32_programs.dots)
    @ List.filter
        (fun (n, _) -> n = "matvec: 21 outputs, k 4")
        Loop_fp32_programs.extra);
  [%expect
    {|
    multiply-add, 40 cells         relaxed-simd,simd128 admissible: true; fused and unfused differ: true
    dot, 96 terms                  simd128        admissible: true; fused and unfused differ: false
    dot, 83 terms                  simd128        admissible: true; fused and unfused differ: false
    dot, 70 terms                  simd128        admissible: true; fused and unfused differ: false
    dot, 64 terms                  simd128        admissible: true; fused and unfused differ: false
    matvec: 21 outputs, k 4        relaxed-simd,simd128 admissible: true; fused and unfused differ: true |}]

(* ---- register blocking over rows ------------------------------------------ *)

(* The same programs as the C suite: a blocked kernel equals its plan's oracle
   bit for bit and the unblocked plan's oracle too; overlapping rows stay
   unblocked. Wasm plans no blocking by default (it did not pay on whole models
   under V8), so the factor is asked for. *)
let%expect_test "row blocking on SIMD Wasm" =
  let numerics = Loop_numerics.Simd_fp32_relaxed in
  let bind = Loop_fp32_programs.moderate_bind in
  let m = Loop_fp32_programs.matmul in
  let target = Loop_target.with_row_block 2 Loop_target.wasm128 in
  List.iter
    (fun (name, p) ->
      let plan = Loop_plan.resolve ~target ~numerics p in
      let plain = Loop_plan.resolve ~target:Loop_target.wasm128 ~numerics p in
      let precision = plan.Loop_plan.precision in
      let run plan =
        Loop_interp.run ~precision ~outputs:(zeroed p) (Loop_plan.oracle plan p)
          ~bind
      in
      let w =
        Loop_wasm_exec.exec ~vector:target ~numerics ~outputs:(zeroed p) p ~bind
      in
      Fmt.pr
        "%-28s blocked %d (default %d); Wasm vs oracle %s; vs unblocked %s@."
        name plan.Loop_plan.blocked plain.Loop_plan.blocked
        (verdict (run plan) w)
        (verdict (run plain) (run plan)))
    [
      ("5 rows, 17 outputs, k 3", m ~m:5 ~k:3 ~n:17 ());
      ("4 rows, 16 outputs, k 4", m ~m:4 ~k:4 ~n:16 ());
      ("one row", m ~m:1 ~k:4 ~n:16 ());
      ("rows overlap (stride 8)", m ~row_stride:8 ~m:3 ~k:3 ~n:16 ());
    ];
  [%expect
    {|
    5 rows, 17 outputs, k 3      blocked 1 (default 0); Wasm vs oracle equal; vs unblocked equal
    4 rows, 16 outputs, k 4      blocked 1 (default 0); Wasm vs oracle equal; vs unblocked equal
    one row                      blocked 0 (default 0); Wasm vs oracle equal; vs unblocked equal
    rows overlap (stride 8)      blocked 0 (default 0); Wasm vs oracle equal; vs unblocked equal |}]
