open Loop_ir

(* The vector corpus as forced-binary32 scalar C, bitwise against the fp32
   oracle ([Loop_interp.run ~precision:F32]). A program [Loop_numerics.admit]
   refuses is reported, not skipped silently. *)

let zeroed (p : Loop_program.t) id =
  List.find_map
    (fun (b : Loop_buffer.t) ->
      if
        b.Loop_buffer.role = Loop_buffer.Output
        && Tensor_id.equal b.Loop_buffer.id id
      then Some (Loop_interp.allocate b)
      else None)
    p.Loop_program.buffers

let contains s sub =
  let n = String.length sub in
  let rec go i =
    i + n <= String.length s && (String.sub s i n = sub || go (i + 1))
  in
  go 0

(* The kernel function alone, minus the scratch parameter: the only [double] an
   fp32 kernel may name. *)
let kernel_text p =
  match
    Err.payload
      (Loop_c.kernel ~precision:Loop_numerics.Precision.F32 ~name:"k" p)
  with
  | Ok k ->
      Some
        (Str.global_replace
           (Str.regexp_string "double *local")
           "" k.Loop_c.source)
  | Error _ -> None

let verdict a b =
  match (Err.payload a, Err.payload b) with
  | Ok a, Ok b ->
      if Tensor_id.Map.equal Loop_check.tensors_equal a b then "equal"
      else "DIFFERS"
  | Error _, Error _ -> "both fail"
  | _ -> "ONE FAILED"

module P = Loop_vector_programs

let extra = Loop_fp32_programs.extra

let corpus () =
  List.iter
    (fun (name, p) ->
      let bind = P.bind in
      match Loop_numerics.admit p with
      | Error r -> Fmt.pr "%-40s refused: %a@." name Loop_numerics.Refusal.pp r
      | Ok () ->
          let oracle =
            Loop_interp.run ~precision:Loop_numerics.Precision.F32
              ~outputs:(zeroed p) p ~bind
          in
          let c =
            Loop_c_exec.exec ~precision:Loop_numerics.Precision.F32
              ~outputs:(zeroed p) p ~bind
          in
          let double_free =
            match kernel_text p with
            | Some t -> not (contains t "double")
            | None -> false
          in
          Fmt.pr "%-40s %-8s %s@." name (verdict oracle c)
            (if double_free then "no widening" else "double helper"))
    (P.all @ extra)

let%expect_test "every admitted vector program: fp32 C equals the fp32 oracle" =
  corpus ();
  [%expect
    {|
    double rounding                          equal    no widening
    arith chain                              equal    no widening
    offset view                              equal    no widening
    broadcast and invariant                  equal    no widening
    strided                                  equal    no widening
    maximum                                  equal    no widening
    select and compare                       equal    no widening
    pool_better                              equal    no widening
    not and or                               equal    no widening
    index value                              equal    no widening
    temporaries                              equal    no widening
    sqrt and trunc                           equal    no widening
    transcendentals                          equal    double helper
    int32 source                             equal    no widening
    bool store                               equal    no widening
    nested loops, vector inner               equal    no widening
    matmul: reduction per output             equal    no widening
    reduction with a triangular inner bound  equal    no widening
    uniform index temporary                  equal    no widening
    extents around the width: 3              equal    no widening
    extents around the width: 4              equal    no widening
    extents around the width: 5              equal    no widening
    matvec: 21 outputs, k 4                  equal    no widening
    matmul: 3 rows, 19 outputs, k 4          equal    no widening
    matvec: exactly one vector               equal    no widening
    int64 to float, once rounded             equal    no widening
    index to float                           equal    no widening
    erf                                      equal    no widening
    log, cos, sqrt                           equal    double helper
    packed float scratch                     equal    no widening |}]

(* ---- the policy: fp32 vector kernels, f64 for the rest --------------------- *)

let precision_of_plan ~target ~numerics p =
  (Loop_plan.resolve ~target ~numerics p).Loop_plan.precision

let policy_corpus ~target ~numerics =
  List.iter
    (fun (name, p) ->
      let bind = P.bind in
      let precision = precision_of_plan ~target ~numerics p in
      let oracle = Loop_interp.run ~precision ~outputs:(zeroed p) p ~bind in
      let c =
        Loop_c_exec.exec ~vector:target ~numerics ~outputs:(zeroed p) p ~bind
      in
      let vector_code =
        match Err.payload (Loop_c_exec.source ~vector:target ~numerics p) with
        | Ok (text, _) -> contains text "vs_load" || contains text "vs_splat"
        | Error _ -> false
      in
      Fmt.pr "%-40s %s %-8s %s@." name
        (Loop_numerics.Precision.name precision)
        (verdict oracle c)
        (if vector_code then "f32 vectors" else "no f32 vectors"))
    (P.all @ extra)

let%expect_test
    "Simd_fp32_ordered: vectorized kernels binary32, bitwise against the oracle"
    =
  policy_corpus ~target:Loop_target.neon128
    ~numerics:Loop_numerics.Simd_fp32_ordered;
  [%expect
    {|
    double rounding                          f32 equal    f32 vectors
    arith chain                              f32 equal    f32 vectors
    offset view                              f32 equal    f32 vectors
    broadcast and invariant                  f32 equal    f32 vectors
    strided                                  f64 equal    no f32 vectors
    maximum                                  f32 equal    f32 vectors
    select and compare                       f32 equal    f32 vectors
    pool_better                              f32 equal    f32 vectors
    not and or                               f32 equal    f32 vectors
    index value                              f32 equal    f32 vectors
    temporaries                              f32 equal    f32 vectors
    sqrt and trunc                           f32 equal    f32 vectors
    transcendentals                          f32 equal    f32 vectors
    int32 source                             f32 equal    f32 vectors
    bool store                               f32 equal    f32 vectors
    nested loops, vector inner               f32 equal    f32 vectors
    matmul: reduction per output             f64 equal    no f32 vectors
    reduction with a triangular inner bound  f64 equal    no f32 vectors
    uniform index temporary                  f64 equal    no f32 vectors
    extents around the width: 3              f64 equal    no f32 vectors
    extents around the width: 4              f64 equal    no f32 vectors
    extents around the width: 5              f64 equal    no f32 vectors
    matvec: 21 outputs, k 4                  f32 equal    f32 vectors
    matmul: 3 rows, 19 outputs, k 4          f32 equal    f32 vectors
    matvec: exactly one vector               f32 equal    f32 vectors
    int64 to float, once rounded             f32 equal    f32 vectors
    index to float                           f64 equal    no f32 vectors
    erf                                      f32 equal    f32 vectors
    log, cos, sqrt                           f32 equal    f32 vectors
    packed float scratch                     f64 equal    no f32 vectors |}]

let%expect_test "the same with every cost zero" =
  policy_corpus
    ~target:(Loop_target.forced Loop_target.neon128)
    ~numerics:Loop_numerics.Simd_fp32_ordered;
  [%expect
    {|
    double rounding                          f32 equal    f32 vectors
    arith chain                              f32 equal    f32 vectors
    offset view                              f32 equal    f32 vectors
    broadcast and invariant                  f32 equal    f32 vectors
    strided                                  f32 equal    f32 vectors
    maximum                                  f32 equal    f32 vectors
    select and compare                       f32 equal    f32 vectors
    pool_better                              f32 equal    f32 vectors
    not and or                               f32 equal    f32 vectors
    index value                              f32 equal    f32 vectors
    temporaries                              f32 equal    f32 vectors
    sqrt and trunc                           f32 equal    f32 vectors
    transcendentals                          f32 equal    f32 vectors
    int32 source                             f32 equal    f32 vectors
    bool store                               f32 equal    f32 vectors
    nested loops, vector inner               f32 equal    f32 vectors
    matmul: reduction per output             f64 equal    no f32 vectors
    reduction with a triangular inner bound  f64 equal    no f32 vectors
    uniform index temporary                  f64 equal    no f32 vectors
    extents around the width: 3              f64 equal    no f32 vectors
    extents around the width: 4              f64 equal    no f32 vectors
    extents around the width: 5              f64 equal    no f32 vectors
    matvec: 21 outputs, k 4                  f32 equal    f32 vectors
    matmul: 3 rows, 19 outputs, k 4          f32 equal    f32 vectors
    matvec: exactly one vector               f32 equal    f32 vectors
    int64 to float, once rounded             f32 equal    f32 vectors
    index to float                           f64 equal    no f32 vectors
    erf                                      f32 equal    f32 vectors
    log, cos, sqrt                           f32 equal    f32 vectors
    packed float scratch                     f64 equal    no f32 vectors |}]

let%expect_test "inner loops: reductions per lane in binary32" =
  policy_corpus
    ~target:(Loop_target.with_inner_loops true Loop_target.neon128)
    ~numerics:Loop_numerics.Simd_fp32_ordered;
  [%expect
    {|
    double rounding                          f32 equal    f32 vectors
    arith chain                              f32 equal    f32 vectors
    offset view                              f32 equal    f32 vectors
    broadcast and invariant                  f32 equal    f32 vectors
    strided                                  f64 equal    no f32 vectors
    maximum                                  f32 equal    f32 vectors
    select and compare                       f32 equal    f32 vectors
    pool_better                              f32 equal    f32 vectors
    not and or                               f32 equal    f32 vectors
    index value                              f32 equal    f32 vectors
    temporaries                              f32 equal    f32 vectors
    sqrt and trunc                           f32 equal    f32 vectors
    transcendentals                          f32 equal    f32 vectors
    int32 source                             f32 equal    f32 vectors
    bool store                               f32 equal    f32 vectors
    nested loops, vector inner               f32 equal    f32 vectors
    matmul: reduction per output             f64 equal    no f32 vectors
    reduction with a triangular inner bound  f64 equal    no f32 vectors
    uniform index temporary                  f64 equal    no f32 vectors
    extents around the width: 3              f64 equal    no f32 vectors
    extents around the width: 4              f64 equal    no f32 vectors
    extents around the width: 5              f64 equal    no f32 vectors
    matvec: 21 outputs, k 4                  f32 equal    f32 vectors
    matmul: 3 rows, 19 outputs, k 4          f32 equal    f32 vectors
    matvec: exactly one vector               f32 equal    f32 vectors
    int64 to float, once rounded             f32 equal    f32 vectors
    index to float                           f64 equal    no f32 vectors
    erf                                      f32 equal    f32 vectors
    log, cos, sqrt                           f32 equal    f32 vectors
    packed float scratch                     f64 equal    no f32 vectors |}]

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

(* The scalar interpreter on [Loop_plan.oracle] -- the plan's vector program
   spelled out lane by lane, scheduled sums as the accumulator statements their
   definition gives -- is the answer; the compiled C must equal it bit for bit. *)
let relaxed_corpus ?(bind = P.bind) ?fuse_reductions ~target programs =
  List.iter
    (fun (name, p) ->
      let numerics = Loop_numerics.Simd_fp32_relaxed in
      let plan = Loop_plan.resolve ~target ~numerics p in
      let precision = plan.Loop_plan.precision in
      let oracle =
        Loop_interp.run ~precision ~outputs:(zeroed p) (Loop_plan.oracle plan p)
          ~bind
      in
      let c =
        Loop_c_exec.exec ~vector:target ~numerics ?fuse_reductions
          ~outputs:(zeroed p) p ~bind
      in
      let fused =
        match
          Err.payload
            (Loop_c_exec.source ~vector:target ~numerics ?fuse_reductions p)
        with
        | Ok (text, _) -> contains text "fmaf(" || contains text "vs_fma("
        | Error _ -> false
      in
      Fmt.pr "%-40s %s %-8s %d scheduled sum(s)%s@." name
        (Loop_numerics.Precision.name precision)
        (verdict oracle c) (reductions_of plan)
        (if fused then ", fused" else ""))
    programs

let%expect_test "Simd_fp32_relaxed: a long sum is scheduled along its own axis"
    =
  relaxed_corpus ~target:Loop_target.neon128
    (P.all @ extra @ Loop_fp32_programs.dots);
  [%expect
    {|
    double rounding                          f32 equal    0 scheduled sum(s)
    arith chain                              f32 equal    0 scheduled sum(s)
    offset view                              f32 equal    0 scheduled sum(s)
    broadcast and invariant                  f32 equal    0 scheduled sum(s), fused
    strided                                  f64 equal    0 scheduled sum(s)
    maximum                                  f32 equal    0 scheduled sum(s)
    select and compare                       f32 equal    0 scheduled sum(s)
    pool_better                              f32 equal    0 scheduled sum(s)
    not and or                               f32 equal    0 scheduled sum(s)
    index value                              f32 equal    0 scheduled sum(s)
    temporaries                              f32 equal    0 scheduled sum(s)
    sqrt and trunc                           f32 equal    0 scheduled sum(s)
    transcendentals                          f32 equal    0 scheduled sum(s)
    int32 source                             f32 equal    0 scheduled sum(s)
    bool store                               f32 equal    0 scheduled sum(s)
    nested loops, vector inner               f32 equal    0 scheduled sum(s)
    matmul: reduction per output             f64 equal    0 scheduled sum(s)
    reduction with a triangular inner bound  f64 equal    0 scheduled sum(s)
    uniform index temporary                  f64 equal    0 scheduled sum(s)
    extents around the width: 3              f64 equal    0 scheduled sum(s)
    extents around the width: 4              f64 equal    0 scheduled sum(s)
    extents around the width: 5              f64 equal    0 scheduled sum(s)
    matvec: 21 outputs, k 4                  f32 equal    0 scheduled sum(s), fused
    matmul: 3 rows, 19 outputs, k 4          f32 equal    0 scheduled sum(s), fused
    matvec: exactly one vector               f32 equal    0 scheduled sum(s), fused
    int64 to float, once rounded             f32 equal    0 scheduled sum(s)
    index to float                           f64 equal    0 scheduled sum(s)
    erf                                      f32 equal    0 scheduled sum(s)
    log, cos, sqrt                           f32 equal    0 scheduled sum(s), fused
    packed float scratch                     f64 equal    0 scheduled sum(s)
    dot, 96 terms                            f32 equal    1 scheduled sum(s)
    dot, 83 terms                            f32 equal    1 scheduled sum(s)
    dot, 70 terms                            f32 equal    1 scheduled sum(s)
    dot, 64 terms                            f32 equal    1 scheduled sum(s) |}]

(* Small integers are summed exactly in binary32 in any order, so a scheduled sum
   must equal the plain sequential one here whatever its tree: a dropped or
   duplicated term (a lost tail, a leftover vector added twice) changes the
   answer without any tolerance to hide in. *)
let integer_bind id =
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

let%expect_test "scheduled sums are exact on integers: no term lost or repeated"
    =
  List.iter
    (fun (name, p) ->
      let plan =
        Loop_plan.resolve ~target:Loop_target.neon128
          ~numerics:Loop_numerics.Simd_fp32_relaxed p
      in
      let sequential =
        Loop_interp.run ~precision:Loop_numerics.Precision.F32
          ~outputs:(zeroed p) p ~bind:integer_bind
      in
      let c =
        Loop_c_exec.exec ~vector:Loop_target.neon128
          ~numerics:Loop_numerics.Simd_fp32_relaxed ~outputs:(zeroed p) p
          ~bind:integer_bind
      in
      Fmt.pr "%-20s %d scheduled; equals the sequential sum: %s@." name
        (reductions_of plan) (verdict sequential c))
    Loop_fp32_programs.dots;
  [%expect
    {|
    dot, 96 terms        1 scheduled; equals the sequential sum: equal
    dot, 83 terms        1 scheduled; equals the sequential sum: equal
    dot, 70 terms        1 scheduled; equals the sequential sum: equal
    dot, 64 terms        1 scheduled; equals the sequential sum: equal |}]

let%expect_test
    "scheduled sums, moderate values: C equals the oracle bit for bit" =
  relaxed_corpus ~bind:Loop_fp32_programs.moderate_bind
    ~target:Loop_target.neon128
    (Loop_fp32_programs.dots @ [ Loop_fp32_programs.multiply_add ]);
  [%expect
    {|
    dot, 96 terms                            f32 equal    1 scheduled sum(s)
    dot, 83 terms                            f32 equal    1 scheduled sum(s)
    dot, 70 terms                            f32 equal    1 scheduled sum(s)
    dot, 64 terms                            f32 equal    1 scheduled sum(s)
    multiply-add, 40 cells                   f32 equal    0 scheduled sum(s), fused |}]

(* The fused accumulate is off in the default plan (it measured slower) but is a
   permission with a definition and an oracle like the rest: opted into, the
   compiled sum equals the oracle, which fuses each step with [fma32]. *)
let%expect_test "scheduled sums with fused accumulates: C equals the oracle" =
  relaxed_corpus ~fuse_reductions:true ~bind:Loop_fp32_programs.moderate_bind
    ~target:Loop_target.neon128 Loop_fp32_programs.dots;
  [%expect
    {|
    dot, 96 terms                            f32 equal    1 scheduled sum(s), fused
    dot, 83 terms                            f32 equal    1 scheduled sum(s), fused
    dot, 70 terms                            f32 equal    1 scheduled sum(s), fused
    dot, 64 terms                            f32 equal    1 scheduled sum(s), fused |}]

(* ---- register blocking over rows ------------------------------------------ *)

(* A blocked kernel is checked three ways: compiled C equals its plan's oracle
   bit for bit; the blocked oracle equals the unblocked plan's, so blocking
   changed no cell; and a program whose rows overlap is left unblocked. *)
let blocking_corpus ~target programs =
  List.iter
    (fun (name, p) ->
      let numerics = Loop_numerics.Simd_fp32_relaxed in
      let bind = Loop_fp32_programs.moderate_bind in
      let plan = Loop_plan.resolve ~target ~numerics p in
      let plain =
        Loop_plan.resolve
          ~target:(Loop_target.with_row_block 1 target)
          ~numerics p
      in
      let precision = plan.Loop_plan.precision in
      let run plan =
        Loop_interp.run ~precision ~outputs:(zeroed p) (Loop_plan.oracle plan p)
          ~bind
      in
      let c =
        Loop_c_exec.exec ~vector:target ~numerics ~outputs:(zeroed p) p ~bind
      in
      Fmt.pr "%-34s blocked %d; C vs oracle %s; vs unblocked %s@." name
        plan.Loop_plan.blocked
        (verdict (run plan) c)
        (verdict (run plain) (run plan)))
    programs

let blocking_programs =
  let m = Loop_fp32_programs.matmul in
  [
    ("5 rows, 17 outputs, k 3", m ~m:5 ~k:3 ~n:17 ());
    ("4 rows, 16 outputs, k 4", m ~m:4 ~k:4 ~n:16 ());
    ("one row", m ~m:1 ~k:4 ~n:16 ());
    ("rows overlap (stride 8)", m ~row_stride:8 ~m:3 ~k:3 ~n:16 ());
  ]

let%expect_test "row blocking: two rows per iteration" =
  blocking_corpus ~target:Loop_target.neon128 blocking_programs;
  [%expect
    {|
    5 rows, 17 outputs, k 3            blocked 1; C vs oracle equal; vs unblocked equal
    4 rows, 16 outputs, k 4            blocked 1; C vs oracle equal; vs unblocked equal
    one row                            blocked 0; C vs oracle equal; vs unblocked equal
    rows overlap (stride 8)            blocked 0; C vs oracle equal; vs unblocked equal |}]

let%expect_test "row blocking: four rows per iteration, rows left over" =
  blocking_corpus
    ~target:(Loop_target.with_row_block 4 Loop_target.neon128)
    blocking_programs;
  [%expect
    {|
    5 rows, 17 outputs, k 3            blocked 1; C vs oracle equal; vs unblocked equal
    4 rows, 16 outputs, k 4            blocked 1; C vs oracle equal; vs unblocked equal
    one row                            blocked 0; C vs oracle equal; vs unblocked equal
    rows overlap (stride 8)            blocked 0; C vs oracle equal; vs unblocked equal |}]
