open Loop_ir

(* The vector corpus through the SIMD-lowered Wasm, run under node, bitwise
   against the interpreter on the scalar program. The corpus is chosen so a lane
   that skipped a rounding, took the wrong lane of a load, or lost a NaN's rule
   shows. Each program is also required to have been vectorized and to need the
   [simd128] feature, so a pass is not the scalar path in disguise. *)

let zeroed (p : Loop_program.t) id =
  List.find_map
    (fun (b : Loop_buffer.t) ->
      if
        b.Loop_buffer.role = Loop_buffer.Output
        && Tensor_id.equal b.Loop_buffer.id id
      then Some (Loop_interp.allocate b)
      else None)
    p.Loop_program.buffers

let features ~target p =
  match Err.payload (Loop_wasm.lower ~vector:target p) with
  | Ok l ->
      List.map Wasm_features.name (Wasm_features.of_module l.Loop_wasm.module_)
  | Error e -> Fmt.failwith "%a" Loop_wasm.pp_error e

let corpus ~target =
  List.iter
    (fun (name, p) ->
      let bind = Loop_vector_programs.bind in
      let reference = Loop_interp.run ~outputs:(zeroed p) p ~bind in
      let simd =
        Loop_wasm_exec.exec ~vector:target ~outputs:(zeroed p) p ~bind
      in
      let verdict =
        match (Err.payload reference, Err.payload simd) with
        | Ok a, Ok b ->
            if Tensor_id.Map.equal Loop_check.tensors_equal a b then "equal"
            else "DIFFERS"
        | Error _, Error _ -> "both fail"
        | _ -> "ONE FAILED"
      in
      Fmt.pr "%-30s %-8s %s@." name verdict
        (String.concat "," (features ~target p)))
    Loop_vector_programs.all

let%expect_test "every vector program: SIMD Wasm equals the interpreter" =
  corpus ~target:Loop_target.wasm128;
  [%expect
    {|
    double rounding                equal    simd128
    arith chain                    equal    simd128
    offset view                    equal    simd128
    broadcast and invariant        equal    simd128
    strided                        equal
    maximum                        equal    simd128
    select and compare             equal    simd128
    pool_better                    equal    simd128
    not and or                     equal    simd128
    index value                    equal    simd128
    temporaries                    equal    simd128
    sqrt and trunc                 equal    simd128
    transcendentals                equal    simd128
    int32 source                   equal    simd128
    bool store                     equal
    nested loops, vector inner     equal    simd128
    matmul: reduction per output   equal    simd128
    reduction with a triangular inner bound equal    simd128
    uniform index temporary        equal    simd128
    extents around the width: 3    equal
    extents around the width: 4    equal    simd128
    extents around the width: 5    equal    simd128 |}]

let%expect_test
    "the same with every cost zero: strided, expanded and bool paths" =
  corpus ~target:(Loop_target.forced Loop_target.wasm128);
  [%expect
    {|
    double rounding                equal    simd128
    arith chain                    equal    simd128
    offset view                    equal    simd128
    broadcast and invariant        equal    simd128
    strided                        equal    simd128
    maximum                        equal    simd128
    select and compare             equal    simd128
    pool_better                    equal    simd128
    not and or                     equal    simd128
    index value                    equal    simd128
    temporaries                    equal    simd128
    sqrt and trunc                 equal    simd128
    transcendentals                equal    simd128
    int32 source                   equal    simd128
    bool store                     equal    simd128
    nested loops, vector inner     equal    simd128
    matmul: reduction per output   equal    simd128
    reduction with a triangular inner bound equal    simd128
    uniform index temporary        equal    simd128
    extents around the width: 3    equal
    extents around the width: 4    equal    simd128
    extents around the width: 5    equal    simd128 |}]

(* Execution marks keep their multiplicity: a vector iteration bumps a mark once
   per lane, so the counting build's counts equal the scalar program's. *)
let%expect_test "marks under vectorization count as the scalar program does" =
  List.iter
    (fun (name, p) ->
      let bind = Loop_vector_programs.bind in
      let counts vector =
        match
          Err.payload
            (Loop_wasm_exec.exec_counted ?vector ~outputs:(zeroed p) p ~bind)
        with
        | Ok (_, c) -> c
        | Error e -> Fmt.failwith "%a" Loop_wasm_exec.pp_error e
      in
      let scalar = counts None in
      let vec = counts (Some (Loop_target.forced Loop_target.wasm128)) in
      Fmt.pr "%-30s %s; counts %s@." name
        (String.concat " "
           (List.filter_map
              (fun (m, n) ->
                if n = 0 then None
                else Some (Printf.sprintf "%s=%d" (Loop_mark.name m) n))
              scalar))
        (if scalar = vec then "equal" else "DIFFER"))
    (List.filter
       (fun (name, _) ->
         String.length name >= 6 && String.sub name 0 6 = "matmul")
       Loop_vector_programs.all);
  [%expect
    {| matmul: reduction per output   emitter=40 reduction=240; counts equal |}]
