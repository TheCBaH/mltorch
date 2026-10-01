open Loop_ir

(* The vector corpus through vectorized C compiled with the host compiler,
   bitwise against the interpreter on the scalar program. Each program must also
   have produced vector code (the generic-vector helpers appear in its source),
   so a pass is not the scalar path in disguise. *)

let zeroed (p : Loop_program.t) id =
  List.find_map
    (fun (b : Loop_buffer.t) ->
      if
        b.Loop_buffer.role = Loop_buffer.Output
        && Tensor_id.equal b.Loop_buffer.id id
      then Some (Loop_interp.allocate b)
      else None)
    p.Loop_program.buffers

let vectorized ~target p =
  match Err.payload (Loop_c_exec.source ~vector:target p) with
  | Ok (text, _) ->
      let rec contains s sub i =
        i + String.length sub <= String.length s
        && (String.sub s i (String.length sub) = sub || contains s sub (i + 1))
      in
      contains text "vf_" 0
  | Error _ -> false

let corpus ~target =
  List.iter
    (fun (name, p) ->
      let bind = Loop_vector_programs.bind in
      let reference = Loop_interp.run ~outputs:(zeroed p) p ~bind in
      let vec = Loop_c_exec.exec ~vector:target ~outputs:(zeroed p) p ~bind in
      let verdict =
        match (Err.payload reference, Err.payload vec) with
        | Ok a, Ok b ->
            if Tensor_id.Map.equal Loop_check.tensors_equal a b then "equal"
            else "DIFFERS"
        | Error _, Error _ -> "both fail"
        | _ -> "ONE FAILED"
      in
      Fmt.pr "%-30s %-8s %s@." name verdict
        (if vectorized ~target p then "vector code" else "scalar"))
    Loop_vector_programs.all

let%expect_test "every vector program: vectorized C equals the interpreter" =
  corpus ~target:Loop_target.neon128;
  [%expect
    {|
    double rounding                equal    vector code
    arith chain                    equal    vector code
    offset view                    equal    vector code
    broadcast and invariant        equal    vector code
    strided                        equal    scalar
    maximum                        equal    vector code
    select and compare             equal    vector code
    pool_better                    equal    vector code
    not and or                     equal    vector code
    index value                    equal    vector code
    temporaries                    equal    vector code
    sqrt and trunc                 equal    vector code
    transcendentals                equal    vector code
    int32 source                   equal    vector code
    bool store                     equal    scalar
    nested loops, vector inner     equal    vector code
    matmul: reduction per output   equal    vector code
    reduction with a triangular inner bound equal    vector code
    uniform index temporary        equal    vector code
    extents around the width: 3    equal    scalar
    extents around the width: 4    equal    vector code
    extents around the width: 5    equal    vector code |}]

let%expect_test
    "the same with every cost zero: strided, expanded and bool paths" =
  corpus ~target:(Loop_target.forced Loop_target.neon128);
  [%expect
    {|
    double rounding                equal    vector code
    arith chain                    equal    vector code
    offset view                    equal    vector code
    broadcast and invariant        equal    vector code
    strided                        equal    vector code
    maximum                        equal    vector code
    select and compare             equal    vector code
    pool_better                    equal    vector code
    not and or                     equal    vector code
    index value                    equal    vector code
    temporaries                    equal    vector code
    sqrt and trunc                 equal    vector code
    transcendentals                equal    vector code
    int32 source                   equal    vector code
    bool store                     equal    vector code
    nested loops, vector inner     equal    vector code
    matmul: reduction per output   equal    vector code
    reduction with a triangular inner bound equal    vector code
    uniform index temporary        equal    vector code
    extents around the width: 3    equal    scalar
    extents around the width: 4    equal    vector code
    extents around the width: 5    equal    vector code |}]
