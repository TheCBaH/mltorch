module M = Machine_model.Mir_model
module T = Model_test

(* bfloat16 results under [Fp32_ordered] on neon128: the narrowing kernel's
   store is a vector loop (the bfloat16 rounding in int32 lanes and a 16-bit
   store), with no binary64 anywhere, bitwise the binary32 oracle on the
   generic route and both scanned routes. *)
let%expect_test "bfloat16 results: vector loops, the binary32 oracle" =
  List.iter
    (fun (name, g) ->
      let b = T.bundle (g ()) in
      let census =
        match M.prepare ~pipeline:T.fp32_vector b with
        | Error _ -> "refused"
        | Ok m ->
            let cs =
              List.map Machine_ir.Mir_f64_census.of_program (M.generic m)
            in
            String.concat ", "
              (List.map
                 (fun r ->
                   Fmt.str "%s %d"
                     (Machine_ir.Mir_f64_census.Role.name r)
                     (List.fold_left
                        (fun n c -> n + Machine_ir.Mir_f64_census.count c r)
                        0 cs))
                 Machine_ir.Mir_f64_census.Role.all)
      in
      Fmt.pr "%s: %d vector loops; binary64 %s@." name (T.vector_loops b) census;
      List.iter
        (fun (rname, route) ->
          T.check ?route ~pipeline:T.fp32_vector ~oracle:T.Fp32
            ~constant:T.positive ("  " ^ rname) g)
        [
          ("generic", None);
          ("aarch64 scanned", Some (M.Route.Aarch64 M.Stage.Scanned));
          ("x86_64 scanned", Some (M.Route.X86_64 M.Stage.Scanned));
        ])
    Transformer_fixtures.bf16_out;
  [%expect
    {|
    residual add bf16->bf16 2x160: 2 vector loops; binary64 arithmetic 0, conversion 0, helper_call 0, memory_move 0
      generic (2 invocations): bitwise, bitwise
      aarch64 scanned (2 invocations): bitwise, bitwise
      x86_64 scanned (2 invocations): bitwise, bitwise
    residual add bf16->bf16 1x67: 2 vector loops; binary64 arithmetic 0, conversion 0, helper_call 0, memory_move 0
      generic (2 invocations): bitwise, bitwise
      aarch64 scanned (2 invocations): bitwise, bitwise
      x86_64 scanned (2 invocations): bitwise, bitwise
    residual add bf16->bf16 1x33: 2 vector loops; binary64 arithmetic 0, conversion 0, helper_call 0, memory_move 0
      generic (2 invocations): bitwise, bitwise
      aarch64 scanned (2 invocations): bitwise, bitwise
      x86_64 scanned (2 invocations): bitwise, bitwise
    rms norm bf16->bf16 2x160: 2 vector loops; binary64 arithmetic 0, conversion 0, helper_call 0, memory_move 0
      generic (2 invocations): bitwise, bitwise
      aarch64 scanned (2 invocations): bitwise, bitwise
      x86_64 scanned (2 invocations): bitwise, bitwise
    rms norm bf16->bf16 1x67: 2 vector loops; binary64 arithmetic 0, conversion 0, helper_call 0, memory_move 0
      generic (2 invocations): bitwise, bitwise
      aarch64 scanned (2 invocations): bitwise, bitwise
      x86_64 scanned (2 invocations): bitwise, bitwise
    rms norm bf16->bf16 1x33: 2 vector loops; binary64 arithmetic 0, conversion 0, helper_call 0, memory_move 0
      generic (2 invocations): bitwise, bitwise
      aarch64 scanned (2 invocations): bitwise, bitwise
      x86_64 scanned (2 invocations): bitwise, bitwise
    tanh gelu mul bf16->bf16 2x160: 3 vector loops; binary64 arithmetic 0, conversion 0, helper_call 0, memory_move 0
      generic (3 invocations): bitwise, bitwise
      aarch64 scanned (3 invocations): bitwise, bitwise
      x86_64 scanned (3 invocations): bitwise, bitwise
    tanh gelu mul bf16->bf16 1x67: 3 vector loops; binary64 arithmetic 0, conversion 0, helper_call 0, memory_move 0
      generic (3 invocations): bitwise, bitwise
      aarch64 scanned (3 invocations): bitwise, bitwise
      x86_64 scanned (3 invocations): bitwise, bitwise
    tanh gelu mul bf16->bf16 1x33: 3 vector loops; binary64 arithmetic 0, conversion 0, helper_call 0, memory_move 0
      generic (3 invocations): bitwise, bitwise
      aarch64 scanned (3 invocations): bitwise, bitwise
      x86_64 scanned (3 invocations): bitwise, bitwise |}]
