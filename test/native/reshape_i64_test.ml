(* [Eval_direct]'s dtype-preserving [Reshape] dispatch (this session's P5.2
   slice): the default [Reshape.Reshape.Compute(Direct).pixel] arm reads
   through [SEMANTICS.load], which round-trips every format through
   [Payload.get_float] and is therefore silently lossy for an exact I64
   value above float's 2^53 mantissa. [eval_direct.ml]'s new [Reshape] arm
   branches on the operand's declared signature format and routes an I64
   source through [Reshape.Reshape.Compute_i64], which reads via
   [Tensor.read_i64_at6] instead -- never touching float. See the
   implementation tracker's P5.2 note. *)

open Graph_ir
open Graph_direct_fixtures

let%expect_test
    "Direct graph: I64 reshape [H=2 W=3 C=1] -> [C=6] stays exact past 2^53" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"reshape_i64" ~outputs:(fun r -> [ r ])
          @@
          let* x =
            input ~shape:(s 1 1 1 2 3 1) ~name:"x" ~fmt:Payload.(Fmt I64) ()
          in
          reshape ~name:"out" { Reshape.Reshape.shape = s 1 1 1 1 1 6 } x)
    in
    (* Row-major over [H,W]: value = 9_007_199_254_740_993L + (h*3+w), so
       every one of the 6 cells sits strictly above 2^53 and is distinct --
       a float round trip would collapse the first two (9_007_199_254_740_993
       and ...994 both round to the same float). *)
    let x =
      Tensor.materialize_i64 (s 1 1 1 2 3 1) (fun c ->
          Int64.add 9_007_199_254_740_993L
            (Int64.of_int
               ((Dim.to_int (Vec6.get c Axis.H) * 3)
               + Dim.to_int (Vec6.get c Axis.W))))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect
    {|
    out = tensor i64 [C=6] {9007199254740993, 9007199254740994, 9007199254740995, 9007199254740996, 9007199254740997, 9007199254740998}
    |}]

let%expect_test
    "Direct graph: I64 reshape preserves format through a non-flattening target"
    =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"reshape_i64_2d" ~outputs:(fun r -> [ r ])
          @@
          let* x =
            input ~shape:(s 1 1 1 2 3 1) ~name:"x" ~fmt:Payload.(Fmt I64) ()
          in
          reshape ~name:"out" { Reshape.Reshape.shape = s 1 1 1 3 2 1 } x)
    in
    let x =
      Tensor.materialize_i64 (s 1 1 1 2 3 1) (fun c ->
          Int64.add 9_007_199_254_740_993L
            (Int64.of_int
               ((Dim.to_int (Vec6.get c Axis.H) * 3)
               + Dim.to_int (Vec6.get c Axis.W))))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect
    {|
    out = tensor i64 [H=3 W=2 C=1] {9007199254740993, 9007199254740994, 9007199254740995, 9007199254740996, 9007199254740997, 9007199254740998}
    |}]
