(* [Eval_direct]'s explicit int64-input promotion for [Pointwise.Mul_scalar]
   (P5.3 follow-up to Reshape/Permute's own exact-I64 dispatch): the default
   [Pointwise.Mul_scalar.Compute(Direct).pixel] arm reads through
   [SEMANTICS.load], which round-trips every format through
   [Payload.get_float] -- numerically exact for this op specifically
   ([Payload.get_float]'s I64 case is [Int64.to_float], bit-identical to an
   explicit cast), but only incidentally so. [eval_direct.ml]'s new
   [Mul_scalar] arm branches on the operand's declared signature format and
   routes an I64 source through [Pointwise.Mul_scalar.Compute_i64], which
   reads via [Tensor.read_i64_at6] and promotes with an explicit
   [i64_to_float], per the implementation plan's "integer-to-float is an
   explicit expression cast" invariant.

   Unlike Reshape/Permute this is NOT a past-2^53 exactness test: this op's
   OUTPUT format is F32 by design (ATen's own int-tensor-times-float-scalar
   promotion), and F32's 24-bit mantissa is already far less precise than a
   double, so pushing the operand past 2^53 would only show F32 rounding --
   unrelated to whether the read used an explicit cast or [SEMANTICS.load]'s
   incidental one, which agree bit-for-bit at every magnitude. This test
   instead exercises the new [Compute_i64] dispatch arm directly, with
   values small enough that F32 represents the product exactly, so a mistake
   in the new arm (e.g. reading the wrong operand, or via [i64_load] but
   forgetting the cast) would show up as a wrong value, not merely an
   already-expected rounding artifact. See the implementation tracker's
   P5.2/P5.3 notes. *)

open Graph_ir
open Graph_direct_fixtures

let%expect_test
    "Direct graph: Mul_scalar reads an I64 operand via an explicit cast" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"mul_scalar_i64" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s1c 3) ~name:"x" ~fmt:Payload.(Fmt I64) () in
          mul_scalar ~name:"out" 2.5 x)
    in
    let x =
      Tensor.materialize_i64 (s1c 3) (fun c ->
          Int64.of_int (1 + Dim.to_int (Vec6.get c Axis.C)))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect {|
    out = tensor f32 [C=3] {2.5, 5, 7.5}
    |}]
