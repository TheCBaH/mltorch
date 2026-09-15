(* [Eval_direct]'s explicit int64-input promotion for [Pointwise.To_copy]'s
   [Float] target (P5.3 follow-up to [Mul_scalar]'s own explicit-cast fix):
   the default [Pointwise.To_copy.Compute(S).pixel] arm reads through
   [SEMANTICS.load], which round-trips every format through
   [Payload.get_float] -- numerically exact for this specific promotion
   ([Payload.get_float]'s I64 case is [Int64.to_float], bit-identical to an
   explicit [i64_to_float] cast), but only incidentally so. [eval_direct.ml]'s
   new [To_copy] arm (restricted to the [Float] target -- [Long]/[Bool] are
   untouched) branches on the operand's declared signature format and routes
   an I64 source through [Pointwise.To_copy.Compute_i64], which reads via
   [Tensor.read_i64_at6] and promotes with an explicit [i64_to_float], per the
   plan's "integer-to-float is an explicit expression cast" invariant. This is
   the EdgeNeXt/mvitv2 "I64 Arange -> Float cast" acceptance pattern's own
   promoted-consumer step (Gate 5 item 6), the same architecture-only shape
   as [Mul_scalar.Compute_i64]: no value this op ever returns changes, only
   how the cast is expressed. *)

open Graph_ir
open Graph_direct_fixtures

(* Small, exactly-F32-representable values, matching [mul_scalar_i64_test.ml]'s
   own rationale: this op's OUTPUT format is F32 by design, so a past-2^53
   operand would only exercise F32's OWN rounding boundary (harmless, and
   identical whether the read used an explicit cast or [SEMANTICS.load]'s
   incidental one -- see this file's own header comment) rather than proving
   the new [Compute_i64] dispatch arm reads the right operand and casts it at
   all. Small values instead let a real mistake in the new arm (e.g. reading
   the wrong operand, or an [i64_load] without the [i64_to_float] cast) show
   up as a wrong value. *)
let%expect_test
    "Direct graph: To_copy(Float) reads an I64 operand via an explicit cast" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"to_copy_i64" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s1c 3) ~name:"x" ~fmt:Payload.(Fmt I64) () in
          to_copy ~name:"out" Pointwise.To_copy.Float x)
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
    out = tensor f32 [C=3] {1, 2, 3}
    |}]
