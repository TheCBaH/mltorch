(* [To_copy]'s [Long]/[Float] targets on a Bool operand: design section 3's
   "Bool to I64 / Float: Exact 0/1 in the destination carrier". [Long]
   previously rejected a Bool source outright
   (`Unsupported_to_copy_long_source`) -- this is its own new
   [Direct.bool_load]-based arm, the reverse direction of
   [to_copy_bool_i64_test.ml]'s I64-to-Bool cast. [Float] needed no new
   arm: [Graph_builder.to_copy]'s [Float] target always declares an F32
   output (unlike [Bool]/[Long], which override [~fmt]), so the generic
   default [Compute(Direct).pixel] path (reading via [Payload.get_float],
   whose Bool policy is already "nonzero reads true", per P6.1) already
   produces the exact contract -- this file's own second test is the
   evidence for that claim, not a code change. *)

open Graph_ir
open Graph_direct_fixtures

let run target =
  let open Err.Syntax in
  let* g =
    lift_build
      Graph_builder.(
        build ~name:"to_copy_bool_source" ~outputs:(fun r -> [ r ])
        @@
        let* x = input ~shape:(s1c 3) ~name:"x" ~fmt:Payload.(Fmt Bool) () in
        to_copy ~name:"out" target x)
  in
  let x = Tensor.materialize_bool (s1c 3) (fun c -> chan c <> 0) in
  let* env =
    lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x ]))
  in
  tensor_of_name g env "out"

let%expect_test "Direct graph: To_copy(Long) on a Bool operand is exact 0/1" =
  Format.printf "%a@."
    (pp_result (pp_named_tensor "out"))
    (run Pointwise.To_copy.Long);
  [%expect {| out = tensor i64 [C=3] {0, 1, 1} |}]

let%expect_test "Direct graph: To_copy(Float) on a Bool operand is exact 0/1" =
  Format.printf "%a@."
    (pp_result (pp_named_tensor "out"))
    (run Pointwise.To_copy.Float);
  [%expect {| out = tensor f32 [C=3] {0, 1, 1} |}]
