(* [To_copy]'s [Bool] target on an I64 operand: an exact comparison with 0L --
   previously rejected outright (`Unsupported_to_copy_bool_source`), since
   [To_copy(Bool)] first only wired the F32 source EdgeNeXt's own mask
   pattern needs. This is an exact int64 zero test via [Direct.i64_load],
   not a route through [Payload.get_float]'s incidental float nonzero test
   -- covers 0L, positive/negative nonzero, and a value past 2^53 to prove
   no float intermediary is involved. *)

open Graph_ir
open Graph_direct_fixtures

let%expect_test
    "Direct graph: To_copy(Bool) on an I64 operand is an exact zero test" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"to_copy_bool_i64" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s1c 5) ~name:"x" ~fmt:Payload.(Fmt I64) () in
          to_copy ~name:"out" Pointwise.To_copy.Bool x)
    in
    let x =
      Tensor.materialize_i64 (s1c 5) (fun c ->
          [| 0L; 1L; -1L; 9_007_199_254_740_993L; Int64.min_int |].(chan c))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect {| out = tensor bool [C=5] {0, 1, 1, 1, 1} |}]
