(* [Eval_direct]'s [Eq_scalar] dispatch arm (P6.4), end to end through
   [Graph_builder.eq_scalar]: [pointwise_test.ml]'s own "Direct: eq_scalar"
   exercises [Compute]'s [SEMANTICS]-generic formula directly, and
   [eval_symbolic_eq_scalar_test.ml] proves the Symbolic/Kernel route
   rejects this op's Bool-declared output rather than silently
   materializing it as float; this fixture is the third leg, proving
   [Eval_direct]'s own early-intercept arm actually lands genuine
   [Payload.Bool] storage from a real built graph, the same shape
   [gt_scalar_bool_test.ml] (test/native4d) proves for Native4D. *)

open Graph_ir
open Graph_direct_fixtures

let%expect_test "Direct graph: eq_scalar writes genuine Bool storage" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"eq_scalar_direct" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s1c 5) ~name:"x" () in
          eq_scalar ~name:"out" 2. x)
    in
    let x =
      Tensor.materialize (s1c 5) (fun c ->
          [| 1.; 2.; 3.; Float.nan; Float.neg_infinity |].(Dim.to_int
                                                             (Vec6.get c Axis.C)))
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result Tensor.pp) result;
  (* Only index 1 (value 2.) equals the scalar; NaN is unequal to
     everything including a finite scalar, matching [pointwise_test.ml]'s
     own "Direct: eq_scalar" values exactly. *)
  [%expect {| tensor bool [C=5] {0, 1, 0, 0, 0} |}]
