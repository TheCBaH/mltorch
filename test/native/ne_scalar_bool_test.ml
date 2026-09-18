(* [Eval_direct]'s [Ne_scalar] dispatch arm (P6.4), end to end through
   [Graph_builder.ne_scalar]: [pointwise_test.ml]'s own "Direct: ne_scalar"
   exercises [Compute]'s [SEMANTICS]-generic formula directly, and
   [eval_symbolic_ne_scalar_test.ml] proves the Symbolic/Kernel route
   rejects this op's Bool-declared output rather than silently
   materializing it as float; this fixture is the third leg, proving
   [Eval_direct]'s own early-intercept arm actually lands genuine
   [Payload.Bool] storage from a real built graph, the same shape
   [eq_scalar_bool_test.ml] proves for [Eq_scalar]. *)

open Graph_ir
open Graph_direct_fixtures

let%expect_test "Direct graph: ne_scalar writes genuine Bool storage" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"ne_scalar_direct" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s1c 5) ~name:"x" () in
          ne_scalar ~name:"out" 2. x)
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
  (* Every index except 1 (value 2.) is unequal to the scalar; NaN is
     unequal to everything, matching [pointwise_test.ml]'s own
     "Direct: ne_scalar" values exactly -- the pointwise negation of
     [eq_scalar_bool_test.ml]'s own {0, 1, 0, 0, 0}. *)
  [%expect {| tensor bool [C=5] {1, 0, 1, 1, 1} |}]
