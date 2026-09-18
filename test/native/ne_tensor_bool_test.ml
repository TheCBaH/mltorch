(* [Eval_direct]'s [Ne_tensor] dispatch arm (P6.4), end to end through
   [Graph_builder.ne_tensor]: [pointwise_test.ml]'s own "Direct: ne_tensor"
   exercises [Compute]'s [SEMANTICS]-generic formula directly, and
   [eval_symbolic_ne_tensor_test.ml] proves the Symbolic/Kernel route
   rejects this op's Bool-declared output rather than silently
   materializing it as float; this fixture is the third leg, proving
   [Eval_direct]'s own early-intercept arm actually lands genuine
   [Payload.Bool] storage from a real built graph, the same shape
   [eq_tensor_bool_test.ml] proves for [Eq_tensor]. *)

open Graph_ir
open Graph_direct_fixtures

let%expect_test "Direct graph: ne_tensor writes genuine Bool storage" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"ne_tensor_direct" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 5) ~name:"a" () in
          let* b = input ~shape:(s1c 5) ~name:"b" () in
          ne_tensor ~name:"out" a b)
    in
    let a =
      Tensor.materialize (s1c 5) (fun c ->
          [| 1.; 2.; 3.; Float.nan; Float.neg_infinity |].(Dim.to_int
                                                             (Vec6.get c Axis.C)))
    in
    let b =
      Tensor.materialize (s1c 5) (fun c ->
          [| 1.; 5.; 3.; Float.nan; Float.neg_infinity |].(Dim.to_int
                                                             (Vec6.get c Axis.C)))
    in
    let* env =
      lift_eval
        (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ a; b ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result Tensor.pp) result;
  (* Matches [pointwise_test.ml]'s own "Direct: ne_tensor" values exactly --
     the pointwise negation of [eq_tensor_bool_test.ml]'s own
     {1, 0, 1, 0, 1}. *)
  [%expect {| tensor bool [C=5] {0, 1, 0, 1, 0} |}]
