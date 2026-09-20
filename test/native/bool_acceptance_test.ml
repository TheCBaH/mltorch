(* Acceptance fixture for EdgeNeXt's `PositionalEncodingFourier` mask pattern: a
   Float mask goes through `_to_copy.default(dtype=BOOL)` then
   `bitwise_not.default` -- see `pointwise_unary.ml`'s own [Bitwise_not]
   comment. Both `Graph_builder.to_copy`'s [Bool] target and `Graph_builder.
   bitwise_not` now declare and write genuine [Payload.Bool] storage for
   their own output edges, matching real ATen's own bool-in/bool-out
   contract for [bitwise_not]. This test was originally written as a
   pre-change baseline pinning the prior F32 0.0/1.0 encoding for both
   nodes, and its own printed VALUES did not change as each arm landed in
   turn -- confirmed, not assumed, by rerunning this exact fixture at each
   step: [Payload.get_float]'s Bool policy (nonzero reads true) agrees with
   the float encoding it replaced, and both ops' generic [Direct] dispatch
   already reads any operand format through [Payload.get_float], so no
   downstream consumer needed a matching change either. *)

open Graph_ir
open Graph_direct_fixtures

let%expect_test
    "edgenext mask acceptance pattern: input -> To_copy(Bool) -> Bitwise_not \
     end-to-end" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"edgenext_mask_pattern" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s1c 4) ~name:"x" () in
          let* b = to_copy ~name:"mask" Pointwise.To_copy.Bool x in
          bitwise_not ~name:"out" b)
    in
    let x =
      Tensor.materialize (s1c 4) (fun c ->
          match Dim.to_int (Vec6.get c Axis.C) with
          | 0 -> 0.0
          | 1 -> 3.0
          | 2 -> -2.0
          | _ -> 0.0)
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x ]))
    in
    let* mask = tensor_of_name g env "mask" in
    let* out = tensor_of_name g env "out" in
    Err.return (mask, out)
  in
  Format.printf "%a@." (pp_result (pp_named_tensor_pair "mask" "out")) result;
  (* x = {0, 3, -2, 0} -> genuine Bool storage {false, true, true, false} ->
     not {true, false, false, true}, [Bitwise_not]'s own output now also
     genuine Bool storage. *)
  [%expect
    {|
    mask = tensor bool [C=4] {0, 1, 1, 0}
    out = tensor bool [C=4] {1, 0, 0, 1} |}]
