(* Acceptance fixture for EdgeNeXt's `PositionalEncodingFourier` mask pattern
   (Gate 6/7): a Float mask goes through `_to_copy.default(dtype=BOOL)` then
   `bitwise_not.default` -- see `pointwise_unary.ml`'s own [Bitwise_not]
   comment. `Graph_builder.to_copy`'s [Bool] target now declares and writes
   genuine [Payload.Bool] storage for its own output edge (P6.3); this test
   was originally written as a pre-change baseline pinning the prior F32
   0.0/1.0 encoding, and its own printed values did NOT change when that
   arm landed -- confirmed, not assumed, by rerunning this exact fixture
   before and after: [Payload.get_float]'s Bool policy (nonzero reads true)
   agrees with the float encoding it replaced, and [Bitwise_not]'s own
   generic [Direct] dispatch reads any operand format through [Payload.
   get_float] already, so nothing downstream needed to change either.
   [Bitwise_not]'s own OUTPUT is still F32 ([Graph_builder.bitwise_not]
   does not thread operand format), which is why "out" below still prints
   "f32" -- only the INTERMEDIATE `to_copy(Bool)` edge is genuinely
   Bool-formatted now. *)

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
     not {true, false, false, true}, [Bitwise_not]'s own output staying F32. *)
  [%expect
    {|
    mask = tensor bool [C=4] {0, 1, 1, 0}
    out = tensor f32 [C=4] {1, 0, 0, 1} |}]
