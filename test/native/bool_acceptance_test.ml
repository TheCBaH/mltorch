(* Baseline for EdgeNeXt's `PositionalEncodingFourier` mask pattern (Gate 6/7):
   a Float mask goes through `_to_copy.default(dtype=BOOL)` then
   `bitwise_not.default` -- see `pointwise_unary.ml`'s own [Bitwise_not]
   comment. As of this fixture, `Graph_builder.to_copy`'s [Bool] target still
   keeps [op1]'s F32 default (Gate 6 has not opened a distinct output format
   for it yet — see that function's own comment), so both nodes compute and
   store as an ordinary F32 0.0/1.0 encoding, not through [Payload.Bool]
   storage. This fixture pins that CURRENT two-hop behavior with independent
   expected values, so a future session that gives [To_copy]'s [Bool] target
   real [Payload.Bool] storage (P6.3/P6.4) has a documented regression
   baseline: after that change, [Payload.get_float]'s own Bool policy
   (nonzero reads true) must still make this exact fixture agree bit-for-bit,
   since [Direct]'s generic read path decodes through [Payload.get_float]
   regardless of storage format. *)

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
          let* b = to_copy Pointwise.To_copy.Bool x in
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
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  (* x = {0, 3, -2, 0} -> Bool {false, true, true, false} -> not {true, false,
     false, true}, encoded as F32 0.0/1.0 today. *)
  [%expect {| out = tensor f32 [C=4] {1, 0, 0, 1} |}]
