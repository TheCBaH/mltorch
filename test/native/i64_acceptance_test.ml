(* End-to-end acceptance fixtures for real multi-node I64 forcing shapes named
   by the implementation plan's Gate 5 item 6 (P5.7): every prior P5.2/P5.3
   session tested one [Compute_i64] arm at a time, fed by a raw [input] tensor
   materialized directly in the test -- never by another node's OWN computed
   I64 output threaded through a real [Graph_builder] chain. That leaves the
   INTERMEDIATE edge's declared [Tensor_sig] unexercised for these specific
   two-/three-hop chains: Reshape/Permute's builder-side fmt-threading defect
   (see reshape_i64_test.ml's second test) was found exactly this way, by a
   second node reading the first one's declared signature rather than a fresh
   materialized tensor. These fixtures build the actual corpus-shaped node
   sequences (mvitv2_tiny's and edgenext_xx_small's real `model.json`s, traced
   in the implementation tracker's P5.2/P5.7 notes) through [Graph_builder]
   and run them through one [Eval_direct.run] call, so a wiring mistake
   between two independently-correct [Compute_i64] arms would show up here
   even if each arm's own isolated test still passes. *)

open Graph_ir
open Graph_direct_fixtures

(* mvitv2_tiny's relative-position-bias pattern: every one of its 40
   `arange.default` (dtype LONG) nodes feeds exactly one `unsqueeze.default`
   (-> [Reshape]), whose only consumer is a `mul.Tensor` with a bare
   `other=1.0` (-> [Mul_scalar]) -- see the tracker's P5.2 continuation note.
   [Reshape] here is fed by [Arange]'s own OWN computed edge, not a
   materialized input, and [Mul_scalar] is fed by [Reshape]'s own edge, not
   [Arange]'s -- the two hops this file adds over the existing per-op tests. *)
let%expect_test
    "mvitv2 acceptance pattern: Arange -> Reshape(unsqueeze) -> Mul_scalar \
     end-to-end" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"mvitv2_pattern" ~outputs:(fun r -> [ r ])
          @@
          let* a =
            arange
              {
                Factory.Arange.start = 0.;
                stop = 4.;
                step = 1.;
                fmt = Payload.(Fmt I64);
                exact = None;
              }
          in
          let* r = reshape { Reshape.Reshape.shape = s 1 1 1 1 4 1 } a in
          mul_scalar ~name:"out" 2.5 r)
    in
    let* env = lift_eval (Eval_direct.run g ~inputs:[]) in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect {| out = tensor f32 [W=4 C=1] {0, 2.5, 5, 7.5} |}]

(* mvitv2_tiny's other named pattern: 20 `_to_copy.default(dtype=LONG)` nodes
   each read a real `add.Tensor(self, other=as_float 52.0)` result (an F32
   [Add_scalar]) as part of its relative-position-bias index computation --
   see the tracker's P5.3 continuation census. [To_copy(Long)] here is fed by
   [Add_scalar]'s own computed F32 edge, not a materialized input, exercising
   the checked [float_to_i64] truncation contract on a value this specific
   producer actually creates (including a fractional, negative result) rather
   than a hand-picked operand. *)
let%expect_test
    "mvitv2 acceptance pattern: Add_scalar -> To_copy(Long) end-to-end" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"mvitv2_bias_pattern" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s1c 4) ~name:"x" () in
          let* a = add_scalar 52. x in
          to_copy ~name:"out" Pointwise.To_copy.Long a)
    in
    let x =
      Tensor.materialize (s1c 4) (fun c ->
          match Dim.to_int (Vec6.get c Axis.C) with
          | 0 -> -53.5
          | 1 -> -0.5
          | 2 -> 0.3
          | _ -> 2.7)
    in
    let* env =
      lift_eval (Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x ]))
    in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  (* self + 52.0 = {-1.5, 51.5, 52.3, 54.7}; truncated toward zero, per the
     plan's own "explicit truncation toward zero" policy, not rounded. *)
  [%expect {| out = tensor i64 [C=4] {-1, 51, 52, 54} |}]

(* EdgeNeXt's own named pattern (Gate 5 item 6): an I64 `arange.default`
   feeding a `_to_copy.default(dtype=FLOAT)`. [Fold_arange_cast]
   (`fold_arange_cast_test.ml`) fuses this exact two-node shape into one F32
   [Arange] whenever start/step are integer-valued -- the common real case --
   so [Eval_direct] only ever sees the UNFUSED chain when the pass declines
   to run at all (as here: [Eval_direct.run] takes a raw, unpassed graph) or
   when a second live consumer blocks the fold. Both `to_copy_i64_test.ml`
   (this exact op) and this file's own [Arange] fixture above test each half
   separately, fed by a materialized input / no consumer; this is the first
   fixture chaining [Arange]'s own computed I64 edge directly into
   [To_copy(Float)], the two-hop wiring the fold pass's OWN fixtures never
   exercise (they inspect the graph transform, not [Eval_direct]'s runtime
   dispatch on the pre-fold graph).

   Unlike the two fixtures above, disabling [To_copy(Float)]'s [Compute_i64]
   dispatch arm does NOT turn this fixture red (checked, not assumed): its
   own header already documents why -- [Payload.get_float]'s I64 case is
   [Int64.to_float], bit-identical to the explicit cast for every value small
   enough for F32 to represent exactly, which these values are. This fixture
   instead guards the ARANGE->TO_COPY wiring itself (the right edge feeds the
   right op with the right values through a real two-node graph), not the
   dispatch-arm choice; a value-level defect in either op would still show up
   as a wrong printed result. *)
let%expect_test
    "edgenext acceptance pattern: Arange -> To_copy(Float) end-to-end" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"edgenext_pattern" ~outputs:(fun r -> [ r ])
          @@
          let* a =
            arange
              {
                Factory.Arange.start = 0.;
                stop = 5.;
                step = 1.;
                fmt = Payload.(Fmt I64);
                exact = None;
              }
          in
          to_copy ~name:"out" Pointwise.To_copy.Float a)
    in
    let* env = lift_eval (Eval_direct.run g ~inputs:[]) in
    tensor_of_name g env "out"
  in
  Format.printf "%a@." (pp_result (pp_named_tensor "out")) result;
  [%expect {| out = tensor f32 [C=5] {0, 1, 2, 3, 4} |}]
