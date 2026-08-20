(* Group 5 activations (silu, hardsigmoid, hardswish) through the serialized
   path. No model in this repo's zoo serializes any of the three FUNCTIONAL
   targets (op5-impl F1) -- mobilenet_v3_small exports hardswish
   pre-decomposed, and none of the five downloaded models use silu.default or
   hardsigmoid.default at all -- so these fixtures are the only place these
   targets reach [Native_interp] at all. efficientnet_b0-b5 DO serialize
   silu_.default (dozens of times each), which is why the in-place spelling is
   covered here too, not just the functional one.

   Numeric agreement against real ATen (both spellings) is checked in
   test/native_bridge_test.ml, whose [verify_print] harness runs both
   backends; this file only asserts what [Native_interp.lower] builds and its
   output binding, mirroring binary_test.ml's [dump] pattern. *)

open Programs

let unary_node target ~out =
  jstr
    {|{"target":"%s","inputs":[{"name":"self","arg":%s,"kind":1}],"outputs":[%s],"metadata":{}}|}
    target (as_tensor "x") out

let prog node =
  program ~x_sizes:[ 2; 3 ] ~nodes:[ node ] ~graph_outputs:[ as_tensor "y" ] ()

let dump label json =
  Format.printf "%s@." label;
  match lower json with
  | Error e -> Format.printf "  %a@." Native_interp.pp_error (Err.Error.kind e)
  | Ok l -> Format.printf "%a@." Graph_ir.pp l.Pt2_native_graph.graph

let%expect_test "silu.default lowers to a Silu node" =
  dump "silu.default:"
    (prog (unary_node "torch.ops.aten.silu.default" ~out:(as_tensor "y")));
  [%expect
    {|
    silu.default:
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=3]] = silu x=t0
    outputs: [t1 f32 [W=2 C=3] <-n0] |}]

(* The in-place spelling: same node, same output binding -- efficientnet's
   actual coverage (op5-impl F1). *)
let%expect_test "silu_.default lowers to a Silu node" =
  dump "silu_.default:"
    (prog (unary_node "torch.ops.aten.silu_.default" ~out:(as_tensor "y")));
  [%expect
    {|
    silu_.default:
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=3]] = silu x=t0
    outputs: [t1 f32 [W=2 C=3] <-n0] |}]

let%expect_test "hardsigmoid.default lowers to a Hardsigmoid node" =
  dump "hardsigmoid.default:"
    (prog
       (unary_node "torch.ops.aten.hardsigmoid.default" ~out:(as_tensor "y")));
  [%expect
    {|
    hardsigmoid.default:
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=3]] = hardsigmoid x=t0
    outputs: [t1 f32 [W=2 C=3] <-n0] |}]

let%expect_test "hardsigmoid_.default lowers to a Hardsigmoid node" =
  dump "hardsigmoid_.default:"
    (prog
       (unary_node "torch.ops.aten.hardsigmoid_.default" ~out:(as_tensor "y")));
  [%expect
    {|
    hardsigmoid_.default:
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=3]] = hardsigmoid x=t0
    outputs: [t1 f32 [W=2 C=3] <-n0] |}]

let%expect_test "hardswish.default lowers to a Hardswish node" =
  dump "hardswish.default:"
    (prog (unary_node "torch.ops.aten.hardswish.default" ~out:(as_tensor "y")));
  [%expect
    {|
    hardswish.default:
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=3]] = hardswish x=t0
    outputs: [t1 f32 [W=2 C=3] <-n0] |}]

let%expect_test "hardswish_.default lowers to a Hardswish node" =
  dump "hardswish_.default:"
    (prog (unary_node "torch.ops.aten.hardswish_.default" ~out:(as_tensor "y")));
  [%expect
    {|
    hardswish_.default:
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=3]] = hardswish x=t0
    outputs: [t1 f32 [W=2 C=3] <-n0] |}]
