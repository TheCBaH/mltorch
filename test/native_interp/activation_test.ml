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

let%expect_test "sigmoid.default lowers to a Sigmoid node" =
  dump "sigmoid.default:"
    (prog (unary_node "torch.ops.aten.sigmoid.default" ~out:(as_tensor "y")));
  [%expect
    {|
    sigmoid.default:
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=3]] = sigmoid x=t0
    outputs: [t1 f32 [W=2 C=3] <-n0] |}]

let leaky_relu_node ?negative_slope () =
  let slope =
    match negative_slope with
    | None -> ""
    | Some x ->
        jstr {|,{"name":"negative_slope","arg":{"as_float":%f},"kind":1}|} x
  in
  jstr
    {|{"target":"torch.ops.aten.leaky_relu.default","inputs":[{"name":"self","arg":%s,"kind":1}%s],"outputs":[%s],"metadata":{}}|}
    (as_tensor "x") slope (as_tensor "y")

let%expect_test "leaky_relu.default lowers with its schema default and option" =
  dump "default:" (prog (leaky_relu_node ()));
  dump "explicit:" (prog (leaky_relu_node ~negative_slope:0.2 ()));
  [%expect
    {|
    default:
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=3]] = leaky_relu x=t0 params={negative_slope=0.01}
    outputs: [t1 f32 [W=2 C=3] <-n0]
    explicit:
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=3]] = leaky_relu x=t0 params={negative_slope=0.2}
    outputs: [t1 f32 [W=2 C=3] <-n0] |}]

let zeros_node ?dtype () =
  let dtype =
    match dtype with
    | None -> ""
    | Some d ->
        jstr {|,{"name":"dtype","arg":{"as_scalar_type":%d},"kind":1}|} d
  in
  jstr
    {|{"target":"torch.ops.aten.zeros.default","inputs":[{"name":"size","arg":{"as_sym_ints":[{"as_int":2},{"as_int":3}]},"kind":1}%s,{"name":"device","arg":{"as_device":{"type":"cpu","index":null}},"kind":2},{"name":"pin_memory","arg":{"as_bool":false},"kind":2}],"outputs":[%s],"metadata":{}}|}
    dtype (as_tensor "y")

let%expect_test "zeros.default lowers with default and DOUBLE dtype" =
  dump "default:" (prog (zeros_node ()));
  dump "double:" (prog (zeros_node ~dtype:8 ()));
  [%expect
    {|
    default:
    graph
    inputs: [t0 f32 [W=2 C=3]]
    nodes:
      n0: [t1 f32 [W=2 C=3]] = zeros params={shape=[W=2 C=3]; fmt=f32}
    outputs: [t1 f32 [W=2 C=3] <-n0]
    double:
    graph
    inputs: [t0 f32 [W=2 C=3]]
    nodes:
      n0: [t1 f64 [W=2 C=3]] = zeros params={shape=[W=2 C=3]; fmt=f64}
    outputs: [t1 f64 [W=2 C=3] <-n0] |}]

let arange_node target inputs =
  jstr
    {|{"target":"%s","inputs":[%s,{"name":"dtype","arg":{"as_scalar_type":%d},"kind":1},{"name":"device","arg":{"as_device":{"type":"cpu","index":null}},"kind":2},{"name":"pin_memory","arg":{"as_bool":false},"kind":2}],"outputs":[%s],"metadata":{}}|}
    target inputs
    (if target = "torch.ops.aten.arange.default" then 5 else 7)
    (as_tensor "y")

let%expect_test "arange default Long and start Float lower" =
  dump "default:"
    (prog
       (arange_node "torch.ops.aten.arange.default"
          {|{"name":"end","arg":{"as_int":5},"kind":1}|}));
  dump "start:"
    (prog
       (arange_node "torch.ops.aten.arange.start"
          {|{"name":"start","arg":{"as_float":0.5},"kind":1},{"name":"end","arg":{"as_int":4},"kind":1}|}));
  [%expect
    {|
    default:
    graph
    inputs: [t0 f32 [W=2 C=3]]
    nodes:
      n0: [t1 i64 [C=5]] = arange params={start=0; stop=5; step=1; fmt=i64}
    outputs: [t1 i64 [C=5] <-n0]
    start:
    graph
    inputs: [t0 f32 [W=2 C=3]]
    nodes:
      n0: [t1 f32 [C=4]] = arange params={start=0.5; stop=4; step=1; fmt=f32}
    outputs: [t1 f32 [C=4] <-n0] |}]

let gelu_node ?approximate () =
  let arg name a = jstr {|{"name":"%s","arg":%s,"kind":1}|} name a in
  let inputs =
    [ arg "self" (as_tensor "x") ]
    @
    match approximate with
    | None -> []
    | Some a -> [ arg "approximate" (jstr {|{"as_string":"%s"}|} a) ]
  in
  jstr
    {|{"target":"torch.ops.aten.gelu.default","inputs":[%s],"outputs":[%s],"metadata":{}}|}
    (String.concat "," inputs) (as_tensor "y")

let%expect_test "gelu.default lowers to a Gelu node" =
  dump "approximate omitted:" (prog (gelu_node ()));
  dump "approximate=\"none\":" (prog (gelu_node ~approximate:"none" ()));
  dump "approximate=\"tanh\":" (prog (gelu_node ~approximate:"tanh" ()));
  [%expect
    {|
    approximate omitted:
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=3]] = gelu x=t0 approximate=none
    outputs: [t1 f32 [W=2 C=3] <-n0]
    approximate="none":
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=3]] = gelu x=t0 approximate=none
    outputs: [t1 f32 [W=2 C=3] <-n0]
    approximate="tanh":
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=3]] = gelu x=t0 approximate=tanh
    outputs: [t1 f32 [W=2 C=3] <-n0] |}]

(* The exact ("none") and tanh forms are the only two ATen spellings this
   engine implements; any other value would silently compute a different
   function under the right op name, so it is rejected by name rather than
   dropped. *)
let%expect_test "gelu.default rejects an unrecognized approximate" =
  dump "approximate=\"unsupported\":"
    (prog (gelu_node ~approximate:"unsupported" ()));
  [%expect
    {|
    approximate="unsupported":
      malformed PT2 graph: torch.ops.aten.gelu.default: approximate="unsupported" is not supported (only "none" or "tanh") |}]

let mul_node ?(other = `Tensor) () =
  let other_arg =
    match other with
    | `Tensor -> jstr {|{"name":"other","arg":%s,"kind":1}|} (as_tensor "other")
    | `Int i -> jstr {|{"name":"other","arg":{"as_int":%d},"kind":1}|} i
    | `Float f -> jstr {|{"name":"other","arg":{"as_float":%f},"kind":1}|} f
  in
  jstr
    {|{"target":"torch.ops.aten.mul.Tensor","inputs":[{"name":"self","arg":%s,"kind":1},%s],"outputs":[%s],"metadata":{}}|}
    (as_tensor "x") other_arg (as_tensor "y")

let mul_prog ?(x_sizes = [ 2; 3 ]) ?other_sizes node =
  match other_sizes with
  | None -> program ~x_sizes ~nodes:[ node ] ~graph_outputs:[ as_tensor "y" ] ()
  | Some other_sizes ->
      program ~x_sizes
        ~extra_tensor_values:[ ("other", tensor_meta other_sizes) ]
        ~params:[ "other" ] ~nodes:[ node ]
        ~graph_outputs:[ as_tensor "y" ]
        ()

let%expect_test "mul.Tensor tensor/tensor lowers to a Mul node" =
  dump "tensor/tensor:" (mul_prog ~other_sizes:[ 2; 3 ] (mul_node ()));
  [%expect
    {|
    tensor/tensor:
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0], t1 f32 [W=2 C=3] ->[n0] constant]
    nodes:
      n0: [t2 f32 [W=2 C=3]] = mul a=t0 b=t1
    outputs: [t2 f32 [W=2 C=3] <-n0] |}]

let%expect_test
    "mul.Tensor with a serialized scalar lowers to a Mul_scalar node" =
  dump "Int 3:" (mul_prog (mul_node ~other:(`Int 3) ()));
  dump "Float 0.1:" (mul_prog (mul_node ~other:(`Float 0.1) ()));
  [%expect
    {|
    Int 3:
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=3]] = mul_scalar x=t0 scalar=3
    outputs: [t1 f32 [W=2 C=3] <-n0]
    Float 0.1:
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=3]] = mul_scalar x=t0 scalar=0.1
    outputs: [t1 f32 [W=2 C=3] <-n0] |}]

let mul_scalar_node ?other () =
  let other_arg =
    match other with
    | None -> ""
    | Some (`Int i) -> jstr {|,{"name":"other","arg":{"as_int":%d},"kind":1}|} i
    | Some (`Float f) ->
        jstr {|,{"name":"other","arg":{"as_float":%f},"kind":1}|} f
    | Some `None -> jstr {|,{"name":"other","arg":{"as_none":true},"kind":1}|}
  in
  jstr
    {|{"target":"torch.ops.aten.mul.Scalar","inputs":[{"name":"self","arg":%s,"kind":1}%s],"outputs":[%s],"metadata":{}}|}
    (as_tensor "x") other_arg (as_tensor "y")

let%expect_test "mul.Scalar lowers to a Mul_scalar node" =
  dump "Int 3:" (prog (mul_scalar_node ~other:(`Int 3) ()));
  dump "Float 0.1:" (prog (mul_scalar_node ~other:(`Float 0.1) ()));
  [%expect
    {|
    Int 3:
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=3]] = mul_scalar x=t0 scalar=3
    outputs: [t1 f32 [W=2 C=3] <-n0]
    Float 0.1:
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=3]] = mul_scalar x=t0 scalar=0.1
    outputs: [t1 f32 [W=2 C=3] <-n0] |}]

(* [other] has no schema default: omission and an explicit none must both be
   rejected rather than silently read as multiplication by some placeholder
   value (the same failure mode [required_scalar_arg]'s doc comment names for
   batch-norm's [eps]). *)
let%expect_test "mul.Scalar rejects a missing or explicit-none other" =
  dump "omitted:" (prog (mul_scalar_node ()));
  dump "explicit none:" (prog (mul_scalar_node ~other:`None ()));
  [%expect
    {|
    omitted:
      malformed PT2 graph: torch.ops.aten.mul.Scalar: missing argument "other"
    explicit none:
      malformed PT2 graph: torch.ops.aten.mul.Scalar.other is not a scalar |}]
