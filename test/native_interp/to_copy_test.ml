(* torch.ops.aten._to_copy.default through the serialized path. Numeric
   agreement (against real ATen, where reachable) is checked in
   test/native_bridge/to_copy_test.ml; this file asserts what
   [Native_interp.lower] builds, mirroring copy_test.ml's [dump] pattern.
   Restricted to the three-way [Pointwise.To_copy.target] domain the corpus
   actually uses (bool/float/long) -- see native_interp_lower_shape.ml's own
   comment. *)

open Programs

(* [dtype] is [Pytorch_types.ScalarType]'s int code -- 5 = LONG, 7 = FLOAT,
   12 = BOOL (see modules/pytorch/torch/_export/serde/schema.yaml). [None]
   omits the argument entirely, matching an exporter that keeps self's own
   dtype. *)
let to_copy_node ?dtype ~non_blocking () =
  let dtype_arg =
    match dtype with
    | None -> {|{"name":"dtype","arg":{"as_none":false},"kind":2}|}
    | Some code ->
        jstr {|{"name":"dtype","arg":{"as_scalar_type":%d},"kind":2}|} code
  in
  jstr
    {|{"target":"torch.ops.aten._to_copy.default","inputs":[{"name":"self","arg":%s,"kind":1},%s,{"name":"non_blocking","arg":{"as_bool":%b},"kind":2}],"outputs":[%s],"metadata":{}}|}
    (as_tensor "x") dtype_arg non_blocking (as_tensor "y")

let prog ?(x_sizes = [ 3 ]) ?dtype ?(non_blocking = false) () =
  program ~x_sizes
    ~nodes:[ to_copy_node ?dtype ~non_blocking () ]
    ~graph_outputs:[ as_tensor "y" ]
    ()

let dump label json =
  Format.printf "%s@." label;
  match lower json with
  | Error e -> Format.printf "  %a@." Native_interp.pp_error (Err.Error.kind e)
  | Ok l -> Format.printf "%a@." Graph_ir.pp l.Pt2_native_graph.graph

let%expect_test "_to_copy with no dtype lowers to to_copy target=float" =
  dump "no dtype" (prog ());
  [%expect
    {|
    no dtype
    graph
    inputs: [t0 f32 [C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [C=3]] = to_copy x=t0 target=float
    outputs: [t1 f32 [C=3] <-n0] |}]

let%expect_test "_to_copy dtype=FLOAT/LONG/BOOL lowers to the matching target" =
  dump "FLOAT" (prog ~dtype:7 ());
  dump "LONG" (prog ~dtype:5 ());
  dump "BOOL" (prog ~dtype:12 ());
  [%expect
    {|
    FLOAT
    graph
    inputs: [t0 f32 [C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [C=3]] = to_copy x=t0 target=float
    outputs: [t1 f32 [C=3] <-n0]
    LONG
    graph
    inputs: [t0 f32 [C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [C=3]] = to_copy x=t0 target=long
    outputs: [t1 f32 [C=3] <-n0]
    BOOL
    graph
    inputs: [t0 f32 [C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [C=3]] = to_copy x=t0 target=bool
    outputs: [t1 f32 [C=3] <-n0] |}]

(* Outside the three-way corpus-evidenced domain: rejected with a typed
   diagnostic, the same [Unsupported_option]/[`Dtype] mechanism
   [zeros.default]/[arange.default] use for their own dtype restriction. *)
let%expect_test "_to_copy rejects an unsupported dtype" =
  dump "INT (dtype code 4)" (prog ~dtype:4 ());
  [%expect
    {|
    INT (dtype code 4)
      malformed PT2 graph: torch.ops.aten._to_copy.default: dtype is not supported |}]

(* [non_blocking] is read-and-discarded -- same graph, either spelling, the
   same proof [copy_test.ml] makes for its own [non_blocking]. *)
let%expect_test "_to_copy: non_blocking carries no computational effect" =
  dump "non_blocking=false" (prog ~dtype:5 ~non_blocking:false ());
  dump "non_blocking=true" (prog ~dtype:5 ~non_blocking:true ());
  [%expect
    {|
    non_blocking=false
    graph
    inputs: [t0 f32 [C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [C=3]] = to_copy x=t0 target=long
    outputs: [t1 f32 [C=3] <-n0]
    non_blocking=true
    graph
    inputs: [t0 f32 [C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [C=3]] = to_copy x=t0 target=long
    outputs: [t1 f32 [C=3] <-n0] |}]
