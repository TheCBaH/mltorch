(* torch.ops.aten.cumsum.default through the serialized path. Numeric
   agreement (against real ATen) is checked in
   test/native_bridge/cumsum_test.ml; this file asserts what
   [Native_interp.lower] builds, mirroring copy_test.ml's [dump] pattern.
   The only arm in native_interp_lower_reduce.ml, so it is also this file's
   whole subject. *)

open Programs

(* [dtype] is [Pytorch_types.ScalarType]'s int code -- 5 = LONG, 7 = FLOAT
   (see to_copy_test.ml). [None] omits the argument the way the schema
   default does; [`Null] serializes the explicit null the exporter also
   emits. *)
let cumsum_node ?dtype ~dim () =
  let dtype_arg =
    match dtype with
    | None -> ""
    | Some `Null -> {|,{"name":"dtype","arg":{"as_none":false},"kind":2}|}
    | Some (`Code code) ->
        jstr {|,{"name":"dtype","arg":{"as_scalar_type":%d},"kind":2}|} code
  in
  jstr
    {|{"target":"torch.ops.aten.cumsum.default","inputs":[{"name":"self","arg":%s,"kind":1},{"name":"dim","arg":{"as_int":%d},"kind":1}%s],"outputs":[%s],"metadata":{}}|}
    (as_tensor "x") dim dtype_arg (as_tensor "y")

let prog ?(x_sizes = [ 2; 3 ]) ?dtype ~dim () =
  program ~x_sizes
    ~nodes:[ cumsum_node ?dtype ~dim () ]
    ~graph_outputs:[ as_tensor "y" ]
    ()

let dump label json =
  Format.printf "%s@." label;
  match lower json with
  | Error e -> Format.printf "  %a@." Native_interp.pp_error (Err.Error.kind e)
  | Ok l -> Format.printf "%a@." Graph_ir.pp l.Pt2_native_graph.graph

(* Cumsum drops no axis, so the output shape is the input's -- and [dim] is
   read against the SERIALIZED rank, which right-aligns rank 2 onto W/C. *)
let%expect_test "cumsum names the axis dim selects" =
  dump "dim=1" (prog ~dim:1 ());
  [%expect
    {|
    dim=1
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=3]] = cumsum x=t0 params={axis=C}
    outputs: [t1 f32 [W=2 C=3] <-n0] |}]

let%expect_test "cumsum, negative dim normalizes to the last axis" =
  dump "dim=-1" (prog ~dim:(-1) ());
  [%expect
    {|
    dim=-1
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=3]] = cumsum x=t0 params={axis=C}
    outputs: [t1 f32 [W=2 C=3] <-n0] |}]

let%expect_test "cumsum, dim=0" =
  dump "dim=0" (prog ~dim:0 ());
  [%expect
    {|
    dim=0
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=3]] = cumsum x=t0 params={axis=W}
    outputs: [t1 f32 [W=2 C=3] <-n0] |}]

(* Native's compute domain is always float, so a FLOAT target is the
   identity and lowers to the same node the omitted/null forms do; every
   other target is refused rather than silently accepted. *)
let%expect_test "cumsum accepts the schema default and FLOAT dtype" =
  dump "explicit null" (prog ~dim:1 ~dtype:`Null ());
  dump "FLOAT" (prog ~dim:1 ~dtype:(`Code 7) ());
  [%expect
    {|
    explicit null
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=3]] = cumsum x=t0 params={axis=C}
    outputs: [t1 f32 [W=2 C=3] <-n0]
    FLOAT
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=3]] = cumsum x=t0 params={axis=C}
    outputs: [t1 f32 [W=2 C=3] <-n0] |}]

let%expect_test "cumsum rejects a non-FLOAT dtype" =
  dump "LONG" (prog ~dim:1 ~dtype:(`Code 5) ());
  [%expect
    {|
    LONG
      malformed PT2 graph: torch.ops.aten.cumsum.default: dtype is not supported |}]

(* [dim] outside self's own rank is refused with the same typed row every
   other single-[dim] arm reports for the identical fault. *)
let%expect_test "cumsum rejects a dim outside self's rank" =
  dump "out of range" (prog ~dim:2 ());
  [%expect
    {|
    out of range
      malformed PT2 graph: invalid dimension 2 for rank 2 |}]
