(* torch.ops.aten.sub.Tensor through the serialized path.

   No model in this repo's zoo serializes [sub.Tensor] (op3-impl.md F3), so
   these fixtures are the coverage, not a supplement to it. Numeric agreement
   against real ATen is checked in test/native_bridge_test.ml, whose
   [verify_print] harness runs both backends; this file only asserts what
   [Native_interp.lower] builds, mirroring conv_test.ml's [dump] pattern. *)

open Programs

let sub_node ?(other = `Tensor) ?alpha () =
  let other_arg =
    match other with
    | `Tensor -> jstr {|{"name":"other","arg":%s,"kind":1}|} (as_tensor "other")
    | `Int i -> jstr {|{"name":"other","arg":{"as_int":%d},"kind":1}|} i
    | `Float f -> jstr {|{"name":"other","arg":{"as_float":%f},"kind":1}|} f
  in
  let alpha_arg =
    match alpha with
    | None -> ""
    | Some (`Int i) -> jstr {|,{"name":"alpha","arg":{"as_int":%d},"kind":1}|} i
    | Some (`Float f) ->
        jstr {|,{"name":"alpha","arg":{"as_float":%f},"kind":1}|} f
  in
  jstr
    {|{"target":"torch.ops.aten.sub.Tensor","inputs":[{"name":"self","arg":%s,"kind":1},%s%s],"outputs":[%s],"metadata":{}}|}
    (as_tensor "x") other_arg alpha_arg (as_tensor "y")

let prog ?(x_sizes = [ 2; 3 ]) ?other_sizes node =
  match other_sizes with
  | None -> program ~x_sizes ~nodes:[ node ] ~graph_outputs:[ as_tensor "y" ] ()
  | Some other_sizes ->
      program ~x_sizes
        ~extra_tensor_values:[ ("other", tensor_meta other_sizes) ]
        ~params:[ "other" ] ~nodes:[ node ]
        ~graph_outputs:[ as_tensor "y" ]
        ()

let dump label json =
  Format.printf "%s@." label;
  match lower json with
  | Error e -> Format.printf "  %a@." Native_interp.pp_error (Err.Error.kind e)
  | Ok l -> Format.printf "%a@." Graph_ir.pp l.Pt2_native_graph.graph

let%expect_test "sub.Tensor tensor/tensor lowers to a Sub node" =
  dump "tensor/tensor:" (prog ~other_sizes:[ 2; 3 ] (sub_node ()));
  [%expect
    {|
    tensor/tensor:
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0], t1 f32 [W=2 C=3] ->[n0] constant]
    nodes:
      n0: [t2 f32 [W=2 C=3]] = sub a=t0 b=t1
    outputs: [t2 f32 [W=2 C=3] <-n0] |}]

(* op3-impl.md F6: the scalar spelling has no [Op_spec] fixture form, so this
   graph-structure assertion (and test/native_bridge_test.ml's ATen-verified
   evaluation) are the only evidence for it. Both the Int and Float spellings
   route to [Add_scalar] with the NEGATED value -- [x - s] legalizes to
   [x + (-s)], not to a dedicated [Sub_scalar] op. *)
let%expect_test "sub.Tensor with a serialized scalar lowers to add_scalar(-s)" =
  dump "Int -3:" (prog (sub_node ~other:(`Int (-3)) ()));
  dump "Float 2.5:" (prog (sub_node ~other:(`Float 2.5) ()));
  [%expect
    {|
    Int -3:
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=3]] = add_scalar x=t0 scalar=3
    outputs: [t1 f32 [W=2 C=3] <-n0]
    Float 2.5:
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=3]] = add_scalar x=t0 scalar=-2.5
    outputs: [t1 f32 [W=2 C=3] <-n0] |}]

(* [self + alpha * other] is not what this arm computes, matching
   [add.Tensor]'s existing contract: reject rather than silently drop, on
   both the tensor and the scalar path. alpha=1 is the default and must still
   lower. *)
let%expect_test "sub.Tensor rejects a non-default alpha" =
  dump "alpha=2, tensor:"
    (prog ~other_sizes:[ 2; 3 ] (sub_node ~alpha:(`Int 2) ()));
  dump "alpha=2.5, scalar:"
    (prog (sub_node ~other:(`Int 3) ~alpha:(`Float 2.5) ()));
  dump "alpha=1, tensor:"
    (prog ~other_sizes:[ 2; 3 ] (sub_node ~alpha:(`Int 1) ()));
  [%expect
    {|
    alpha=2, tensor:
      malformed PT2 graph: torch.ops.aten.sub.Tensor: alpha=2 is not supported (only 1)
    alpha=2.5, scalar:
      malformed PT2 graph: torch.ops.aten.sub.Tensor: alpha=2.5 is not supported (only 1)
    alpha=1, tensor:
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0], t1 f32 [W=2 C=3] ->[n0] constant]
    nodes:
      n0: [t2 f32 [W=2 C=3]] = sub a=t0 b=t1
    outputs: [t2 f32 [W=2 C=3] <-n0] |}]

(* Broadcasting is the native rule, not an importer check: [other] declared
   [1,3] against a [2,3] self. *)
let%expect_test "sub.Tensor broadcasts a [1,3] other" =
  dump "[1,3] vs [2,3]:" (prog ~other_sizes:[ 1; 3 ] (sub_node ()));
  [%expect
    {|
    [1,3] vs [2,3]:
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0], t1 f32 [C=3] ->[n0] constant]
    nodes:
      n0: [t2 f32 [W=2 C=3]] = sub a=t0 b=t1
    outputs: [t2 f32 [W=2 C=3] <-n0] |}]
