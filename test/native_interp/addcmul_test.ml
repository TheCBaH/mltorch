(* torch.ops.aten.addcmul.default through the serialized path.

   Serialized twin of test/native_bridge/dispatch_test.ml's ATen-linked
   fixtures: self + value * tensor1 * tensor2, decomposed to the existing
   [Mul]/[Add]/[Mul_scalar] nodes rather than a new [Graph_ir] op -- see
   op_bridge_pointwise.ml's arm comment. This file only
   asserts what [Native_interp.lower] builds, mirroring binary_test.ml's
   [dump] pattern; numeric agreement is dispatch_test.ml's job. *)

open Programs

let addcmul_node ?value () =
  let value_arg =
    match value with
    | None -> ""
    | Some v -> jstr {|,{"name":"value","arg":{"as_float":%f},"kind":1}|} v
  in
  jstr
    {|{"target":"torch.ops.aten.addcmul.default","inputs":[{"name":"self","arg":%s,"kind":1},{"name":"tensor1","arg":%s,"kind":1},{"name":"tensor2","arg":%s,"kind":1}%s],"outputs":[%s],"metadata":{}}|}
    (as_tensor "x") (as_tensor "t1") (as_tensor "t2") value_arg (as_tensor "y")

let prog ?(x_sizes = [ 2; 3 ]) node =
  let captured = [ ("t1", x_sizes); ("t2", x_sizes) ] in
  program ~x_sizes
    ~extra_tensor_values:(List.map (fun (n, s) -> (n, tensor_meta s)) captured)
    ~params:(List.map fst captured) ~nodes:[ node ]
    ~graph_outputs:[ as_tensor "y" ]
    ()

let dump label json =
  Format.printf "%s@." label;
  match lower json with
  | Error e -> Format.printf "  %a@." Native_interp.pp_error (Err.Error.kind e)
  | Ok l -> Format.printf "%a@." Graph_ir.pp l.Pt2_native_graph.graph

let%expect_test "addcmul.default default value=1 decomposes to Mul+Add" =
  dump "value=1 (default):" (prog (addcmul_node ()));
  [%expect
    {|
    value=1 (default):
    graph
    inputs:
      [t0 f32 [W=2 C=3] ->[n1], t1 f32 [W=2 C=3] ->[n0] constant,
       t2 f32 [W=2 C=3] ->[n0] constant]
    nodes:
      n0: [t3 f32 [W=2 C=3] ->[n1]] = mul a=t1 b=t2
      n1: [t4 f32 [W=2 C=3]] = add a=t0 b=t3 <-n0
    outputs: [t4 f32 [W=2 C=3] <-n1] |}]

let%expect_test "addcmul.default non-unit value adds a Mul_scalar node" =
  dump "value=2:" (prog (addcmul_node ~value:2.0 ()));
  [%expect
    {|
    value=2:
    graph
    inputs:
      [t0 f32 [W=2 C=3] ->[n2], t1 f32 [W=2 C=3] ->[n0] constant,
       t2 f32 [W=2 C=3] ->[n0] constant]
    nodes:
      n0: [t3 f32 [W=2 C=3] ->[n1]] = mul a=t1 b=t2
      n1: [t4 f32 [W=2 C=3] ->[n2]] = mul_scalar x=t3 <-n0 scalar=2
      n2: [t5 f32 [W=2 C=3]] = add a=t0 b=t4 <-n1
    outputs: [t5 f32 [W=2 C=3] <-n2] |}]
