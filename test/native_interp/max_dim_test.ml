(* torch.ops.aten.max.dim through the serialized path. Numeric agreement
   (against real ATen) is checked in test/native_bridge/reduce_dispatch_test.ml
   via [Op_bridge]; this file asserts what [Native_interp.lower] builds,
   mirroring pool_test.ml's [adaptive_max_pool2d_with_indices] pattern --
   except [max.dim]'s index is liveness-checked ([ctx.reads]) rather than
   unconditionally discarded, so both the dead- and live-index shapes need
   their own fixture here. *)

open Programs

let max_dim_node ~dim ~keepdim =
  jstr
    {|{"target":"torch.ops.aten.max.dim","inputs":[{"name":"self","arg":%s,"kind":1},{"name":"dim","arg":{"as_int":%d},"kind":1},{"name":"keepdim","arg":{"as_bool":%b},"kind":1}],"outputs":[%s,%s],"metadata":{}}|}
    (as_tensor "x") dim keepdim (as_tensor "y") (as_tensor "i")

let dump label json =
  Format.printf "%s@." label;
  match lower json with
  | Error e -> Format.printf "  %a@." Native_interp.pp_error (Err.Error.kind e)
  | Ok l -> Format.printf "%a@." Graph_ir.pp l.Pt2_native_graph.graph

let%expect_test "max.dim discards a dead index" =
  dump "dead index:"
    (program ~x_sizes:[ 2; 3 ]
       ~nodes:[ max_dim_node ~dim:1 ~keepdim:false ]
       ~graph_outputs:[ as_tensor "y" ]
       ());
  [%expect
    {|
    dead index:
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      group g1 torch.ops.aten.max.dim:
        n0: [t1 f32 [C=2], t2 i64 [C=2] ->[n1]] =
          max_dim x=t0 params={axis=C; keepdim=false}
        n1: [] = discard x=t2 <-n0
    outputs: [t1 f32 [C=2] <-n0] |}]

(* Both names appear in [graph_outputs], so [ctx.reads] marks both live: no
   [Discard] node at all, unlike the dead-index fixture above. *)
let%expect_test "max.dim retains a live index" =
  dump "live index:"
    (program ~x_sizes:[ 2; 3 ]
       ~nodes:[ max_dim_node ~dim:1 ~keepdim:false ]
       ~graph_outputs:[ as_tensor "y"; as_tensor "i" ]
       ());
  [%expect
    {|
    live index:
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      group g1 torch.ops.aten.max.dim:
        n0: [t1 f32 [C=2], t2 i64 [C=2]] =
          max_dim x=t0 params={axis=C; keepdim=false}
    outputs: [t1 f32 [C=2] <-n0, t2 i64 [C=2] <-n0] |}]

let%expect_test "max.dim keepdim=true collapses the axis in place" =
  dump "keepdim:"
    (program ~x_sizes:[ 2; 3 ]
       ~nodes:[ max_dim_node ~dim:1 ~keepdim:true ]
       ~graph_outputs:[ as_tensor "y"; as_tensor "i" ]
       ());
  [%expect
    {|
    keepdim:
    graph
    inputs: [t0 f32 [W=2 C=3] ->[n0]]
    nodes:
      group g1 torch.ops.aten.max.dim:
        n0: [t1 f32 [W=2 C=1], t2 i64 [W=2 C=1]] =
          max_dim x=t0 params={axis=C; keepdim=true}
    outputs: [t1 f32 [W=2 C=1] <-n0, t2 i64 [W=2 C=1] <-n0] |}]
