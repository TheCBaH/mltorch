open Programs

let node ?(args = []) () =
  jstr
    {|{"target":"torch.ops.aten.diff.default","inputs":[{"name":"self","arg":%s,"kind":1}%s],"outputs":[%s],"metadata":{}}|}
    (as_tensor "x")
    (String.concat "" (List.map (fun a -> "," ^ a) args))
    (as_tensor "y")

let int name value =
  jstr {|{"name":"%s","arg":{"as_int":%d},"kind":1}|} name value

let prepend = jstr {|{"name":"prepend","arg":%s,"kind":1}|} (as_tensor "p")

let%expect_test "diff importer: defaults and checked source rank" =
  let run ?(x_sizes = [ 2; 3 ]) ?p_sizes args =
    let params, extra_tensor_values =
      match p_sizes with
      | None -> ([], [])
      | Some sizes -> ([ "p" ], [ ("p", tensor_meta sizes) ])
    in
    show "diff"
      (program ~x_sizes ~params ~extra_tensor_values
         ~nodes:[ node ~args () ]
         ~graph_outputs:[ as_tensor "y" ]
         ())
  in
  run [];
  run ~p_sizes:[ 2; 1 ] [ prepend ];
  run ~p_sizes:[ 1; 2; 1 ] [ prepend ];
  run ~p_sizes:[ 3; 1 ] [ prepend ];
  run [ int "dim" 2 ];
  run [ int "n" 2 ];
  run ~x_sizes:[] [];
  run ~x_sizes:[ 1 ] [];
  run [ int "prepend" 2 ];
  [%expect
    {|
    diff                       lowered, nodes=1
    diff                       lowered, nodes=1
    diff                       malformed PT2 graph: p is rank 3, expected 2
    diff                       concat: axis W extent must agree across every tensor (it is not the concatenated axis): 2 vs 3
    diff                       malformed PT2 graph: invalid dimension 2 for rank 2
    diff                       diff: supported orders are 0 and 1, got 2
    diff                       malformed PT2 graph: invalid dimension -1 for rank 0
    diff                       diff: empty results are not represented
    diff                       malformed PT2 graph: torch.ops.aten.diff.default.prepend is not an optional tensor |}]

let%expect_test "diff importer: int64, Bool and unsupported source metadata" =
  let p =
    program ~x_sizes:[ 3 ]
      ~nodes:[ node () ]
      ~graph_outputs:[ as_tensor "y" ]
      ()
    |> Jsont_bytesrw.decode_string Pytorch_types.ExportedProgram.jsont
    |> Result.get_ok
  in
  let gm = p.Pytorch_types.ExportedProgram.graph_module in
  let g = gm.Pytorch_types.GraphModule.graph in
  let module Sm = Schema_runtime.String_map in
  let x = Sm.find "x" g.Pytorch_types.Graph.tensor_values in
  List.iter
    (fun dtype ->
      let tensor_values =
        Sm.add "x" { x with Pytorch_types.TensorMeta.dtype } g.tensor_values
      in
      let p =
        { p with graph_module = { gm with graph = { g with tensor_values } } }
      in
      match Native_interp.lower p with
      | Error e ->
          Format.printf "%a@." Native_interp.pp_error (Err.Error.kind e)
      | Ok lowered ->
          Format.printf "%a@." Graph_ir.pp lowered.Pt2_native_graph.graph)
    Pytorch_types.ScalarType.[ BOOL; DOUBLE; LONG ];
  [%expect
    {|
    diff: unsupported dtype bool
    malformed PT2 graph: torch.ops.aten.diff.default: dtype is not supported
    graph
    inputs: [t0 i64 [C=3] ->[n0]]
    nodes:
      n0: [t1 i64 [C=2]] = diff x=t0 axis=C order=1 prepend= append=
    outputs: [t1 i64 [C=2] <-n0] |}]
