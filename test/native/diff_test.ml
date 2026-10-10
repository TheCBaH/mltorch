open Graph_ir

let params ?(order = 1) axis =
  Diff.Diff.{ axis; order = Aten_int.Size.of_int order }

let shape n = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:n

let graph fmt ?(order = 1) () =
  Graph_builder.(
    build ~name:"diff"
      ~outputs:(fun r -> [ r ])
      (let* x = input ~fmt ~shape:(shape 4) () in
       let* p = input ~fmt ~shape:(shape 2) () in
       let* a = input ~fmt ~shape:(shape 1) () in
       diff (params ~order Axis.C) x ~prepend:p ~append:a ()))
  |> Err.or_raise ~pp_error:Graph_builder.pp_error

let compare_routes g tensors =
  let bindings = List.combine g.Graph.inputs tensors in
  let direct =
    Eval_direct.run g ~inputs:bindings
    |> Err.or_raise ~pp_error:Eval_direct.pp_error
  in
  let staged = Eval_symbolic.run g in
  let kernel =
    Kernel_adapt.of_stage_program { staged with Stage_program.outputs = [] }
    |> Err.or_raise ~pp_error:Kernel_adapt.pp_error
  in
  let grounded =
    Kernel_eval.run kernel ~bind:(fun id -> List.assoc_opt id bindings)
    |> Err.or_raise ~pp_error:Kernel_eval.pp_error
  in
  let id = List.hd g.Graph.outputs in
  Format.printf "direct: %a@.kernel: %a@." Tensor.pp
    (Tensor_id.Map.find id direct)
    Tensor.pp
    (Tensor_id.Map.find id grounded)

let%expect_test
    "diff: exact integer direct and symbolic segments, including wrapping" =
  let g = graph Payload.(Fmt I64) () in
  let tensor values =
    Tensor.materialize_i64
      (shape (Array.length values))
      (fun coord -> values.((coord.Vec6.c :> int)))
  in
  let tensors =
    List.map tensor
      [
        [|
          9_007_199_254_740_993L;
          9_007_199_254_740_994L;
          Int64.max_int;
          Int64.min_int;
        |];
        [| 9_007_199_254_740_990L; 9_007_199_254_740_992L |];
        [| 0L |];
      ]
  in
  compare_routes g tensors;
  let encoded =
    Jsont_bytesrw.encode_string Graph_ir.graph_jsont g |> Result.get_ok
  in
  let decoded =
    Jsont_bytesrw.decode_string Graph_ir.graph_jsont encoded |> Result.get_ok
  in
  compare_routes decoded tensors;
  compare_routes (graph Payload.(Fmt I64) ~order:0 ()) tensors;
  [%expect
    {|
    direct: tensor i64 [C=6] {2, 1, 1, 9214364837600034813, 1, -9223372036854775808}
    kernel: tensor i64 [C=6] {2, 1, 1, 9214364837600034813, 1, -9223372036854775808}
    direct: tensor i64 [C=6] {2, 1, 1, 9214364837600034813, 1, -9223372036854775808}
    kernel: tensor i64 [C=6] {2, 1, 1, 9214364837600034813, 1, -9223372036854775808}
    direct: tensor i64 [C=4] {9007199254740993, 9007199254740994, 9223372036854775807, -9223372036854775808}
    kernel: tensor i64 [C=4] {9007199254740993, 9007199254740994, 9223372036854775807, -9223372036854775808} |}]

let%expect_test "diff: float segments and zero-order identity" =
  let tensor values =
    Tensor.materialize
      (shape (Array.length values))
      (fun coord -> values.((coord.Vec6.c :> int)))
  in
  let tensors =
    List.map tensor [ [| 1.; 4.; 9.; 16. |]; [| -3.; 0. |]; [| 25. |] ]
  in
  compare_routes (graph Payload.(Fmt F32) ()) tensors;
  compare_routes (graph Payload.(Fmt F32) ~order:0 ()) tensors;
  [%expect
    {|
    direct: tensor f32 [C=6] {3, 1, 3, 5, 7, 9}
    kernel: tensor f32 [C=6] {3, 1, 3, 5, 7, 9}
    direct: tensor f32 [C=4] {1, 4, 9, 16}
    kernel: tensor f32 [C=4] {1, 4, 9, 16} |}]

let%expect_test
    "diff: order and empty-output checks are shared by graph construction" =
  let build n order =
    Graph_builder.(
      build ~name:"diff"
        ~outputs:(fun r -> [ r ])
        (let* x = input ~shape:(shape n) () in
         diff (params ~order Axis.C) x ()))
  in
  List.iter
    (fun (n, order) ->
      Format.printf "%a@."
        (Core.Pretty.err_result
           ~ok:(Fmt.any "unexpected success")
           ~error:Graph_builder.pp_error)
        (build n order))
    [ (4, -1); (4, 2); (1, 1) ];
  [%expect
    {|
    diff: supported orders are 0 and 1, got -1
    diff: supported orders are 0 and 1, got 2
    diff: empty results are not represented |}]

let%expect_test "diff: decoded graph cannot bypass order validation" =
  let g = graph Payload.(Fmt I64) () in
  let nodes =
    List.map
      (fun (node : node) ->
        match node.Node.op with
        | Diff d ->
            {
              node with
              Node.op =
                Diff { d with Diff.Diff.params = params ~order:2 Axis.C };
            }
        | _ -> node)
      g.Graph.nodes
  in
  let g = { g with Graph.nodes } in
  let encoded =
    Jsont_bytesrw.encode_string Graph_ir.graph_jsont g |> Result.get_ok
  in
  let g =
    Jsont_bytesrw.decode_string Graph_ir.graph_jsont encoded |> Result.get_ok
  in
  let bindings =
    List.map
      (fun id ->
        let sg = Tensor_id.Map.find id g.Graph.tensors in
        (id, Tensor.materialize_i64 sg.Tensor_sig.shape (fun _ -> 0L)))
      g.Graph.inputs
  in
  Format.printf "direct: %a@."
    (Core.Pretty.err_result
       ~ok:(Fmt.any "unexpected success")
       ~error:Eval_direct.pp_error)
    (Eval_direct.run g ~inputs:bindings);
  (try
     ignore (Eval_symbolic.run g);
     print_endline "unexpected symbolic success"
   with Err.Exn.E e -> Format.printf "symbolic: %a@." Err.Exn.pp_kind e);
  [%expect
    {|
    direct: diff: supported orders are 0 and 1, got 2
    symbolic: diff: supported orders are 0 and 1, got 2 |}]
