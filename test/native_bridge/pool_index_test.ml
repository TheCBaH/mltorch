(* The live index output of max_pool2d_with_indices is an exact int64 like
   ATen's: real ATen's [max_pool2d_with_indices.default] is run on the same input
   and its int64 indices compared, element for element, with what Native Direct
   writes ([Verify.compare_tensors] matches [Long] against [I64] exactly). *)

open Helpers

let x_sizes = [ 1; 1; 4; 4 ]

(* (5*i mod 11): every 2x2 window has a distinct maximum, so the answer does not
   lean on a tie convention. *)
let xs = List.init 16 (fun i -> float_of_int (5 * i mod 11))

let program =
  let as_t n = jstr {|{"as_tensor":{"name":"%s"}}|} n in
  let ints (h, w) = jstr {|{"as_ints":[%d,%d]}|} h w in
  let node =
    jstr
      {|{"target":"torch.ops.aten.max_pool2d_with_indices.default","inputs":[{"name":"self","arg":%s,"kind":1},{"name":"kernel_size","arg":%s,"kind":1},{"name":"stride","arg":%s,"kind":1},{"name":"padding","arg":%s,"kind":1}],"outputs":[%s,%s],"metadata":{}}|}
      (as_t "x")
      (ints (2, 2))
      (ints (2, 2))
      (ints (0, 0))
      (as_t "y") (as_t "idx")
  in
  jstr
    {|{"graph_module":{"graph":{"inputs":[%s],"outputs":[%s,%s],"nodes":[%s],"tensor_values":{%s},"sym_int_values":{},"sym_bool_values":{},"is_single_tensor_return":false},"signature":{"input_specs":[{"user_input":{"arg":%s}}],"output_specs":[{"user_output":{"arg":%s}},{"user_output":{"arg":%s}}]},"module_call_graph":[]},"opset_version":{"aten":15},"range_constraints":{},"schema_version":{"major":8,"minor":5}}|}
    (as_t "x") (as_t "y") (as_t "idx") node
    (jstr {|"x":%s|} (meta_json x_sizes))
    (as_t "x") (as_t "y") (as_t "idx")

let pool_params : Pool.MaxPool2dWithIndices.params =
  {
    ceil_mode = false;
    kernel = { h = Dim.extent 2; w = Dim.extent 2 };
    stride = { h = Op_config.Pos.of_int 2; w = Op_config.Pos.of_int 2 };
    pad = { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
  }

(* Channel-last back to NCHW, as the importer does for its own outputs. *)
let nhwc_to_nchw : Permute.Permute.perm =
  let open Axis in
  [ (N, N); (T, T); (D, D); (H, C); (W, H); (C, W) ]

let%expect_test "native live indices equal ATen's int64 indices" =
  match Jsont_bytesrw.decode_string PT.ExportedProgram.jsont program with
  | Error e -> Format.printf "fixture did not decode: %s@." e
  | Ok program ->
      (let node = List.hd program.PT.ExportedProgram.graph_module.graph.nodes in
       match
         Interp_dispatch.dispatch
           (Sm.add "x" (float_tensor x_sizes xs) Sm.empty)
           node
       with
       | Error e ->
           Format.printf "aten: %a@." Interp_verify.pp_interp_error
             (Err.Error.kind e)
       | Ok aten_out ->
           (* N=1, C=1, so NCHW and NHWC share one flat order. *)
           let shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:4 ~w:4 ~c:1 in
           let g =
             Graph_builder.build ~name:"pool_index"
               ~outputs:(fun (v, i) -> [ v; i ])
               Graph_builder.(
                 let* x = input ~shape ~name:"x" () in
                 let* v, i = max_pool2d_with_indices pool_params x in
                 let* v = permute nhwc_to_nchw v in
                 let* i = permute nhwc_to_nchw i in
                 return (v, i))
             |> Err.or_raise ~pp_error:Graph_builder.pp_error
           in
           let x =
             Tensor.materialize shape (fun c ->
                 float_of_int
                   (5
                   * ((Dim.to_int (Vec6.get c Axis.H) * 4)
                     + Dim.to_int (Vec6.get c Axis.W))
                   mod 11))
           in
           let env =
             Eval_direct.run g
               ~inputs:(List.combine g.Graph_ir.Graph.inputs [ x ])
             |> Err.or_raise ~pp_error:Eval_direct.pp_error
           in
           let value_id, index_id =
             match g.Graph_ir.Graph.outputs with
             | [ v; i ] -> (v, i)
             | _ -> assert false
           in
           let compare name aten native =
             Format.printf "%s: %a@." name pp_result
               (Verify.compare_tensors ~atol:0. ~output:name aten native)
           in
           compare "values" (Sm.find "y" aten_out)
             (Graph_ir.Tensor_id.Map.find value_id env);
           compare "indices" (Sm.find "idx" aten_out)
             (Graph_ir.Tensor_id.Map.find index_id env));
      [%expect {|
    values: Ok
    indices: Ok |}]
