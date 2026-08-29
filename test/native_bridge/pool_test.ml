(* the PT2 importer's max_pool2d.default arm, against ATen. Split from the former native_bridge_test.ml; promote with [dune promote test/native_bridge/pool_test.ml]. *)

open Helpers

(* ---- the PT2 importer's max_pool2d.default arm, against ATen ------------- *)

(* The input values are all NEGATIVE on purpose. Max-pooling's padding region
   must contribute -inf, not 0; a naive guarded read returning 0 beats every
   real value here and the result is wrong at every border window -- which, with
   pad = 1 on a small input, is everywhere. A non-negative fixture cannot see
   that at all. *)
let pool_program ~x_sizes ~kernel ~stride ~padding =
  let as_t n = jstr {|{"as_tensor":{"name":"%s"}}|} n in
  let ints (h, w) = jstr {|{"as_ints":[%d,%d]}|} h w in
  let node =
    jstr
      {|{"target":"torch.ops.aten.max_pool2d.default","inputs":[{"name":"self","arg":%s,"kind":1},{"name":"kernel_size","arg":%s,"kind":1},{"name":"stride","arg":%s,"kind":1},{"name":"padding","arg":%s,"kind":1}],"outputs":[%s],"metadata":{}}|}
      (as_t "x") (ints kernel) (ints stride) (ints padding) (as_t "y")
  in
  jstr
    {|{"graph_module":{"graph":{"inputs":[%s],"outputs":[%s],"nodes":[%s],"tensor_values":{%s},"sym_int_values":{},"sym_bool_values":{},"is_single_tensor_return":true},"signature":{"input_specs":[{"user_input":{"arg":%s}}],"output_specs":[{"user_output":{"arg":%s}}]},"module_call_graph":[]},"opset_version":{"aten":15},"range_constraints":{},"schema_version":{"major":8,"minor":5}}|}
    (as_t "x") (as_t "y") node
    (jstr {|"x":%s|} (meta_json x_sizes))
    (as_t "x") (as_t "y")

let pool_vs_aten label ~x_sizes ~kernel ~stride ~padding =
  let n = List.fold_left ( * ) 1 x_sizes in
  let xs = List.init n (fun i -> -1. -. (float_of_int (i mod 11) /. 2.)) in
  Format.printf "%-22s " label;
  match
    Jsont_bytesrw.decode_string PT.ExportedProgram.jsont
      (pool_program ~x_sizes ~kernel ~stride ~padding)
  with
  | Error e -> Format.printf "fixture did not decode: %s@." e
  | Ok program -> (
      let node = List.hd program.PT.ExportedProgram.graph_module.graph.nodes in
      match
        Interp_dispatch.dispatch
          (Sm.add "x" (float_tensor x_sizes xs) Sm.empty)
          node
      with
      | Error e ->
          Format.printf "aten: %a@." Interp_verify.pp_interp_error
            (Err.Error.kind e)
      | Ok aten_out -> (
          match Native_interp.lower program with
          | Error e ->
              Format.printf "lower: %a@." Native_interp.pp_error
                (Err.Error.kind e)
          | Ok lowered -> (
              let g = lowered.Pt2_native_graph.graph in
              match
                Eval_direct.run g
                  ~inputs:
                    [ (List.hd g.Graph_ir.Graph.inputs, native_f32 x_sizes xs) ]
              with
              | Error e ->
                  Format.printf "eval: %a@." Eval_direct.pp_error
                    (Err.Error.kind e)
              | Ok result ->
                  Format.printf "%a@." pp_result
                    (Verify.compare_tensors ~atol:0. ~output:"y"
                       (Sm.find "y" aten_out)
                       (Graph_ir.Tensor_id.Map.find
                          (List.hd g.Graph_ir.Graph.outputs)
                          result)))))

let%expect_test "importer: max_pool2d.default matches ATen" =
  pool_vs_aten "square, no pad:" ~x_sizes:[ 1; 3; 8; 8 ] ~kernel:(2, 2)
    ~stride:(2, 2) ~padding:(0, 0);
  pool_vs_aten "padded:" ~x_sizes:[ 1; 3; 5; 5 ] ~kernel:(3, 3) ~stride:(1, 1)
    ~padding:(1, 1);
  (* Rectangular everything: a swapped stride or padding pair is invariant for a
     square configuration and visible here. *)
  pool_vs_aten "rectangular:" ~x_sizes:[ 1; 3; 9; 7 ] ~kernel:(3, 2)
    ~stride:(2, 1) ~padding:(1, 0);
  [%expect
    {|
    square, no pad:        Ok
    padded:                Ok
    rectangular:           Ok |}]
