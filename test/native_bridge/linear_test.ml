(* the PT2 importer's linear.default arm, against ATen. Split from the former native_bridge_test.ml; promote with [dune promote test/native_bridge/linear_test.ml]. *)

open Helpers

(* ---- the PT2 importer's linear.default arm, against ATen ----------------- *)

(* Same shape of evidence, and the same reason for it: `linear.default` is not
   serialized by any downloadable model either. The weights here are all
   NON-SQUARE and the input rank varies, because the two mutations that matter
   -- transposing the weight permutation, and taking [in_features] from dim 0
   instead of dim 1 -- both leave a square weight's graph buildable. *)
(* [bias] is three-valued for the same reason it is in conv_test.ml, and here it
   turns up a divergence: the ATen decode path requires the argument to be
   PRESENT (Interp_decode.tensor_arg reports `Missing_argument for an omitted
   one, and maps an explicit none to a null tensor), while the importer and
   Op_bridge both read omission as None. The generated dispatch cannot tell a
   `Tensor?` slot from a `Tensor` one, so this is the ATen decoder's gap rather
   than the importer's -- pinned here rather than worked around. *)
let linear_program ~x_sizes ~w_sizes ~bias =
  let as_t n = jstr {|{"as_tensor":{"name":"%s"}}|} n in
  let captured =
    ("w", w_sizes)
    :: (if bias = `Tensor then [ ("b", [ List.nth w_sizes 0 ]) ] else [])
  in
  let node =
    jstr
      {|{"target":"torch.ops.aten.linear.default","inputs":[{"name":"input","arg":%s,"kind":1},{"name":"weight","arg":%s,"kind":1}%s],"outputs":[%s],"metadata":{}}|}
      (as_t "x") (as_t "w")
      (match bias with
      | `Absent -> ""
      | `None -> {|,{"name":"bias","arg":{"as_none":true},"kind":1}|}
      | `Tensor -> jstr {|,{"name":"bias","arg":%s,"kind":1}|} (as_t "b"))
      (as_t "y")
  in
  jstr
    {|{"graph_module":{"graph":{"inputs":[%s],"outputs":[%s],"nodes":[%s],"tensor_values":{%s},"sym_int_values":{},"sym_bool_values":{},"is_single_tensor_return":true},"signature":{"input_specs":[%s],"output_specs":[{"user_output":{"arg":%s}}]},"module_call_graph":[]},"opset_version":{"aten":15},"range_constraints":{},"schema_version":{"major":8,"minor":5}}|}
    (String.concat "," (as_t "x" :: List.map (fun (n, _) -> as_t n) captured))
    (as_t "y") node
    (String.concat ","
       (jstr {|"x":%s|} (meta_json x_sizes)
       :: List.map (fun (n, sz) -> jstr {|"%s":%s|} n (meta_json sz)) captured))
    (String.concat ","
       (jstr {|{"user_input":{"arg":%s}}|} (as_t "x")
       :: List.map
            (fun (n, _) ->
              jstr {|{"parameter":{"arg":{"name":"%s"},"parameter_name":"%s"}}|}
                n n)
            captured))
    (as_t "y")

let linear_vs_aten label ~x_sizes ~w_sizes ~bias =
  let numel = List.fold_left ( * ) 1 in
  let xs = ramp (numel x_sizes) and ws = ramp (numel w_sizes) in
  let out_features = List.nth w_sizes 0 in
  let bs = ramp out_features in
  Format.printf "%-22s " label;
  match
    Jsont_bytesrw.decode_string PT.ExportedProgram.jsont
      (linear_program ~x_sizes ~w_sizes ~bias)
  with
  | Error e -> Format.printf "fixture did not decode: %s@." e
  | Ok program -> (
      let node = List.hd program.PT.ExportedProgram.graph_module.graph.nodes in
      let aten_env =
        List.fold_left
          (fun m (k, sizes, vals) -> Sm.add k (float_tensor sizes vals) m)
          Sm.empty
          ([ ("x", x_sizes, xs); ("w", w_sizes, ws) ]
          @ if bias = `Tensor then [ ("b", [ out_features ], bs) ] else [])
      in
      match Interp_dispatch.dispatch aten_env node with
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
              let natives =
                [ native_f32 x_sizes xs; native_f32 w_sizes ws ]
                @
                if bias = `Tensor then [ native_f32 [ out_features ] bs ]
                else []
              in
              let inputs, constants =
                List.partition
                  (fun (id, _) ->
                    Graph_ir.input_kind g id = Graph_ir.Input.Input)
                  (List.combine g.Graph_ir.Graph.inputs natives)
              in
              match Eval_direct.run g ~constants ~inputs with
              | Error e ->
                  Format.printf "eval: %a@." Eval_direct.pp_error
                    (Err.Error.kind e)
              | Ok result ->
                  Format.printf "%a@." pp_result
                    (Verify.compare_tensors ~atol:1e-6 ~output:"y"
                       (Sm.find "y" aten_out)
                       (Graph_ir.Tensor_id.Map.find
                          (List.hd g.Graph_ir.Graph.outputs)
                          result)))))

let%expect_test "importer: linear.default matches ATen" =
  linear_vs_aten "bias omitted:" ~x_sizes:[ 4; 3 ] ~w_sizes:[ 5; 3 ]
    ~bias:`Absent;
  linear_vs_aten "bias explicit none:" ~x_sizes:[ 4; 3 ] ~w_sizes:[ 5; 3 ]
    ~bias:`None;
  linear_vs_aten "bias tensor:" ~x_sizes:[ 4; 3 ] ~w_sizes:[ 5; 3 ]
    ~bias:`Tensor;
  (* Leading axes pass through untouched; only the trailing extent reduces. *)
  linear_vs_aten "rank 4 input:" ~x_sizes:[ 2; 7; 4; 3 ] ~w_sizes:[ 5; 3 ]
    ~bias:`Tensor;
  (* out < in as well as out > in, so a transposed permutation is unbuildable in
     one direction and merely wrong in the other. *)
  linear_vs_aten "in > out:" ~x_sizes:[ 4; 6 ] ~w_sizes:[ 2; 6 ] ~bias:`Tensor;
  [%expect
    {|
    bias omitted:          Ok
    bias explicit none:    Ok
    bias tensor:           Ok
    rank 4 input:          Ok
    in > out:              Ok |}]
