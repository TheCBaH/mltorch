(* the PT2 importer's conv2d.default arm, against ATen. Split from the former native_bridge_test.ml; promote with [dune promote test/native_bridge/conv_test.ml]. *)

open Helpers

(* ---- the PT2 importer's conv2d.default arm, against ATen ---------------- *)

(* [Native_interp] is the OTHER importer: it reads SERIALIZED metadata where
   [Op_bridge] reads live tensors. Nothing above compares the two. The generated
   walks verify the bridge against ATen and test/native_interp/conv_test.ml pins
   the importer's graph structure, but neither says the importer computes ATen's
   answer -- and since no downloadable model serialises `conv2d.default` (every
   one is exported post-decomposition, carrying `convolution.default`), no cram
   over real weights ever will either.

   ONE serialized node feeds both paths: the ATen side takes it out of the
   DECODED graph rather than rebuilding it, so the two cannot come to disagree
   about what was asked. What is left to differ is the only thing under test --
   the weight relayout, the H/W pairing, and the channel arithmetic. *)

(* [padding] is the whole serialized argument, because the two overloads spell it
   differently: `conv2d.default` takes an int pair and `conv2d.padding` a mode
   string. Everything else about the node -- and everything the comparison is
   actually about -- is identical, so they share one builder. *)
let conv_program ~target ~x_sizes ~w_sizes ~b_size ~stride ~padding ~dilation
    ~groups =
  let captured = [ ("w", w_sizes); ("b", [ b_size ]) ] in
  let as_t n = jstr {|{"as_tensor":{"name":"%s"}}|} n in
  let node =
    jstr
      {|{"target":"%s","inputs":[{"name":"input","arg":%s,"kind":1},{"name":"weight","arg":%s,"kind":1},{"name":"bias","arg":%s,"kind":1},{"name":"stride","arg":{"as_ints":[%d,%d]},"kind":1},{"name":"padding","arg":%s,"kind":1},{"name":"dilation","arg":{"as_ints":[%d,%d]},"kind":1},{"name":"groups","arg":{"as_int":%d},"kind":1}],"outputs":[%s],"metadata":{}}|}
      target (as_t "x") (as_t "w") (as_t "b") (fst stride) (snd stride) padding
      (fst dilation) (snd dilation) groups (as_t "y")
  in
  jstr
    {|{"graph_module":{"graph":{"inputs":[%s],"outputs":[%s],"nodes":[%s],"tensor_values":{%s},"sym_int_values":{},"sym_bool_values":{},"is_single_tensor_return":true},"signature":{"input_specs":[%s],"output_specs":[{"user_output":{"arg":%s}}]},"module_call_graph":[]},"opset_version":{"aten":15},"range_constraints":{},"schema_version":{"major":8,"minor":5}}|}
    (String.concat "," (as_t "x" :: List.map (fun (n, _) -> as_t n) captured))
    (as_t "y") node
    (String.concat ","
       (jstr {|"x":%s|} (meta_json x_sizes)
       :: List.map (fun (n, s) -> jstr {|"%s":%s|} n (meta_json s)) captured))
    (String.concat ","
       (jstr {|{"user_input":{"arg":%s}}|} (as_t "x")
       :: List.map
            (fun (n, _) ->
              jstr {|{"parameter":{"arg":{"name":"%s"},"parameter_name":"%s"}}|}
                n n)
            captured))
    (as_t "y")

let importer_vs_aten ?(target = "torch.ops.aten.conv2d.default") label ~x_sizes
    ~w_sizes ~stride ~padding ~dilation ~groups =
  let numel = List.fold_left ( * ) 1 in
  let b_size = List.nth w_sizes 0 in
  let xs = ramp (numel x_sizes) and ws = ramp (numel w_sizes) in
  let bs = ramp b_size in
  let json =
    conv_program ~target ~x_sizes ~w_sizes ~b_size ~stride ~padding ~dilation
      ~groups
  in
  Format.printf "%-22s " label;
  match Jsont_bytesrw.decode_string PT.ExportedProgram.jsont json with
  | Error e -> Format.printf "fixture did not decode: %s@." e
  | Ok program -> (
      let node = List.hd program.PT.ExportedProgram.graph_module.graph.nodes in
      let aten_env =
        List.fold_left
          (fun m (k, sizes, vals) -> Sm.add k (float_tensor sizes vals) m)
          Sm.empty
          [ ("x", x_sizes, xs); ("w", w_sizes, ws); ("b", [ b_size ], bs) ]
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
              (* Captured tensors are graph inputs of kind [Constant], and
                 [Eval_direct] takes those through [~constants] -- the split the
                 importer's own [run] makes when it resolves payloads. *)
              let inputs, constants =
                List.partition
                  (fun (id, _) ->
                    Graph_ir.input_kind g id = Graph_ir.Input.Input)
                  (List.combine g.Graph_ir.Graph.inputs
                     [
                       native_f32 x_sizes xs;
                       native_f32 w_sizes ws;
                       native_f32 [ b_size ] bs;
                     ])
              in
              match Eval_direct.run g ~constants ~inputs with
              | Error e ->
                  Format.printf "eval: %a@." Eval_direct.pp_error
                    (Err.Error.kind e)
              | Ok result ->
                  let native_y =
                    Graph_ir.Tensor_id.Map.find
                      (List.hd g.Graph_ir.Graph.outputs)
                      result
                  in
                  Format.printf "%a@." pp_result
                    (Verify.compare_tensors ~atol:1e-6 ~output:"y"
                       (Sm.find "y" aten_out) native_y))))

let ints_pad (h, w) = jstr {|{"as_ints":[%d,%d]}|} h w
let mode_pad m = jstr {|{"as_string":"%s"}|} m

let%expect_test "importer: conv2d.default matches ATen" =
  importer_vs_aten "defaults:" ~x_sizes:[ 1; 2; 5; 5 ] ~w_sizes:[ 3; 2; 3; 3 ]
    ~stride:(1, 1)
    ~padding:(ints_pad (0, 0))
    ~dilation:(1, 1) ~groups:1;
  (* Asymmetric everything: a transposed H/W pair survives every square
     configuration and nothing else here would catch it. *)
  importer_vs_aten "asymmetric:" ~x_sizes:[ 1; 2; 7; 5 ] ~w_sizes:[ 3; 2; 3; 2 ]
    ~stride:(2, 1)
    ~padding:(ints_pad (1, 0))
    ~dilation:(1, 2) ~groups:1;
  (* Depthwise: [in_channels] is the weight's per-group extent TIMES groups, so
     dropping the factor builds a conv over one channel instead of four. *)
  importer_vs_aten "depthwise:" ~x_sizes:[ 1; 4; 5; 5 ] ~w_sizes:[ 4; 1; 3; 3 ]
    ~stride:(1, 1)
    ~padding:(ints_pad (1, 1))
    ~dilation:(1, 1) ~groups:4;
  (* General grouping, which Native holds and only Native4D refuses. *)
  importer_vs_aten "groups=2:" ~x_sizes:[ 1; 4; 5; 5 ] ~w_sizes:[ 6; 2; 3; 3 ]
    ~stride:(1, 1)
    ~padding:(ints_pad (1, 1))
    ~dilation:(1, 1) ~groups:2;
  [%expect
    {|
    defaults:              Ok
    asymmetric:            Ok
    depthwise:             Ok
    groups=2:              Ok |}]

(* "same" is where an ATen oracle earns its place. The importer does not resolve
   the mode -- [Conv2d_padding.same_padding] does -- so what these check is that
   the mode SURVIVES the importer intact and that the resolution downstream of
   it agrees with ATen for the same weight.

   NOT COVERED HERE: an even kernel, where the total padding is odd and the
   split is genuinely asymmetric. ATen reaches [constant_pad_nd] for that case
   and this repository's minimal static-dispatch build does not carry it, so the
   comparison cannot be made -- it aborts the process rather than returning an
   error. [walk_meta.ml]'s conv2d_padding axes are all-odd kernels for the same
   reason. The asymmetric split is pinned against hand-computed values in
   test/native instead, where the rule actually lives. *)
let%expect_test "importer: conv2d.padding matches ATen" =
  let pad ?(groups = 1) label ~mode ~x_sizes ~w_sizes ~dilation =
    importer_vs_aten ~target:"torch.ops.aten.conv2d.padding" label ~x_sizes
      ~w_sizes ~stride:(1, 1) ~padding:(mode_pad mode) ~dilation ~groups
  in
  pad "valid, 3x3:" ~mode:"valid" ~x_sizes:[ 1; 2; 5; 5 ]
    ~w_sizes:[ 3; 2; 3; 3 ] ~dilation:(1, 1);
  pad "same, 3x3:" ~mode:"same" ~x_sizes:[ 1; 2; 5; 5 ] ~w_sizes:[ 3; 2; 3; 3 ]
    ~dilation:(1, 1);
  (* Asymmetric spatial extents, so a transposed H/W pair is visible. *)
  pad "same, 5x3:" ~mode:"same" ~x_sizes:[ 1; 2; 7; 5 ] ~w_sizes:[ 3; 2; 5; 3 ]
    ~dilation:(1, 1);
  (* Dilated "same": the total is [dilation * (kernel - 1)], not [kernel - 1],
     so dropping the dilation factor pads too little and moves the shape. *)
  pad "same, dilated:" ~mode:"same" ~x_sizes:[ 1; 2; 9; 9 ]
    ~w_sizes:[ 3; 2; 3; 3 ] ~dilation:(2, 3);
  (* Depthwise, to pin that the mode and the grouping compose. *)
  pad ~groups:4 "same, depthwise:" ~mode:"same" ~x_sizes:[ 1; 4; 5; 5 ]
    ~w_sizes:[ 4; 1; 3; 3 ] ~dilation:(1, 1);
  [%expect
    {|
    valid, 3x3:            Ok
    same, 3x3:             Ok
    same, 5x3:             Ok
    same, dilated:         Ok
    same, depthwise:       Ok |}]
