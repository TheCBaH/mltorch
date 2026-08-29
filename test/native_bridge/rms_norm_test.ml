(* the PT2 importer's rms_norm.default arm, against ATen. Split from the former native_bridge_test.ml; promote with [dune promote test/native_bridge/rms_norm_test.ml]. *)

open Helpers

(* ---- the PT2 importer's rms_norm.default arm, against ATen --------------- *)

(* This is the row where shape-only evidence is worth least. Normalizing over
   the LEADING axes instead of the trailing ones, dividing by the wrong count,
   adding eps outside the sqrt instead of inside, or indexing the weight on the
   wrong axis all preserve the output shape exactly. Only a value comparison
   separates them, and ATen is the only oracle for it.

   The input is NON-UNIFORM across every axis and the extents are all distinct,
   so a wrong axis choice cannot coincide with the right one. *)
let rms_program ~x_sizes ~normalized ~weight =
  let as_t n = jstr {|{"as_tensor":{"name":"%s"}}|} n in
  let captured = if weight = `Tensor then [ ("w", normalized) ] else [] in
  let node =
    jstr
      {|{"target":"torch.ops.aten.rms_norm.default","inputs":[{"name":"input","arg":%s,"kind":1},{"name":"normalized_shape","arg":{"as_ints":[%s]},"kind":1}%s],"outputs":[%s],"metadata":{}}|}
      (as_t "x")
      (String.concat "," (List.map string_of_int normalized))
      (match weight with
      | `Absent -> ""
      | `None -> {|,{"name":"weight","arg":{"as_none":true},"kind":1}|}
      | `Tensor -> jstr {|,{"name":"weight","arg":%s,"kind":1}|} (as_t "w"))
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

let rms_vs_aten label ~x_sizes ~normalized ~weight =
  let numel = List.fold_left ( * ) 1 in
  (* Well away from zero, and never repeating within a normalized group: a
     constant input has RMS equal to itself and normalizes to 1 whatever axes
     were chosen. *)
  let scale n =
    List.init n (fun i -> 0.5 +. (float_of_int (i * 7 mod 13) /. 4.))
  in
  let xs = scale (numel x_sizes) and ws = scale (numel normalized) in
  Format.printf "%-22s " label;
  match
    Jsont_bytesrw.decode_string PT.ExportedProgram.jsont
      (rms_program ~x_sizes ~normalized ~weight)
  with
  | Error e -> Format.printf "fixture did not decode: %s@." e
  | Ok program -> (
      let node = List.hd program.PT.ExportedProgram.graph_module.graph.nodes in
      let aten_env =
        List.fold_left
          (fun m (k, sizes, vals) -> Sm.add k (float_tensor sizes vals) m)
          Sm.empty
          ([ ("x", x_sizes, xs) ]
          @ if weight = `Tensor then [ ("w", normalized, ws) ] else [])
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
                [ native_f32 x_sizes xs ]
                @ if weight = `Tensor then [ native_f32 normalized ws ] else []
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
                    (Verify.compare_tensors ~atol:1e-5 ~output:"y"
                       (Sm.find "y" aten_out)
                       (Graph_ir.Tensor_id.Map.find
                          (List.hd g.Graph_ir.Graph.outputs)
                          result)))))

let%expect_test "importer: rms_norm.default matches ATen" =
  rms_vs_aten "one axis, weight:" ~x_sizes:[ 2; 3; 4 ] ~normalized:[ 4 ]
    ~weight:`Tensor;
  (* No weight, spelled as an explicit none. The OMITTED spelling cannot be
     compared: Interp_decode.tensor_arg reports it as missing, the same gap the
     linear.default block above pins for [bias]. Both importers read omission as
     None, so it is the ATen decoder that cannot express this node. *)
  rms_vs_aten "one axis, none:" ~x_sizes:[ 2; 3; 4 ] ~normalized:[ 4 ]
    ~weight:`None;
  rms_vs_aten "one axis, omitted:" ~x_sizes:[ 2; 3; 4 ] ~normalized:[ 4 ]
    ~weight:`Absent;
  (* Two axes: the divisor is the PRODUCT of the normalized extents, so a count
     taken from one axis alone is off by a factor of the other. *)
  rms_vs_aten "two axes:" ~x_sizes:[ 2; 3; 4 ] ~normalized:[ 3; 4 ]
    ~weight:`Tensor;
  (* Every axis normalized -- the one case a leading/trailing mix-up CANNOT be
     distinguished by, included so that its agreement is not read as evidence. *)
  rms_vs_aten "all axes:" ~x_sizes:[ 2; 3; 4 ] ~normalized:[ 2; 3; 4 ]
    ~weight:`None;
  (* Rank 4 with four distinct extents, one normalized axis: the leading three
     must pass through and the weight must be read on the trailing one. *)
  rms_vs_aten "rank 4, one axis:" ~x_sizes:[ 2; 3; 4; 5 ] ~normalized:[ 5 ]
    ~weight:`Tensor;
  (* Rank 4, two normalized axes, so the pass-through and the multi-axis
     reduction are exercised together. *)
  rms_vs_aten "rank 4, two axes:" ~x_sizes:[ 2; 3; 4; 5 ] ~normalized:[ 4; 5 ]
    ~weight:`Tensor;
  [%expect
    {|
    one axis, weight:      Ok
    one axis, none:        Ok
    one axis, omitted:     aten: missing required argument "weight"
    two axes:              Ok
    all axes:              Ok
    rank 4, one axis:      Ok
    rank 4, two axes:      Ok |}]
