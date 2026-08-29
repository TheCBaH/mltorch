(* the PT2 importer's layer_norm.default arm, against ATen. Split from the former native_bridge_test.ml; promote with [dune promote test/native_bridge/layer_norm_importer_test.ml]. *)

open Helpers

(* ---- the PT2 importer's layer_norm.default arm, against ATen ------------- *)

(* Everything the rms_norm block above says about shape-only evidence applies
   here and then some: layer_norm has a second reduction and an affine step with
   an order, so OMITTING the centring (which turns it into rms_norm), dividing
   either reduction by the wrong count, putting eps outside the sqrt, indexing
   the affine operands at the full output coordinate, and applying bias before
   weight are all shape-preserving. ATen is the only oracle that separates them,
   and this is the only fixture in the tree that consults it for this row.

   The input is non-uniform across every axis and the extents are all distinct,
   so a wrong axis choice cannot coincide with the right one; the weight and the
   bias are drawn from DIFFERENT sequences, so swapping them is visible. *)
let ln_program ~target ~x_sizes ~normalized ~weight ~bias ~cudnn =
  let as_t n = jstr {|{"as_tensor":{"name":"%s"}}|} n in
  (* The decomposed target returns [(out, mean, rstd)]; the functional one
     returns a bare tensor. Only the first is compared -- the leading-outputs
     rule -- but the node must DECLARE all three, or neither importer sees the
     shape a real graph has. Both statistics stay dead, which is the state the
     corpus is in and the only one the importer accepts. *)
  let outs =
    if target = "torch.ops.aten.native_layer_norm.default" then
      String.concat "," [ as_t "y"; as_t "mean"; as_t "rstd" ]
    else as_t "y"
  in
  let captured =
    List.filter_map
      (fun (n, present) ->
        if present = `Tensor then Some (n, normalized) else None)
      [ ("w", weight); ("b", bias) ]
  in
  let affine name id = function
    | `Absent -> ""
    | `None -> jstr {|,{"name":"%s","arg":{"as_none":true},"kind":1}|} name
    | `Tensor -> jstr {|,{"name":"%s","arg":%s,"kind":1}|} name (as_t id)
  in
  let node =
    jstr
      {|{"target":"%s","inputs":[{"name":"input","arg":%s,"kind":1},{"name":"normalized_shape","arg":{"as_ints":[%s]},"kind":1}%s%s,{"name":"eps","arg":{"as_float":1e-05},"kind":1}%s],"outputs":[%s],"metadata":{}}|}
      target (as_t "x")
      (String.concat "," (List.map string_of_int normalized))
      (affine "weight" "w" weight)
      (affine "bias" "b" bias)
      (match cudnn with
      | None -> ""
      | Some b ->
          jstr {|,{"name":"cudnn_enable","arg":{"as_bool":%b},"kind":1}|} b)
      outs
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

let ln_vs_aten ?(target = "torch.ops.aten.layer_norm.default") label ~x_sizes
    ~normalized ?(weight = `None) ?(bias = `None) ?cudnn () =
  let numel = List.fold_left ( * ) 1 in
  (* Well away from zero and never repeating within a normalized group. Unlike
     rms_norm, a CONSTANT input is not the degenerate case here (it normalizes
     to 0, not to 1) -- but a constant SPACING is: an arithmetic progression has
     the same normalized values whichever of its axes was reduced when the
     extents happen to agree, so the sequence is deliberately not one. *)
  let scale off n =
    List.init n (fun i -> off +. (float_of_int (i * 7 mod 13) /. 4.))
  in
  let xs = scale 0.5 (numel x_sizes) in
  let ws = scale 1.25 (numel normalized) in
  let bs = scale (-2.5) (numel normalized) in
  Format.printf "%-24s " label;
  match
    Jsont_bytesrw.decode_string PT.ExportedProgram.jsont
      (ln_program ~target ~x_sizes ~normalized ~weight ~bias ~cudnn)
  with
  | Error e -> Format.printf "fixture did not decode: %s@." e
  | Ok program -> (
      let node = List.hd program.PT.ExportedProgram.graph_module.graph.nodes in
      let aten_env =
        List.fold_left
          (fun m (k, sizes, vals) -> Sm.add k (float_tensor sizes vals) m)
          Sm.empty
          ([ ("x", x_sizes, xs) ]
          @ (if weight = `Tensor then [ ("w", normalized, ws) ] else [])
          @ if bias = `Tensor then [ ("b", normalized, bs) ] else [])
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
                @ (if weight = `Tensor then [ native_f32 normalized ws ] else [])
                @ if bias = `Tensor then [ native_f32 normalized bs ] else []
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

let%expect_test "importer: layer_norm.default matches ATen" =
  (* All four affine states, at the default tolerance. *)
  ln_vs_aten "neither:" ~x_sizes:[ 2; 3; 4 ] ~normalized:[ 4 ] ();
  ln_vs_aten "weight:" ~x_sizes:[ 2; 3; 4 ] ~normalized:[ 4 ] ~weight:`Tensor ();
  ln_vs_aten "bias:" ~x_sizes:[ 2; 3; 4 ] ~normalized:[ 4 ] ~bias:`Tensor ();
  ln_vs_aten "both:" ~x_sizes:[ 2; 3; 4 ] ~normalized:[ 4 ] ~weight:`Tensor
    ~bias:`Tensor ();
  (* The OMITTED spelling cannot be compared here, exactly as for rms_norm's
     weight: [Interp_decode.tensor_arg] reports an absent [Tensor?] key as
     missing rather than decoding it to None. Both native importers read
     omission as None (op_bridge's [optional_tensor_present],
     native_interp's), so it is the ATen decoder that cannot express the node,
     not the arm under test. *)
  ln_vs_aten "omitted:" ~x_sizes:[ 2; 3; 4 ] ~normalized:[ 4 ] ~weight:`Absent
    ~bias:`Absent ();
  (* Two axes: BOTH reduction divisors are the product of the normalized
     extents, so a count taken from one axis alone is off by a factor of the
     other in the mean and again in the variance. *)
  ln_vs_aten "two axes:" ~x_sizes:[ 2; 3; 4 ] ~normalized:[ 3; 4 ]
    ~weight:`Tensor ~bias:`Tensor ();
  (* Every axis normalized -- the one case a leading/trailing mix-up CANNOT be
     distinguished by, included so its agreement is not read as evidence. *)
  ln_vs_aten "all axes:" ~x_sizes:[ 2; 3; 4 ] ~normalized:[ 2; 3; 4 ] ();
  (* Rank 4, four distinct extents: the leading three must pass through and both
     affine operands must be read on the trailing one. *)
  ln_vs_aten "rank 4, one axis:" ~x_sizes:[ 2; 3; 4; 5 ] ~normalized:[ 5 ]
    ~weight:`Tensor ~bias:`Tensor ();
  ln_vs_aten "rank 4, three axes:" ~x_sizes:[ 2; 3; 4; 5 ]
    ~normalized:[ 3; 4; 5 ] ~weight:`Tensor ~bias:`Tensor ();
  (* cudnn_enable at both values against the same ATen call: the composite
     discards it, and so must the native arm. *)
  ln_vs_aten "cudnn_enable=true:" ~x_sizes:[ 2; 3; 4 ] ~normalized:[ 4 ]
    ~weight:`Tensor ~bias:`Tensor ~cudnn:true ();
  ln_vs_aten "cudnn_enable=false:" ~x_sizes:[ 2; 3; 4 ] ~normalized:[ 4 ]
    ~weight:`Tensor ~bias:`Tensor ~cudnn:false ();
  (* The DECOMPOSED target, through the same comparison. Its arithmetic is the
     functional one's -- ATen's composite IS a call to it -- so agreement here
     is not new evidence about the formula; what it proves is that the shared
     body reaches the same node from a 3-tuple node with a required eps and no
     cudnn_enable, and that exposing one output verifies against [out] rather
     than silently against [mean]. *)
  let decomposed = "torch.ops.aten.native_layer_norm.default" in
  ln_vs_aten ~target:decomposed "decomposed, neither:" ~x_sizes:[ 2; 3; 4 ]
    ~normalized:[ 4 ] ();
  ln_vs_aten ~target:decomposed "decomposed, both:" ~x_sizes:[ 2; 3; 4 ]
    ~normalized:[ 4 ] ~weight:`Tensor ~bias:`Tensor ();
  (* The corpus configuration: rank 3, one normalized axis, both affine
     operands present -- all 148 nodes across the four ViT models. *)
  ln_vs_aten ~target:decomposed "decomposed, corpus:" ~x_sizes:[ 1; 50; 768 ]
    ~normalized:[ 768 ] ~weight:`Tensor ~bias:`Tensor ();
  ln_vs_aten ~target:decomposed "decomposed, two axes:" ~x_sizes:[ 2; 3; 4 ]
    ~normalized:[ 3; 4 ] ~weight:`Tensor ~bias:`Tensor ();
  [%expect
    {|
    neither:                 Ok
    weight:                  Ok
    bias:                    Ok
    both:                    Ok
    omitted:                 aten: missing required argument "weight"
    two axes:                Ok
    all axes:                Ok
    rank 4, one axis:        Ok
    rank 4, three axes:      Ok
    cudnn_enable=true:       Ok
    cudnn_enable=false:      Ok
    decomposed, neither:     Ok
    decomposed, both:        Ok
    decomposed, corpus:      Ok
    decomposed, two axes:    Ok |}]
