(* `index.Tensor(Tensor self, Tensor?[] indices) -> Tensor`, restricted to
   the one evidenced shape family (`.ai/index_tensor_design.md`): [indices]
   has exactly one live entry, of ATen rank 1, every other position [None].
   Metadata-only decode (unlike [Op_bridge]'s dispatch, tested separately in
   test/native_bridge/index_tensor_dispatch_test.ml), including the
   trace-past-Clone rule (round 6) for a Long constant behind a plain,
   format-preserving `clone.default` -- CSATv2's own real occurrence. *)

open Programs

let long = 5 (* ScalarType.LONG, per schema.yaml *)

let tensor_meta_dtype dtype sizes =
  jstr
    {|{"dtype":%d,"sizes":[%s],"requires_grad":false,"device":{"type":"cpu"},"strides":[{"as_int":1}],"storage_offset":{"as_int":0},"layout":7}|}
    dtype
    (String.concat "," (List.map (fun i -> jstr {|{"as_int":%d}|} i) sizes))

let as_optional_tensors entries =
  jstr {|{"as_optional_tensors":[%s]}|}
    (String.concat ","
       (List.map
          (function
            | `T n -> jstr {|{"as_tensor":{"name":"%s"}}|} n
            | `None -> {|{"as_none":false}|})
          entries))

let clone_node ~self ~out =
  jstr
    {|{"target":"torch.ops.aten.clone.default","inputs":[{"name":"self","arg":%s,"kind":1}],"outputs":[%s],"metadata":{}}|}
    (as_tensor self) (as_tensor out)

let index_node ~self ~indices ~out =
  jstr
    {|{"target":"torch.ops.aten.index.Tensor","inputs":[{"name":"self","arg":%s,"kind":1},{"name":"indices","arg":%s,"kind":1}],"outputs":[%s],"metadata":{}}|}
    (as_tensor self)
    (as_optional_tensors indices)
    (as_tensor out)

(* [x] is rank 2 ([2;3]) throughout, so [indices] always has 2 entries. *)

let%expect_test
    "index.Tensor: the full Long constant -> clone.default -> index.Tensor \
     fixture binds the ORIGINAL constant, not Clone's output" =
  let prog =
    program ~x_sizes:[ 2; 3 ] ~params:[ "idx_raw" ]
      ~extra_tensor_values:
        [
          ("idx_raw", tensor_meta_dtype long [ 2 ]);
          ("clone_1", tensor_meta_dtype long [ 2 ]);
        ]
      ~nodes:
        [
          clone_node ~self:"idx_raw" ~out:"clone_1";
          index_node ~self:"x" ~indices:[ `T "clone_1"; `None ] ~out:"y";
        ]
      ~graph_outputs:[ as_tensor "y" ]
      ()
  in
  match
    Jsont_bytesrw.decode_string Pytorch_types.ExportedProgram.jsont prog
  with
  | Error e -> failwith ("fixture did not decode: " ^ e)
  | Ok exported ->
      (match Native_interp.lower exported with
      | Error e ->
          Format.printf "error: %a@." Native_interp.pp_error (Err.Error.kind e)
      | Ok (l : Pt2_native_graph.t) -> (
          let index_tensor_node =
            List.find_opt
              (fun (n : Graph_ir.node) ->
                match n.Graph_ir.Node.op with
                | Graph_ir.Index_tensor _ -> true
                | _ -> false)
              l.Pt2_native_graph.graph.Graph_ir.Graph.nodes
          in
          match index_tensor_node with
          | None -> print_string "no Index_tensor node built"
          | Some { Graph_ir.Node.op = Graph_ir.Index_tensor it; _ } -> (
              let index_id = it.Index_tensor.Index_tensor.index in
              match
                Graph_ir.Tensor_id.Map.find_opt index_id
                  l.Pt2_native_graph.tensor_origins
              with
              | Some (Pt2_native_graph.Source { ssa_name; _ }) ->
                  Format.printf "index operand's ssa name: %s@." ssa_name
              | Some Pt2_native_graph.Derived ->
                  print_string "index operand has no PT2 provenance"
              | None -> print_string "index operand id not found")
          | Some _ -> assert false));
      [%expect {| index operand's ssa name: idx_raw |}]

let%expect_test "index.Tensor: rejects a wrong-length indices list" =
  let prog =
    program ~x_sizes:[ 2; 3 ] ~params:[ "idx_raw" ]
      ~extra_tensor_values:[ ("idx_raw", tensor_meta_dtype long [ 2 ]) ]
      ~nodes:[ index_node ~self:"x" ~indices:[ `T "idx_raw" ] ~out:"y" ]
      ~graph_outputs:[ as_tensor "y" ]
      ()
  in
  show "wrong length:" prog;
  [%expect
    {| wrong length:              malformed PT2 graph: index.Tensor: indices has 1 entries, expected 2 (self's rank) |}]

let%expect_test
    "index.Tensor: rejects a live entry at a non-last position and two live \
     entries" =
  let prog1 =
    program ~x_sizes:[ 2; 3 ] ~params:[ "idx0"; "idx1" ]
      ~extra_tensor_values:
        [
          ("idx0", tensor_meta_dtype long [ 2 ]);
          ("idx1", tensor_meta_dtype long [ 3 ]);
        ]
      ~nodes:[ index_node ~self:"x" ~indices:[ `T "idx0"; `None ] ~out:"y" ]
      ~graph_outputs:[ as_tensor "y" ]
      ()
  in
  show "non-last position (0):" prog1;
  let prog2 =
    program ~x_sizes:[ 2; 3 ] ~params:[ "idx0"; "idx1" ]
      ~extra_tensor_values:
        [
          ("idx0", tensor_meta_dtype long [ 2 ]);
          ("idx1", tensor_meta_dtype long [ 3 ]);
        ]
      ~nodes:[ index_node ~self:"x" ~indices:[ `T "idx0"; `T "idx1" ] ~out:"y" ]
      ~graph_outputs:[ as_tensor "y" ]
      ()
  in
  show "two live entries:" prog2;
  [%expect
    {|
    non-last position (0):     lowered, nodes=1
    two live entries:          malformed PT2 graph: index.Tensor: indices has more than one live entry, at positions 0, 1
    |}]

let%expect_test "index.Tensor: rejects a boolean-mask entry (wrong dtype)" =
  let bool_ =
    12
    (* ScalarType.BOOL *)
  in
  let prog =
    program ~x_sizes:[ 2; 3 ] ~params:[ "mask" ]
      ~extra_tensor_values:[ ("mask", tensor_meta_dtype bool_ [ 2 ]) ]
      ~nodes:[ index_node ~self:"x" ~indices:[ `T "mask"; `None ] ~out:"y" ]
      ~graph_outputs:[ as_tensor "y" ]
      ()
  in
  show "boolean mask:" prog;
  [%expect
    {| boolean mask:              malformed PT2 graph: index.Tensor: indices[0] must be Long, got BOOL |}]

let%expect_test "index.Tensor: rejects a live entry of ATen rank 2" =
  let prog =
    program ~x_sizes:[ 2; 3 ] ~params:[ "idx_raw" ]
      ~extra_tensor_values:[ ("idx_raw", tensor_meta_dtype long [ 2; 1 ]) ]
      ~nodes:[ index_node ~self:"x" ~indices:[ `T "idx_raw"; `None ] ~out:"y" ]
      ~graph_outputs:[ as_tensor "y" ]
      ()
  in
  show "rank 2 live entry:" prog;
  [%expect
    {| rank 2 live entry:         malformed PT2 graph: index.Tensor: indices[0] must be rank 1, got rank 2 |}]

let%expect_test "index.Tensor: rejects an all-None indices list (no live entry)"
    =
  let prog =
    program ~x_sizes:[ 2; 3 ]
      ~nodes:[ index_node ~self:"x" ~indices:[ `None; `None ] ~out:"y" ]
      ~graph_outputs:[ as_tensor "y" ]
      ()
  in
  show "all None:" prog;
  [%expect
    {| all None:                  malformed PT2 graph: index.Tensor: indices has no live (non-None) entry |}]
