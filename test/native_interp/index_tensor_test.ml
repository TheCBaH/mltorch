(* `index.Tensor(Tensor self, Tensor?[] indices) -> Tensor`, restricted to
   the evidenced shape family (`.ai/index_tensor_design.md`): [indices] has
   at most [self]'s ATen rank many entries, exactly one live entry of ATen
   rank at least 1, every other listed position [None]. Metadata-only
   decode (unlike [Op_bridge]'s dispatch, tested separately in
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

(* [x] is rank 2 ([2;3]) in most fixtures below, so a 2-entry [indices]
   matches its rank exactly; a few name their own [x_sizes]/list length to
   exercise the shorter-list and rank-3 cases. *)

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

(* Real ATen implicitly full-slices any dims [indices] doesn't mention, so a
   list SHORTER than [self]'s rank is not itself a fault -- exactly
   `mvitv2_tiny`/`maxxvitv2_nano_rw_256`'s own encoding
   (`.ai/index_tensor_design.md`): a length-1 [indices] against a rank-2
   [self]. *)
let%expect_test "index.Tensor: accepts an indices list shorter than self's rank"
    =
  let prog =
    program ~x_sizes:[ 2; 3 ] ~params:[ "idx_raw" ]
      ~extra_tensor_values:[ ("idx_raw", tensor_meta_dtype long [ 2 ]) ]
      ~nodes:[ index_node ~self:"x" ~indices:[ `T "idx_raw" ] ~out:"y" ]
      ~graph_outputs:[ as_tensor "y" ]
      ()
  in
  show "short list:" prog;
  [%expect {| short list:                lowered, nodes=1 |}]

let%expect_test "index.Tensor: rejects an indices list longer than self's rank"
    =
  let prog =
    program ~x_sizes:[ 2; 3 ] ~params:[ "idx_raw" ]
      ~extra_tensor_values:[ ("idx_raw", tensor_meta_dtype long [ 2 ]) ]
      ~nodes:
        [
          index_node ~self:"x" ~indices:[ `T "idx_raw"; `None; `None ] ~out:"y";
        ]
      ~graph_outputs:[ as_tensor "y" ]
      ()
  in
  show "long list:" prog;
  [%expect
    {| long list:                 malformed PT2 graph: index.Tensor: indices has 3 entries, more than self's rank 2 |}]

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

(* `mvitv2_tiny`/`maxxvitv2_nano_rw_256`'s own shape family: a live entry of
   ATen rank 2, accepted since [axis] (dim 0) is [self]'s own outermost dim
   here, so there is nothing of [self]'s own to lose to index's extra axis. *)
let%expect_test "index.Tensor: accepts a live entry of ATen rank 2" =
  let prog =
    program ~x_sizes:[ 2; 3 ] ~params:[ "idx_raw" ]
      ~extra_tensor_values:[ ("idx_raw", tensor_meta_dtype long [ 2; 1 ]) ]
      ~nodes:[ index_node ~self:"x" ~indices:[ `T "idx_raw"; `None ] ~out:"y" ]
      ~graph_outputs:[ as_tensor "y" ]
      ()
  in
  show "rank 2 live entry:" prog;
  [%expect {| rank 2 live entry:         lowered, nodes=1 |}]

(* [Self_collision]: a rank-3 [self] with real content on every axis,
   gathered at a MIDDLE position (dim 1) with a rank-2 live entry -- index's
   extra axis would have to borrow [self]'s own leading axis, which real
   ATen never discards. Reachable through the importer itself, unlike
   [Rank_overflow] (`test/native/graph_direct_index_tensor_test.ml`'s own
   proof), since a real PT2 graph can name any dim position, not only
   self's outermost. *)
let%expect_test
    "index.Tensor: rejects a rank-2 entry that would overwrite self's own axis"
    =
  let prog =
    program ~x_sizes:[ 2; 3; 4 ] ~params:[ "idx_raw" ]
      ~extra_tensor_values:[ ("idx_raw", tensor_meta_dtype long [ 2; 1 ]) ]
      ~nodes:
        [
          index_node ~self:"x" ~indices:[ `None; `T "idx_raw"; `None ] ~out:"y";
        ]
      ~graph_outputs:[ as_tensor "y" ]
      ()
  in
  show "middle-position rank-2 entry:" prog;
  [%expect
    {| middle-position rank-2 entry: index.Tensor: a multi-axis index at axis W would overwrite self's own axis H, which must have extent 1 (got 2) |}]

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
