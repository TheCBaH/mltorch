(* [Eval_direct.run ~retain]: under [Only], outputs and errors are those of an
   [All] run, and the result holds exactly the outputs and the retained edges.
   That the released payloads really become garbage is native-only and lives in
   test/native_gc. See .ai/ (tensor release). *)

open Graph_ir

(* Deterministic, sign-varying, small: every input and constant a fixture
   declares, from its signature alone. *)
let tensor_of_sig (sg : Tensor_sig.t) =
  let v c =
    float_of_int (((Vec6.offset sg.Tensor_sig.shape c :> int) mod 7) - 3) /. 4.
  in
  match sg.Tensor_sig.fmt with
  | Payload.Fmt Payload.I64 ->
      Tensor.materialize_i64 sg.Tensor_sig.shape (fun c ->
          Int64.of_int ((Vec6.offset sg.Tensor_sig.shape c :> int) mod 2))
  | Payload.Fmt Payload.Bool ->
      Tensor.materialize_bool sg.Tensor_sig.shape (fun c -> v c > 0.)
  | _ -> Tensor.materialize sg.Tensor_sig.shape v

let run ~retain (g : graph) =
  let bound kind =
    List.filter_map
      (fun id ->
        if input_kind g id = kind then
          Some (id, tensor_of_sig (Tensor_id.Map.find id g.Graph.tensors))
        else None)
      g.Graph.inputs
  in
  Eval_direct.run ~retain ~constants:(bound Input.Constant) g
    ~inputs:(bound Input.Input)

let only_empty = Release_schedule.Retain.Only Tensor_id.Set.empty

(* One line per fixture: outputs bit-identical, or the same error. *)
let equivalent name (g : graph) =
  let pp_err ppf e = Eval_direct.pp_error ppf (Err.Error.kind e) in
  match (run ~retain:All g, run ~retain:only_empty g) with
  | Ok all, Ok only ->
      let same id =
        Tensor.equal_bits
          (Tensor_id.Map.find id all)
          (Tensor_id.Map.find id only)
      in
      if List.for_all same g.Graph.outputs then
        Fmt.pr "%s: %d outputs equal@." name (List.length g.Graph.outputs)
      else Fmt.pr "%s: OUTPUTS DIFFER@." name
  | Error a, Error b ->
      let a = Fmt.str "%a" pp_err a and b = Fmt.str "%a" pp_err b in
      if String.equal a b then Fmt.pr "%s: same error: %s@." name a
      else Fmt.pr "%s: ERRORS DIFFER: %s / %s@." name a b
  | Ok _, Error e -> Fmt.pr "%s: ONLY FAILED: %a@." name pp_err e
  | Error e, Ok _ -> Fmt.pr "%s: ALL FAILED: %a@." name pp_err e

let fixtures =
  Graph_fixtures.
    [
      ("bypass_permute_fanout", bypass_permute_fanout);
      ("bypass_permute_mixed_compatibility", bypass_permute_mixed_compatibility);
      ("bypass_permute_output", bypass_permute_output);
      ("bypass_permute_pair", bypass_permute_pair);
      ("bypass_permute_shared", bypass_permute_shared);
      ("bypass_unlocks_sink", bypass_unlocks_sink);
      ("chain", chain);
      ("const_arith", const_arith);
      ("const_permute", const_permute);
      ("const_pointwise", const_pointwise);
      ("const_pool", const_pool);
      ("conv_add", conv_add);
      ("diamond", diamond);
      ("grouped", grouped);
      ("multi_output", multi_output);
      ("permute_identity_chain", permute_identity_chain);
      ("permute_noop", permute_noop);
      ("permute_pair", permute_pair);
      ("permute_partial_cancel", permute_partial_cancel);
      ("permute_sequence", permute_sequence);
      ("permute_shared", permute_shared);
      ("reshape_flatten", reshape_flatten);
      ("reshape_relabel", reshape_relabel);
      ("residual", residual);
      ("reuse_permute_backtrack_candidate", reuse_permute_backtrack_candidate);
      ("reuse_permute_basic", reuse_permute_basic);
      ("reuse_permute_competing_matches", reuse_permute_competing_matches);
      ("reuse_permute_div_order", reuse_permute_div_order);
      ("reuse_permute_missing_alternate", reuse_permute_missing_alternate);
      ("reuse_permute_self_inverse", reuse_permute_self_inverse);
      ("reuse_permute_sub_order", reuse_permute_sub_order);
      ("reuse_permute_wide_fanout", reuse_permute_wide_fanout);
      ("reuse_permute_wrong_alternate", reuse_permute_wrong_alternate);
      ("sink_permute_allowlist", sink_permute_allowlist);
      ("sink_permute_binary", sink_permute_binary);
      ("sink_permute_broadcast", sink_permute_broadcast);
      ("sink_permute_fuse", sink_permute_fuse);
      ("sink_permute_layer_norm", sink_permute_layer_norm);
      ("sink_permute_mean_basic", sink_permute_mean_basic);
      ("sink_permute_mean_cycle", sink_permute_mean_cycle);
      ("sink_permute_mean_not_keepdim", sink_permute_mean_not_keepdim);
      ("sink_permute_mean_shared", sink_permute_mean_shared);
      ("sink_permute_mismatch", sink_permute_mismatch);
      ("sink_permute_output", sink_permute_output);
      ("sink_permute_pad", sink_permute_pad);
      ("sink_permute_sdpa", sink_permute_sdpa);
      ("sink_permute_shared", sink_permute_shared);
      ("sink_permute_slice", sink_permute_slice);
      ("sink_permute_unary", sink_permute_unary);
    ]

let%expect_test "Only empty: every fixture's outputs equal an All run's" =
  List.iter (fun (name, build) -> equivalent name (build ())) fixtures;
  [%expect
    {|
    bypass_permute_fanout: 2 outputs equal
    bypass_permute_mixed_compatibility: 2 outputs equal
    bypass_permute_output: 2 outputs equal
    bypass_permute_pair: 1 outputs equal
    bypass_permute_shared: 2 outputs equal
    bypass_unlocks_sink: 1 outputs equal
    chain: 1 outputs equal
    const_arith: 1 outputs equal
    const_permute: 1 outputs equal
    const_pointwise: 1 outputs equal
    const_pool: 1 outputs equal
    conv_add: 1 outputs equal
    diamond: 1 outputs equal
    grouped: 1 outputs equal
    multi_output: 1 outputs equal
    permute_identity_chain: 1 outputs equal
    permute_noop: 1 outputs equal
    permute_pair: 1 outputs equal
    permute_partial_cancel: 1 outputs equal
    permute_sequence: 1 outputs equal
    permute_shared: 1 outputs equal
    reshape_flatten: 1 outputs equal
    reshape_relabel: 1 outputs equal
    residual: 1 outputs equal
    reuse_permute_backtrack_candidate: 1 outputs equal
    reuse_permute_basic: 1 outputs equal
    reuse_permute_competing_matches: 2 outputs equal
    reuse_permute_div_order: 1 outputs equal
    reuse_permute_missing_alternate: 1 outputs equal
    reuse_permute_self_inverse: 1 outputs equal
    reuse_permute_sub_order: 1 outputs equal
    reuse_permute_wide_fanout: 16 outputs equal
    reuse_permute_wrong_alternate: 1 outputs equal
    sink_permute_allowlist: 1 outputs equal
    sink_permute_binary: 1 outputs equal
    sink_permute_broadcast: 1 outputs equal
    sink_permute_fuse: 1 outputs equal
    sink_permute_layer_norm: 1 outputs equal
    sink_permute_mean_basic: 1 outputs equal
    sink_permute_mean_cycle: 1 outputs equal
    sink_permute_mean_not_keepdim: 1 outputs equal
    sink_permute_mean_shared: 1 outputs equal
    sink_permute_mismatch: 1 outputs equal
    sink_permute_output: 2 outputs equal
    sink_permute_pad: 1 outputs equal
    sink_permute_sdpa: 1 outputs equal
    sink_permute_shared: 1 outputs equal
    sink_permute_slice: 1 outputs equal
    sink_permute_unary: 1 outputs equal |}]

let%expect_test "Only empty: a discarded index output and a dead branch" =
  equivalent "discarded_indices"
    (Graph_builder.build ~name:"discarded_indices"
       ~outputs:(fun o -> [ o ])
       (let open Graph_builder in
        let* x = input ~shape:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:4 ~w:4 ~c:3) () in
        let* _dead = sqrt x in
        let* values, indices =
          max_pool2d_with_indices
            {
              ceil_mode = false;
              kernel = { h = Dim.extent 2; w = Dim.extent 2 };
              stride =
                { h = Op_config.Pos.of_int 2; w = Op_config.Pos.of_int 2 };
              pad =
                { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
            }
            x
        in
        let* () = discard indices in
        relu values)
    |> Graph_fixtures.or_fixture "discarded_indices");
  [%expect {| discarded_indices: 1 outputs equal |}]

let pp_keys = Fmt.(brackets (list ~sep:sp Tensor_id.pp))

let print_keys ~retain g =
  match Err.payload (run ~retain g) with
  | Ok env -> Fmt.pr "%a@." pp_keys (List.map fst (Tensor_id.Map.bindings env))
  | Error e -> Fmt.pr "error: %a@." Eval_direct.pp_error e

(* chain: t0 input, t1..t6 constants, t7 conv, t8 batch_norm, t9 relu. *)
let%expect_test "the result holds exactly the outputs and the retained edges" =
  let g = Graph_fixtures.chain () in
  let conv = List.hd (List.hd g.Graph.nodes).Node.outputs in
  print_keys ~retain:All g;
  print_keys ~retain:only_empty g;
  print_keys ~retain:(Only (Tensor_id.Set.singleton conv)) g;
  (* An input stays when retained, although its last reader has run. *)
  print_keys ~retain:(Only (Tensor_id.Set.singleton (List.hd g.Graph.inputs))) g;
  [%expect
    {|
    [t0 t1 t2 t3 t4 t5 t6 t7 t8 t9]
    [t9]
    [t7 t9]
    [t0 t9] |}]

(* residual with its second relu's output declared I64: the add rejects the
   mixed pair after [a] has already been released. *)
let%expect_test "Only empty: an error mid-graph is the All run's error" =
  let g = Graph_fixtures.residual () in
  let b = List.hd (List.nth g.Graph.nodes 1).Node.outputs in
  equivalent "residual_mixed"
    (Graph_fixtures.with_fmt b (Payload.Fmt Payload.I64) g);
  [%expect
    {| residual_mixed: same error: add: unsupported mixed dtype, a=f32 b=i64 |}]
