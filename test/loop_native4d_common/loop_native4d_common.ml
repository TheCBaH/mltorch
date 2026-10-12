open Native4d
open Native4d_test

type case = {
  name : string;
  graph : Graph.graph;
  inputs : (Tensor_id.t * Tensor.packed) list;
  constants : (Tensor_id.t * Tensor.packed) list;
}

let per_op () =
  List.map
    (fun (name, graph, inputs, constants) -> { name; graph; inputs; constants })
    (Fixtures4.per_op ())

(* Deterministic, sign-varying, small, from the signature alone. Strictly
   positive where a normalisation divides by it would matter is not needed: the
   comparison is bitwise against the same evaluator's own answer. *)
let tensor_of_sig (sg : Tensor_sig.t) =
  match sg.Tensor_sig.fmt with
  | Payload.Fmt Payload.I64 ->
      Tensor.materialize_i64 sg.Tensor_sig.shape (fun c ->
          Int64.of_int ((Vec6.offset sg.Tensor_sig.shape c :> int) mod 2))
  | _ ->
      Tensor.materialize sg.Tensor_sig.shape (fun c ->
          float_of_int (((Vec6.offset sg.Tensor_sig.shape c :> int) mod 7) - 3)
          /. 4.)

let lowered_case name native =
  let result = ref None in
  let (_ : string) =
    Lower_fixtures.described native ~render:(fun dst ->
        let bound kind =
          List.filter_map
            (fun id ->
              if Graph.input_kind dst id = kind then
                Some
                  ( id,
                    tensor_of_sig
                      (Tensor_id.Map.find id dst.Graph_common.Graph.tensors) )
              else None)
            dst.Graph_common.Graph.inputs
        in
        result :=
          Some
            {
              name;
              graph = dst;
              inputs = bound Graph_ir.Input.Input;
              constants = bound Graph_ir.Input.Constant;
            };
        "")
  in
  match !result with
  | Some c -> c
  | None -> failwith (Fmt.str "%s: outside the Native4D dialect" name)

let lowered () =
  List.map
    (fun (name, g) -> lowered_case name g)
    Fixtures.
      [
        ("adaptive_maxpool_indices_live", adaptive_maxpool_indices_live ());
        ("expand", expand ());
        ("group_norm_tiny", group_norm_tiny ());
        ("layer_norm_tiny", layer_norm_tiny ());
        ("linear_layer", linear_layer ());
        ("lstm_states_live", lstm_states_live ());
        ("maxpool_indices_live", maxpool_indices_live ());
        ("sdpa_no_mask", sdpa 1 ());
        ("softmax_w", softmax_over Axis.W ());
      ]

(* A convolution whose weight is a captured six-dimensional tensor behind a
   Const-SSA permutation: after the canonical fold the Native4D graph's weight
   is a DEFERRED export, with no payload until it is resolved from the capture.
   The case's [constants] are what a driver must hand a backend: the lowering's
   own payloads with that export materialized over them, as
   [Loop_native4d_model] does. The check below fails the fixture if the export
   ever stops being deferred, since the case would then no longer exercise the
   materialization. *)
let deferred_constant () =
  let source =
    Lower_fixtures.build "deferred_weight"
      (let open Graph_builder in
       let* x = input ~shape:(Fixtures.nhwc ~n:1 ~h:3 ~w:3 ~c:2) () in
       let* w = constant ~shape:(Fixtures.s 1 3 1 2 2 2) () in
       let* wp =
         permute
           (Fixtures.perm_of
              [
                (Axis.N, Axis.T);
                (Axis.T, Axis.N);
                (Axis.H, Axis.W);
                (Axis.W, Axis.H);
              ])
           w
       in
       let base = Fixtures.conv_params ~in_channels:2 ~groups:1 in
       let params =
         {
           base with
           h = Fixtures.conv_axis ~kernel:2;
           w = Fixtures.conv_axis ~kernel:2;
         }
       in
       conv2d params ~x ~weight:wp ())
  in
  let weight =
    Tensor_id.Map.find (Tensor_id.of_int 1) source.Graph_ir.Graph.tensors
  in
  let store =
    Constant_store.bind_captured Constant_store.empty ~tensor:weight
      (Const_ssa.Capture.of_string "layer.weight")
    |> Err.or_raise ~pp_error:Constant_store.pp_error
  in
  let fail fmt = Fmt.kstr failwith fmt in
  match Rewrite.origin ~constant_store:store source with
  | Error e -> fail "origin: %a" Rewrite.pp_error (Err.Error.kind e)
  | Ok (Rewrite.Origin state) -> (
      match Pass.run_all state [ Fold_const.pass ] with
      | Error e -> fail "fold: %a" Pass.pp_error (Err.Error.kind e)
      | Ok (Rewrite.Step (folded, _)) -> (
          match
            Lower.convert
              ~constant_store:(Rewrite.constant_store folded)
              (Rewrite.snapshot folded)
          with
          | Error e -> fail "convert: %a" Error.pp (Err.Error.kind e)
          | Ok (Lower.Pack r) ->
              let graph = Lower.graph r in
              let missing =
                List.filter
                  (fun id ->
                    Graph.input_kind graph id = Graph_ir.Input.Constant
                    && not (Tensor_id.Map.mem id r.Lower.constants))
                  graph.Graph_common.Graph.inputs
              in
              if missing = [] then
                fail "deferred_constant: no deferred export left to resolve";
              let resolver capture =
                ignore capture;
                Err.return
                  (tensor_of_sig
                     (Tensor_id.Map.find (Tensor_id.of_int 1)
                        source.Graph_ir.Graph.tensors))
              in
              let store, _ =
                Const_ssa_materialize.materialize resolver
                  r.Lower.constant_store
                |> Err.or_raise ~pp_error:Const_ssa_materialize.pp_error
              in
              let constants =
                Tensor_id.Map.union
                  (fun _ _ planned -> Some planned)
                  r.Lower.constants
                  (Constant_store.materialized store)
              in
              {
                name = "deferred_constant";
                graph;
                inputs =
                  List.filter_map
                    (fun id ->
                      if Graph.input_kind graph id = Graph_ir.Input.Input then
                        Some
                          ( id,
                            tensor_of_sig
                              (Tensor_id.Map.find id
                                 graph.Graph_common.Graph.tensors) )
                      else None)
                    graph.Graph_common.Graph.inputs;
                constants = Tensor_id.Map.bindings constants;
              }))

let all () = per_op () @ lowered () @ [ deferred_constant () ]

let reference c =
  Eval_direct4.run c.graph ~constants:c.constants ~inputs:c.inputs
  |> Err.or_raise ~pp_error:Eval_direct4.pp_error
