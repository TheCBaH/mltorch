open Graph_ir
open Loop_ir

(* The whole-model JavaScript bundle built through the structured SSA backend and
   run in-process under node, against the per-node reference evaluator. *)

let tensor_of ~salt (sg : Tensor_sig.t) =
  let v c =
    float_of_int
      ((((Vec6.offset sg.Tensor_sig.shape c :> int) + salt) mod 7) - 3)
    /. 4.
  in
  Tensor.materialize sg.Tensor_sig.shape v

(* bits, so a NaN equals itself and a signed zero is a different value *)
let values t =
  let (Tensor.Tensor tt) = t in
  let acc = ref [] in
  Vec6.iter tt.Tensor.shape (fun c ->
      acc := Int64.bits_of_float (Tensor.read_at t (Vec6.get c)) :: !acc);
  List.rev !acc

(* strictly positive parameters (a batch-norm variance must be) *)
let positive (sg : Tensor_sig.t) =
  Tensor.materialize sg.Tensor_sig.shape (fun c ->
      0.25
      +. 0.05
         *. float_of_int ((Vec6.offset sg.Tensor_sig.shape c :> int) mod 11))

let sig_of_edge (b : Loop_bundle.t) id =
  List.find_map
    (fun (inv : Loop_bundle.invocation) ->
      List.find_map
        (fun ((buf : Loop_buffer.t), edge) ->
          if Tensor_id.equal edge id then Some buf.Loop_buffer.sg else None)
        (List.combine inv.Loop_bundle.program.Loop_program.buffers
           inv.Loop_bundle.edges))
    b.Loop_bundle.invocations
  |> Option.get

let map_of l id = List.assoc_opt id l

let wide_chain () =
  let module F = Native_test.Graph_fixtures in
  F.build "wide_chain"
    Graph_builder.(
      let* x = input ~shape:(F.nhwc ~h:10 ~w:10 ~c:8) () in
      let* w =
        constant ~shape:(F.weight_shape ~out_channels:16 ~in_channels:8) ()
      in
      let* bias = constant ~shape:(F.s1c 16) () in
      let* gamma = constant ~shape:(F.s1c 16) () in
      let* beta = constant ~shape:(F.s1c 16) () in
      let* mean = constant ~shape:(F.s1c 16) () in
      let* var = constant ~shape:(F.s1c 16) () in
      let* y = conv2d (F.conv_params ~in_channels:8) ~x ~weight:w ~bias () in
      let* n =
        batch_norm F.bn_params ~x:y ~weight:gamma ~bias:beta ~running_mean:mean
          ~running_var:var ()
      in
      relu n)

let run_graph ?kernel g =
  let b = Err.or_raise ~pp_error:Loop_bundle.pp_error (Loop_bundle.build g) in
  let constants =
    List.map
      (fun id -> (id, positive (sig_of_edge b id)))
      b.Loop_bundle.constants
  in
  let p =
    Err.or_raise ~pp_error:Loop_bundle_exec.pp_error
      (Loop_bundle_exec.prepare ?kernel b ~constants:(map_of constants))
  in
  let inputs =
    List.map
      (fun id -> (id, tensor_of ~salt:1 (sig_of_edge b id)))
      b.Loop_bundle.inputs
  in
  let reference =
    Err.or_raise ~pp_error:Eval_direct.pp_error
      (Eval_direct.run g ~constants ~inputs)
  in
  let out =
    Err.or_raise ~pp_error:Loop_bundle_exec.pp_error
      (Loop_bundle_exec.run p ~bind:(map_of inputs))
  in
  List.for_all
    (fun id ->
      values (Tensor_id.Map.find id out)
      = values (Tensor_id.Map.find id reference))
    g.Graph.outputs

let%expect_test "the SSA JavaScript bundle is bitwise the reference" =
  List.iter
    (fun (name, g) ->
      let g = g () in
      Fmt.pr "%s: default path identical: %b@." name (run_graph g);
      List.iter
        (fun pipeline ->
          Fmt.pr "%s %s: identical: %b@." name
            (Ssa_backends.Pipeline.name pipeline)
            (run_graph ~kernel:(Ssa_backends.js ~pipeline) g))
        [ Ssa_backends.Pipeline.Representation; Ssa_backends.Pipeline.Exact ])
    [ ("chain", Native_test.Graph_fixtures.chain); ("wide_chain", wide_chain) ];
  [%expect
    {|
    chain: default path identical: true
    chain representation: identical: true
    chain exact: identical: true
    wide_chain: default path identical: true
    wide_chain representation: identical: true
    wide_chain exact: identical: true |}]
