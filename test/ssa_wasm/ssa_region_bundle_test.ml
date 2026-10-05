open Graph_ir
open Loop_ir

(* Region-authored nodes (softmax, the norms, scaled dot-product attention with
   and without a mask, including rows a mask removes entirely) through the SSA C
   bundle, against the per-node reference and the Loop bundle. These are the
   dependent max / denominator / weighted reductions, the scans and the local
   caches the first-slice kernels do not have. *)

module F = Native_test.Graph_fixtures

let bits t =
  let (Tensor.Tensor tt) = t in
  let acc = ref [] in
  Vec6.iter tt.Tensor.shape (fun c ->
      acc := Int32.bits_of_float (Tensor.read_at t (Vec6.get c)) :: !acc);
  List.rev !acc

let sig_of_edge (b : Loop_bundle.t) id =
  Tensor_id.Map.find id b.Loop_bundle.graph.Graph.tensors

let map_of l id = List.assoc_opt id l

let with_dir f =
  let dir = Loop_c_exec.Proc.temp_dir "ssa_wasm_region_bundle_test" in
  Fun.protect
    ~finally:(fun () -> Loop_c_exec.Proc.remove_tree dir)
    (fun () -> f dir)

module H = Loop_wasm_exec.Host

(* values that vary in sign and size, so a max, a denominator and a masked row
   all matter; [masked] is the value a mask cell takes *)
let values ~salt ~masked (sg : Tensor_sig.t) =
  Tensor.materialize sg.Tensor_sig.shape (fun c ->
      let k = (Vec6.offset sg.Tensor_sig.shape c :> int) + salt in
      if masked && k mod 5 = 0 then neg_infinity
      else float_of_int ((k mod 9) - 4) /. 2.)

let run_graph ?kernel ~constant ~input g =
  with_dir (fun dir ->
      let b =
        Err.or_raise ~pp_error:Loop_bundle.pp_error
          (Loop_bundle.build ~config:Loop_bundle_wasm.default_config g)
      in
      let constants =
        List.map
          (fun id -> (id, constant (sig_of_edge b id)))
          b.Loop_bundle.constants
      in
      let p =
        Err.or_raise ~pp_error:H.pp_error
          (H.prepare ?kernel ~dir b ~constants:(map_of constants))
      in
      let inputs =
        List.mapi
          (fun i id -> (id, input i (sig_of_edge b id)))
          b.Loop_bundle.inputs
      in
      let reference =
        Err.or_raise ~pp_error:Eval_direct.pp_error
          (Eval_direct.run g ~constants ~inputs)
      in
      match Err.payload (H.run p ~bind:(map_of inputs)) with
      | Error e -> Fmt.str "run failed: %a" H.pp_error e
      | Ok outs ->
          Fmt.str "identical to the reference: %b"
            (List.for_all2
               (fun id t -> bits t = bits (Tensor_id.Map.find id reference))
               g.Graph.outputs outs))

let graphs =
  [
    ( "softmax over C",
      fun () ->
        F.build "softmax"
          Graph_builder.(
            let* x = input ~shape:(F.s 1 1 2 3 4 5) () in
            softmax { Reduce.Softmax.axis = Axis.C } x) );
    ( "layer_norm over W, C",
      fun () ->
        F.build "layer_norm"
          Graph_builder.(
            let* x = input ~shape:(F.s 1 1 1 2 4 5) () in
            layer_norm
              { Norm.LayerNorm.dims = [ Axis.W; Axis.C ]; eps = 1e-5 }
              ~x ()) );
    ( "rms_norm over C",
      fun () ->
        F.build "rms_norm"
          Graph_builder.(
            let* x = input ~shape:(F.s 1 1 1 3 4 5) () in
            rms_norm { Norm.RmsNorm.dims = [ Axis.C ]; eps = 1e-5 } ~x ()) );
    ( "sdpa, no mask",
      fun () ->
        F.build "sdpa"
          Graph_builder.(
            let* q = input ~shape:(F.s 1 1 2 3 4 5) () in
            let* k = input ~shape:(F.s 1 1 2 3 6 5) () in
            let* v = input ~shape:(F.s 1 1 2 3 6 5) () in
            sdpa
              { Attention.Sdpa.scale = Attention.Sdpa.Scale.Default }
              ~query:q ~key:k ~value:v ()) );
    ( "sdpa, explicit scale and mask",
      fun () ->
        F.build "sdpa_mask"
          Graph_builder.(
            let* q = input ~shape:(F.s 1 1 2 3 4 5) () in
            let* k = input ~shape:(F.s 1 1 2 3 6 5) () in
            let* v = input ~shape:(F.s 1 1 2 3 6 5) () in
            let* m = input ~shape:(F.s 1 1 2 3 4 6) () in
            sdpa
              { Attention.Sdpa.scale = Attention.Sdpa.Scale.Explicit 0.1 }
              ~query:q ~key:k ~value:v ~mask:m ()) );
  ]

(* the mask of attention: some cells removed, a whole row removed, every cell
   removed. A row with no cell left has no maximum and no denominator. *)
let masks =
  [
    ( "some cells removed",
      fun (sg : Tensor_sig.t) ->
        Tensor.materialize sg.Tensor_sig.shape (fun c ->
            let k = (Vec6.offset sg.Tensor_sig.shape c :> int) in
            if k mod 5 = 0 then neg_infinity else float_of_int (k mod 3) /. 4.)
    );
    ( "a whole row removed",
      fun sg ->
        Tensor.materialize sg.Tensor_sig.shape (fun c ->
            if (Vec6.get c Axis.W :> int) = 1 then neg_infinity else 0.) );
    ( "every cell removed",
      fun sg -> Tensor.materialize sg.Tensor_sig.shape (fun _ -> neg_infinity)
    );
  ]

let%expect_test "attention rows a mask removes entirely" =
  let g () =
    F.build "sdpa_mask"
      Graph_builder.(
        let* q = input ~shape:(F.s 1 1 2 3 4 5) () in
        let* k = input ~shape:(F.s 1 1 2 3 6 5) () in
        let* v = input ~shape:(F.s 1 1 2 3 6 5) () in
        let* m = input ~shape:(F.s 1 1 2 3 4 6) () in
        sdpa
          { Attention.Sdpa.scale = Attention.Sdpa.Scale.Default }
          ~query:q ~key:k ~value:v ~mask:m ())
  in
  List.iter
    (fun (name, mask) ->
      let input i sg =
        if i = 3 then mask sg else values ~salt:(i + 1) ~masked:false sg
      in
      let constant sg = values ~salt:3 ~masked:false sg in
      let g = g () in
      Fmt.pr "%s@." name;
      Fmt.pr "  loop: %s@." (run_graph ~constant ~input g);
      Fmt.pr "  ssa exact: %s@."
        (run_graph
           ~kernel:(fun ~table_alloc inv ->
             Ssa_backends.wasm ~pipeline:Ssa_backends.Pipeline.Exact
               ~table_alloc inv)
           ~constant ~input g))
    masks;
  [%expect
    {|
    some cells removed
      loop: identical to the reference: true
      ssa exact: identical to the reference: true
    a whole row removed
      loop: identical to the reference: true
      ssa exact: identical to the reference: true
    every cell removed
      loop: identical to the reference: true
      ssa exact: identical to the reference: true |}]

let%expect_test "region-authored nodes through the SSA Wasm bundle" =
  List.iter
    (fun (name, g) ->
      let g = g () in
      (* the mask is the last input: removed cells, and whole rows of them *)
      let input i sg = values ~salt:(i + 1) ~masked:(i = 3) sg in
      let constant sg = values ~salt:3 ~masked:false sg in
      Fmt.pr "%s@." name;
      Fmt.pr "  loop: %s@." (run_graph ~constant ~input g);
      List.iter
        (fun pipeline ->
          Fmt.pr "  ssa %s: %s@."
            (Ssa_backends.Pipeline.name pipeline)
            (run_graph
               ~kernel:(fun ~table_alloc inv ->
                 Ssa_backends.wasm ~pipeline ~table_alloc inv)
               ~constant ~input g))
        [ Ssa_backends.Pipeline.Representation; Ssa_backends.Pipeline.Exact ])
    graphs;
  [%expect
    {|
    softmax over C
      loop: identical to the reference: true
      ssa representation: identical to the reference: true
      ssa exact: identical to the reference: true
    layer_norm over W, C
      loop: identical to the reference: true
      ssa representation: identical to the reference: true
      ssa exact: identical to the reference: true
    rms_norm over C
      loop: identical to the reference: true
      ssa representation: identical to the reference: true
      ssa exact: identical to the reference: true
    sdpa, no mask
      loop: identical to the reference: true
      ssa representation: identical to the reference: true
      ssa exact: identical to the reference: true
    sdpa, explicit scale and mask
      loop: identical to the reference: true
      ssa representation: identical to the reference: true
      ssa exact: identical to the reference: true |}]
