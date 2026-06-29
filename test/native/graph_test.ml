(* Direct evaluation of native graphs: simple sequences, the conv NCHW->NHWC
   decomposition, and a nested subgraph. Each test prints intermediate tensors (not
   just outputs) to show the whole computation. See .ai/native_graph_design.md. *)

open Graph_ir

let s n t d h w c = Vec6.shape ~n ~t ~d ~h ~w ~c
let s1c n = s 1 1 1 1 1 n
let chan c = Dim.to_int (Vec6.get c Axis.C)

let conv_axis ~kernel ~stride ~pad : Conv.Conv2d.axis_window =
  {
    kernel = Dim.extent kernel;
    stride = Op_config.Pos.of_int stride;
    pad_before = Op_config.Nonneg.of_int pad;
    pad_after = Op_config.Nonneg.of_int pad;
    dilation = Op_config.Pos.of_int 1;
  }

(* The edge id whose signature carries [name] (edges we look up are named). *)
let id_of_name (g : graph) name =
  Tensor_id.Map.fold
    (fun id (sg : Tensor_sig.t) acc ->
      if String.equal sg.name name then Some id else acc)
    g.Graph.tensors None
  |> Option.get

let tensors_match shape a b =
  let ok = ref true in
  Vec6.iter shape (fun c ->
      if not (Float.equal (Tensor.read a c) (Tensor.read b c)) then ok := false);
  !ok

let%expect_test "Direct graph: sequence add -> relu (with intermediate)" =
  let g =
    Graph_builder.(
      build ~name:"seq" ~outputs:(fun r -> [ r ])
      @@
      let* a = input ~shape:(s1c 4) ~name:"a" () in
      let* b = input ~shape:(s1c 4) ~name:"b" () in
      let* t = add ~name:"sum" a b in
      relu ~name:"out" t)
  in
  let a =
    Tensor.materialize (s1c 4) (fun c -> [| 1.; -5.; 2.; -8. |].(chan c))
  in
  let b =
    Tensor.materialize (s1c 4) (fun c -> [| -3.; 1.; 2.; 3. |].(chan c))
  in
  let env = Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ a; b ]) in
  let get name = Tensor_id.Map.find (id_of_name g name) env in
  Format.printf "sum = %a@." Tensor.pp (get "sum");
  Format.printf "out = %a@." Tensor.pp (get "out");
  [%expect
    {|
    sum = tensor f32 [C=4] {-2, -4, 4, -5}
    out = tensor f32 [C=4] {0, 0, 4, 0} |}]

let%expect_test "Direct graph: add of two inputs" =
  let g =
    Graph_builder.(
      build ~name:"add" ~outputs:(fun r -> [ r ])
      @@
      let* a = input ~shape:(s1c 3) ~name:"a" () in
      let* b = input ~shape:(s1c 3) ~name:"b" () in
      add ~name:"out" a b)
  in
  let a = Tensor.materialize (s1c 3) (fun c -> float_of_int (chan c)) in
  let b = Tensor.materialize (s1c 3) (fun _ -> 10.) in
  let env = Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ a; b ]) in
  Format.printf "out = %a@." Tensor.pp
    (Tensor_id.Map.find (id_of_name g "out") env);
  [%expect {| out = tensor f32 [C=3] {10, 11, 12} |}]

let%expect_test "Direct graph: mul of two inputs" =
  let g =
    Graph_builder.(
      build ~name:"mul" ~outputs:(fun r -> [ r ])
      @@
      let* a = input ~shape:(s1c 3) ~name:"a" () in
      let* b = input ~shape:(s1c 3) ~name:"b" () in
      mul ~name:"out" a b)
  in
  let a = Tensor.materialize (s1c 3) (fun c -> float_of_int (chan c)) in
  let b = Tensor.materialize (s1c 3) (fun _ -> 10.) in
  let env = Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ a; b ]) in
  Format.printf "out = %a@." Tensor.pp
    (Tensor_id.Map.find (id_of_name g "out") env);
  [%expect {| out = tensor f32 [C=3] {0, 10, 20} |}]

(* Conv decomposition. The input is laid out NCHW: in the 6D frame its channel sits
   on H, spatial-H on W, spatial-W on C. Two permutes bracket a native (NHWC)
   conv: NCHW->NHWC moves the channel to C and the spatial axes to H/W; NHWC->NCHW
   is its inverse. *)
let conv_params =
  {
    Conv.Conv2d.h = conv_axis ~kernel:2 ~stride:1 ~pad:0;
    w = conv_axis ~kernel:2 ~stride:1 ~pad:0;
    in_channels = Dim.extent 2;
    groups = Op_config.Pos.of_int 1;
  }

let p_to_nhwc = Axis.[ (N, N); (T, T); (D, D); (H, W); (W, C); (C, H) ]
let p_to_nchw = Axis.[ (N, N); (T, T); (D, D); (H, C); (W, H); (C, W) ]

let%expect_test
    "Direct graph: conv NCHW -> permute -> NHWC conv -> permute -> NCHW" =
  let g =
    Graph_builder.(
      build ~name:"conv_decomp" ~outputs:(fun (_, _, _, y) -> [ y ])
      @@
      let* x = input ~shape:(s 1 1 1 2 3 3) ~name:"x_nchw" () in
      let* w = input ~shape:(s 1 1 1 2 2 2) ~name:"w" () in
      let* b = input ~shape:(s1c 1) ~name:"b" () in
      let* xh = permute ~name:"x_nhwc" p_to_nhwc x in
      let* yh = conv2d ~name:"y_nhwc" conv_params ~x:xh ~weight:w ~bias:b () in
      let* y = permute ~name:"y_nchw" p_to_nchw yh in
      return (x, w, b, y))
  in
  (* NCHW frame: value(chan,spH,spW) = chan*100 + spH*10 + spW *)
  let x =
    Tensor.materialize (s 1 1 1 2 3 3) (fun c ->
        let chan = Dim.to_int (Vec6.get c Axis.H) in
        let sph = Dim.to_int (Vec6.get c Axis.W) in
        let spw = Dim.to_int (Vec6.get c Axis.C) in
        float_of_int ((chan * 100) + (sph * 10) + spw))
  in
  let w = Tensor.materialize (s 1 1 1 2 2 2) (fun _ -> 1.) in
  let b = Tensor.materialize (s1c 1) (fun _ -> 0.) in
  let env =
    Eval_direct.run g ~inputs:(List.combine g.Graph.inputs [ x; w; b ])
  in
  let get name = Tensor_id.Map.find (id_of_name g name) env in
  Format.printf "x_nhwc = %a@." Tensor.pp (get "x_nhwc");
  Format.printf "y_nhwc = %a@." Tensor.pp (get "y_nhwc");
  Format.printf "y_nchw = %a@." Tensor.pp (get "y_nchw");
  (* reference: a single native conv run directly on the NHWC-laid input *)
  let module Cv = Conv.Conv2d.Compute (Direct) in
  let xh = get "x_nhwc" in
  let (Tensor.Tensor r) = xh in
  let ref_shape =
    Conv.Conv2d.output_shape ~x_shape:r.shape ~weight_shape:(s 1 1 1 2 2 2)
      conv_params
  in
  let ref_y =
    Schedule.evaluate ref_shape
      (Cv.pixel conv_params ~x_shape:r.shape ~weight_shape:(s 1 1 1 2 2 2) ~x:xh
         ~weight:w ~bias:b)
  in
  Format.printf "y_nhwc matches single conv: %b@."
    (tensors_match ref_shape ref_y (get "y_nhwc"));
  [%expect
    {|
    x_nhwc = tensor f32 [H=3 W=3 C=2] {0, 100, 1, 101, 2, 102, 10, 110, ...}
    y_nhwc = tensor f32 [H=2 W=2 C=1] {444, 452, 524, 532}
    y_nchw = tensor f32 [W=2 C=2] {444, 452, 524, 532}
    y_nhwc matches single conv: true |}]

(* Conv with no bias: omit [?bias] so the IR carries [bias = None]; the evaluator
   fills a zeros bias. The result must equal the same conv with an explicit zero
   bias. *)
let%expect_test "Direct graph: conv with optional bias omitted (None -> zeros)"
    =
  let build_one ~with_bias =
    Graph_builder.(
      build ~name:"conv" ~outputs:(fun y -> [ y ])
      @@
      let* x = input ~shape:(s 1 1 1 3 3 2) ~name:"x" () in
      let* w = input ~shape:(s 1 1 1 2 2 2) ~name:"w" () in
      if with_bias then
        let* b = input ~shape:(s1c 1) ~name:"b" () in
        conv2d ~name:"y" conv_params ~x ~weight:w ~bias:b ()
      else conv2d ~name:"y" conv_params ~x ~weight:w ())
  in
  let x =
    Tensor.materialize (s 1 1 1 3 3 2) (fun c ->
        float_of_int (Dim.to_int (Vec6.get c Axis.H) + chan c))
  in
  let w = Tensor.materialize (s 1 1 1 2 2 2) (fun _ -> 1.) in
  let b = Tensor.materialize (s1c 1) (fun _ -> 0.) in
  let g_no = build_one ~with_bias:false in
  let g_yes = build_one ~with_bias:true in
  let env_no =
    Eval_direct.run g_no ~inputs:(List.combine g_no.Graph.inputs [ x; w ])
  in
  let env_yes =
    Eval_direct.run g_yes ~inputs:(List.combine g_yes.Graph.inputs [ x; w; b ])
  in
  let y_no = Tensor_id.Map.find (id_of_name g_no "y") env_no in
  let y_yes = Tensor_id.Map.find (id_of_name g_yes "y") env_yes in
  let (Tensor.Tensor r) = y_no in
  Format.printf "y (no bias) = %a@." Tensor.pp y_no;
  Format.printf "matches explicit zero bias: %b@."
    (tensors_match r.shape y_no y_yes);
  [%expect
    {|
    y (no bias) = tensor f32 [H=2 W=2 C=1] {8, 8, 16, 16}
    matches explicit zero bias: true |}]

(* Nested subgraph: an outer graph invokes a named subgraph that computes
   add -> relu. The result must match the same computation built flat. *)
let%expect_test "Direct graph: nested subgraph evaluation" =
  let nested =
    Graph_builder.(
      build ~name:"outer" ~outputs:(fun outs -> outs)
      @@
      let* x = input ~shape:(s1c 4) ~name:"x" () in
      let* y = input ~shape:(s1c 4) ~name:"y" () in
      let* seq =
        subgraph ~name:"add_relu"
          (let* a = input ~shape:(s1c 4) ~name:"a" () in
           let* b = input ~shape:(s1c 4) ~name:"b" () in
           let* t = add ~name:"sum" a b in
           let* r = relu ~name:"r" t in
           return [ r ])
      in
      invoke ~names:[ "nested_out" ] seq [ x; y ])
  in
  let flat =
    Graph_builder.(
      build ~name:"flat" ~outputs:(fun r -> [ r ])
      @@
      let* x = input ~shape:(s1c 4) ~name:"x" () in
      let* y = input ~shape:(s1c 4) ~name:"y" () in
      let* t = add x y in
      relu ~name:"out" t)
  in
  let x =
    Tensor.materialize (s1c 4) (fun c -> [| 1.; -5.; 2.; -8. |].(chan c))
  in
  let y =
    Tensor.materialize (s1c 4) (fun c -> [| -3.; 1.; 2.; 3. |].(chan c))
  in
  let env_n =
    Eval_direct.run nested ~inputs:(List.combine nested.Graph.inputs [ x; y ])
  in
  let env_f =
    Eval_direct.run flat ~inputs:(List.combine flat.Graph.inputs [ x; y ])
  in
  let out_n = Tensor_id.Map.find (id_of_name nested "nested_out") env_n in
  let out_f = Tensor_id.Map.find (id_of_name flat "out") env_f in
  Format.printf "nested_out = %a@." Tensor.pp out_n;
  Format.printf "matches flat: %b@." (tensors_match (s1c 4) out_n out_f);
  [%expect
    {|
    nested_out = tensor f32 [C=4] {0, 0, 4, 0}
    matches flat: true |}]

(* Tensor_sig.id must equal the Tensor_id key in the graph's tensor map.
   This verifies the state-monad id generator produces consistent ids. *)
let%expect_test "Builder: Tensor_sig.id equals Tensor_id for all edges" =
  let g =
    Graph_builder.(
      build ~name:"check" ~outputs:(fun r -> [ r ])
      @@
      let* a = input ~shape:(s1c 2) ~name:"a" () in
      let* b = input ~shape:(s1c 2) ~name:"b" () in
      add ~name:"out" a b)
  in
  Tensor_id.Map.iter
    (fun tid sg ->
      Format.printf "tid=%a sig_id=%a match=%b@." Tensor_id.pp tid Tensor_id.pp
        sg.Tensor_sig.id
        (Tensor_id.equal tid sg.Tensor_sig.id))
    g.Graph.tensors;
  [%expect
    {|
    tid=t0 sig_id=t0 match=true
    tid=t1 sig_id=t1 match=true
    tid=t2 sig_id=t2 match=true |}]

(* Building the same graph twice must produce identical tensor ids: the id
   generator is local to [build], not a global counter. *)
let%expect_test "Builder: ids are deterministic across multiple build calls" =
  let build_add () =
    Graph_builder.(
      build ~name:"add" ~outputs:(fun r -> [ r ])
      @@
      let* a = input ~shape:(s1c 3) ~name:"a" () in
      let* b = input ~shape:(s1c 3) ~name:"b" () in
      add ~name:"out" a b)
  in
  let g1 = build_add () in
  let g2 = build_add () in
  let ids_of g =
    Tensor_id.Map.bindings g.Graph.tensors
    |> List.map (fun (tid, sg) -> (tid, sg.Tensor_sig.id))
  in
  let pp_ids ids =
    ids
    |> List.map (fun (t, s) ->
        Format.asprintf "(%a,%a)" Tensor_id.pp t Tensor_id.pp s)
    |> String.concat " "
  in
  Format.printf "g1 ids: %s@." (pp_ids (ids_of g1));
  Format.printf "g2 ids: %s@." (pp_ids (ids_of g2));
  Format.printf "match: %b@." (ids_of g1 = ids_of g2);
  [%expect
    {|
    g1 ids: (t0,t0) (t1,t1) (t2,t2)
    g2 ids: (t0,t0) (t1,t1) (t2,t2)
    match: true |}]
