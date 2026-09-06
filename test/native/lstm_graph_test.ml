(* End-to-end coverage for the WIP `Lstm` Native op (project step 13 / M3):
   builds one tiny single-layer, single-direction, time-first graph through
   the real Graph_builder/Region_computation/Eval_direct pipeline and checks
   all three outputs (live together, per M3's own exit condition) against a
   plain OCaml reference -- the same cross-check discipline
   test/native/lstm_region_prototype_test.ml used before any graph
   registration existed. Only this one configuration is covered here;
   stacked layers, bidirectionality and batch-first are follow-up work
   (M3b). *)

let k = 2 (* hidden_size *)
let isz = 2 (* input width *)
let seq = 3 (* sequence length *)
let batch = 1
let mat_shape ~rows ~cols = Vec6.shape ~n:rows ~t:1 ~d:1 ~h:1 ~w:1 ~c:cols
let vec_shape ~n = Vec6.shape ~n ~t:1 ~d:1 ~h:1 ~w:1 ~c:1
let state_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:batch ~c:k
let seq_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:seq ~w:batch ~c:isz

let tensor_of_array shape arr =
  Tensor.materialize shape (fun coord -> arr.((Vec6.offset shape coord :> int)))

let weight_hh =
  [|
    0.1;
    -0.2;
    0.05;
    0.3;
    -0.1;
    0.2;
    0.15;
    -0.05;
    0.2;
    0.1;
    -0.3;
    0.1;
    0.05;
    0.2;
    -0.1;
    0.3;
  |]

let weight_ih =
  [|
    0.2;
    -0.1;
    0.1;
    0.3;
    -0.2;
    0.05;
    0.1;
    -0.3;
    0.05;
    0.2;
    -0.1;
    0.1;
    0.3;
    -0.05;
    0.2;
    -0.2;
  |]

let bias_hh = [| 0.1; -0.1; 0.2; 0.05; -0.2; 0.1; 0.05; -0.05 |]
let bias_ih = [| 0.05; 0.1; -0.1; 0.2; 0.1; -0.05; -0.1; 0.05 |]
let h0 = [| 0.3; -0.4 |]
let c0 = [| 0.5; 0.1 |]
let input = [| 1.0; -0.5; 0.2; 0.7; -0.3; 0.4 |]
let params : Lstm.Lstm.params = { hidden_size = k; input_size = isz }

(* Declaration order below is what [graph.Graph_ir.Graph.inputs] returns ("ordered =
   the graph's signature", graph_common.ml) -- this list supplies the
   matching concrete tensor for each, positionally. *)
let input_tensors =
  [
    tensor_of_array seq_shape input;
    tensor_of_array (mat_shape ~rows:(4 * k) ~cols:isz) weight_ih;
    tensor_of_array (mat_shape ~rows:(4 * k) ~cols:k) weight_hh;
    tensor_of_array (vec_shape ~n:(4 * k)) bias_ih;
    tensor_of_array (vec_shape ~n:(4 * k)) bias_hh;
    tensor_of_array state_shape h0;
    tensor_of_array state_shape c0;
  ]

let graph =
  Err.or_raise ~pp_error:Graph_builder.pp_error
    Graph_builder.(
      build ~name:"lstm" ~outputs:(fun (out, hn, cn) -> [ out; hn; cn ])
      @@
      let* input_id = input ~shape:seq_shape ~name:"input" () in
      let* weight_ih_id =
        input ~shape:(mat_shape ~rows:(4 * k) ~cols:isz) ~name:"weight_ih" ()
      in
      let* weight_hh_id =
        input ~shape:(mat_shape ~rows:(4 * k) ~cols:k) ~name:"weight_hh" ()
      in
      let* bias_ih_id =
        input ~shape:(vec_shape ~n:(4 * k)) ~name:"bias_ih" ()
      in
      let* bias_hh_id =
        input ~shape:(vec_shape ~n:(4 * k)) ~name:"bias_hh" ()
      in
      let* h0_id = input ~shape:state_shape ~name:"h0" () in
      let* c0_id = input ~shape:state_shape ~name:"c0" () in
      let layer : Lstm.Lstm.Layer.t =
        {
          forward =
            {
              weight_ih = weight_ih_id;
              weight_hh = weight_hh_id;
              bias = Some (bias_ih_id, bias_hh_id);
            };
          reverse = None;
        }
      in
      Graph_builder.lstm params ~input:input_id ~layers:[ layer ] ~h0:h0_id
        ~c0:c0_id ())

(* A plain OCaml recurrence, independent of Region/Expr/Graph machinery, as
   the cross-check oracle -- mirrors
   lstm_region_prototype_test.ml's [reference], batch folded away (=1). *)
let reference () =
  let sigmoid x = 1. /. (1. +. exp (-.x)) in
  let get2 arr ~cols row col = arr.((row * cols) + col) in
  let h = Array.copy h0 and c = Array.copy c0 in
  let out = Array.make (seq * k) 0. in
  for s = 0 to seq - 1 do
    let gate r j =
      let row = (r * k) + j in
      let hh = ref 0. and ih = ref 0. in
      for jj = 0 to k - 1 do
        hh := !hh +. (get2 weight_hh ~cols:k row jj *. h.(jj))
      done;
      for jj = 0 to isz - 1 do
        ih := !ih +. (get2 weight_ih ~cols:isz row jj *. input.((s * isz) + jj))
      done;
      !hh +. !ih +. bias_hh.(row) +. bias_ih.(row)
    in
    (* Every lane's gates read the SAME previous-step [h]/[c]; computing
       [next_h]/[next_c] for every lane before writing any of them back
       avoids lane 1 reading lane 0's already-advanced state. *)
    let next_h = Array.make k 0. and next_c = Array.make k 0. in
    for j = 0 to k - 1 do
      let i = sigmoid (gate 0 j) in
      let f = sigmoid (gate 1 j) in
      let g = tanh (gate 2 j) in
      let o = sigmoid (gate 3 j) in
      next_c.(j) <- (f *. c.(j)) +. (i *. g);
      next_h.(j) <- o *. tanh next_c.(j)
    done;
    for j = 0 to k - 1 do
      out.((s * k) + j) <- next_h.(j);
      h.(j) <- next_h.(j);
      c.(j) <- next_c.(j)
    done
  done;
  (out, h, c)

let%expect_test
    "a tiny single-layer/direction LSTM graph agrees with a plain OCaml \
     reference on all three outputs" =
  let inputs = List.combine graph.Graph_ir.Graph.inputs input_tensors in
  let env =
    Err.or_raise ~pp_error:Eval_direct.pp_error (Eval_direct.run graph ~inputs)
  in
  let out_id, hn_id, cn_id =
    match graph.Graph_ir.Graph.outputs with
    | [ out; hn; cn ] -> (out, hn, cn)
    | _ -> assert false
  in
  let read t coord = Tensor.read t coord in
  let max_diff = ref 0. in
  let expected_out, expected_hn, expected_cn = reference () in
  let out_t = Tensor_id.Map.find out_id env in
  for s = 0 to seq - 1 do
    for j = 0 to k - 1 do
      let p = read out_t (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:s ~w:0 ~c:j) in
      max_diff :=
        Float.max !max_diff (Float.abs (p -. expected_out.((s * k) + j)))
    done
  done;
  let hn_t = Tensor_id.Map.find hn_id env
  and cn_t = Tensor_id.Map.find cn_id env in
  for j = 0 to k - 1 do
    let hp = read hn_t (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:j) in
    let cp = read cn_t (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:j) in
    max_diff := Float.max !max_diff (Float.abs (hp -. expected_hn.(j)));
    max_diff := Float.max !max_diff (Float.abs (cp -. expected_cn.(j)))
  done;
  Fmt.pr "max_abs_diff=%g@." !max_diff;
  [%expect {| max_abs_diff=4.16169e-09 |}]

(* Q=2 stacked layers, R=1, time-first: layer 0 reads the raw input;
   layer 1 reads layer 0's completed hidden trace as its own per-timestep
   input (I_1=hidden_size, lstm-plan.md §2/§4). Distinct weights per layer
   make a layer mixup or an input-width slip visible. *)
module Stacked = struct
  let k = 2
  let isz = 2
  let seq = 3
  let batch = 1
  let num_layers = 2
  let mat_shape ~rows ~cols = Vec6.shape ~n:rows ~t:1 ~d:1 ~h:1 ~w:1 ~c:cols
  let vec_shape ~n = Vec6.shape ~n ~t:1 ~d:1 ~h:1 ~w:1 ~c:1
  let state_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:num_layers ~w:batch ~c:k
  let seq_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:seq ~w:batch ~c:isz

  let weight_hh0 =
    [|
      0.1;
      -0.2;
      0.05;
      0.3;
      -0.1;
      0.2;
      0.15;
      -0.05;
      0.2;
      0.1;
      -0.3;
      0.1;
      0.05;
      0.2;
      -0.1;
      0.3;
    |]

  let weight_ih0 =
    [|
      0.2;
      -0.1;
      0.1;
      0.3;
      -0.2;
      0.05;
      0.1;
      -0.3;
      0.05;
      0.2;
      -0.1;
      0.1;
      0.3;
      -0.05;
      0.2;
      -0.2;
    |]

  let bias_hh0 = [| 0.1; -0.1; 0.2; 0.05; -0.2; 0.1; 0.05; -0.05 |]
  let bias_ih0 = [| 0.05; 0.1; -0.1; 0.2; 0.1; -0.05; -0.1; 0.05 |]

  let weight_hh1 =
    [|
      -0.05;
      0.15;
      0.2;
      -0.1;
      0.1;
      0.05;
      -0.2;
      0.3;
      0.25;
      -0.15;
      0.05;
      0.1;
      -0.1;
      0.2;
      0.1;
      -0.25;
    |]

  let weight_ih1 =
    [|
      0.3;
      -0.2;
      -0.1;
      0.25;
      0.1;
      -0.05;
      0.2;
      -0.3;
      -0.05;
      0.1;
      0.15;
      -0.1;
      0.2;
      0.05;
      -0.15;
      0.1;
    |]

  let bias_hh1 = [| -0.05; 0.1; 0.05; -0.1; 0.15; -0.05; 0.1; 0.2 |]
  let bias_ih1 = [| 0.1; -0.05; 0.2; 0.05; -0.1; 0.1; -0.05; 0.15 |]
  let h0 = [| 0.3; -0.4; 0.1; 0.2 |] (* [layer0; layer1], each width k *)
  let c0 = [| 0.5; 0.1; -0.2; 0.3 |]
  let input = [| 1.0; -0.5; 0.2; 0.7; -0.3; 0.4 |]
  let params : Lstm.Lstm.params = { hidden_size = k; input_size = isz }

  let graph =
    Err.or_raise ~pp_error:Graph_builder.pp_error
      Graph_builder.(
        build ~name:"lstm_stacked" ~outputs:(fun (out, hn, cn) ->
            [ out; hn; cn ])
        @@
        let* input_id = input ~shape:seq_shape ~name:"input" () in
        let* wih0 =
          input ~shape:(mat_shape ~rows:(4 * k) ~cols:isz) ~name:"wih0" ()
        in
        let* whh0 =
          input ~shape:(mat_shape ~rows:(4 * k) ~cols:k) ~name:"whh0" ()
        in
        let* bih0 = input ~shape:(vec_shape ~n:(4 * k)) ~name:"bih0" () in
        let* bhh0 = input ~shape:(vec_shape ~n:(4 * k)) ~name:"bhh0" () in
        let* wih1 =
          input ~shape:(mat_shape ~rows:(4 * k) ~cols:k) ~name:"wih1" ()
        in
        let* whh1 =
          input ~shape:(mat_shape ~rows:(4 * k) ~cols:k) ~name:"whh1" ()
        in
        let* bih1 = input ~shape:(vec_shape ~n:(4 * k)) ~name:"bih1" () in
        let* bhh1 = input ~shape:(vec_shape ~n:(4 * k)) ~name:"bhh1" () in
        let* h0_id = input ~shape:state_shape ~name:"h0" () in
        let* c0_id = input ~shape:state_shape ~name:"c0" () in
        let layer0 : Lstm.Lstm.Layer.t =
          {
            forward =
              { weight_ih = wih0; weight_hh = whh0; bias = Some (bih0, bhh0) };
            reverse = None;
          }
        in
        let layer1 : Lstm.Lstm.Layer.t =
          {
            forward =
              { weight_ih = wih1; weight_hh = whh1; bias = Some (bih1, bhh1) };
            reverse = None;
          }
        in
        Graph_builder.lstm params ~input:input_id ~layers:[ layer0; layer1 ]
          ~h0:h0_id ~c0:c0_id ())

  let input_tensors =
    [
      tensor_of_array seq_shape input;
      tensor_of_array (mat_shape ~rows:(4 * k) ~cols:isz) weight_ih0;
      tensor_of_array (mat_shape ~rows:(4 * k) ~cols:k) weight_hh0;
      tensor_of_array (vec_shape ~n:(4 * k)) bias_ih0;
      tensor_of_array (vec_shape ~n:(4 * k)) bias_hh0;
      tensor_of_array (mat_shape ~rows:(4 * k) ~cols:k) weight_ih1;
      tensor_of_array (mat_shape ~rows:(4 * k) ~cols:k) weight_hh1;
      tensor_of_array (vec_shape ~n:(4 * k)) bias_ih1;
      tensor_of_array (vec_shape ~n:(4 * k)) bias_hh1;
      tensor_of_array state_shape h0;
      tensor_of_array state_shape c0;
    ]

  let sigmoid x = 1. /. (1. +. exp (-.x))
  let get2 arr ~cols row col = arr.((row * cols) + col)

  let gate ~weight_hh ~weight_ih ~bias_hh ~bias_ih ~input_size r j ~h ~inp =
    let row = (r * k) + j in
    let hh = ref 0. in
    for jj = 0 to k - 1 do
      hh := !hh +. (get2 weight_hh ~cols:k row jj *. h.(jj))
    done;
    let ih = ref 0. in
    for jj = 0 to input_size - 1 do
      ih := !ih +. (get2 weight_ih ~cols:input_size row jj *. inp.(jj))
    done;
    !hh +. !ih +. bias_hh.(row) +. bias_ih.(row)

  let step ~weight_hh ~weight_ih ~bias_hh ~bias_ih ~input_size h c inp =
    let next_h = Array.make k 0. and next_c = Array.make k 0. in
    for j = 0 to k - 1 do
      let i =
        sigmoid
          (gate ~weight_hh ~weight_ih ~bias_hh ~bias_ih ~input_size 0 j ~h ~inp)
      in
      let f =
        sigmoid
          (gate ~weight_hh ~weight_ih ~bias_hh ~bias_ih ~input_size 1 j ~h ~inp)
      in
      let g =
        tanh
          (gate ~weight_hh ~weight_ih ~bias_hh ~bias_ih ~input_size 2 j ~h ~inp)
      in
      let o =
        sigmoid
          (gate ~weight_hh ~weight_ih ~bias_hh ~bias_ih ~input_size 3 j ~h ~inp)
      in
      next_c.(j) <- (f *. c.(j)) +. (i *. g);
      next_h.(j) <- o *. tanh next_c.(j)
    done;
    (next_h, next_c)

  let run_layer ~weight_hh ~weight_ih ~bias_hh ~bias_ih ~input_size ~h0 ~c0
      ~get_input =
    let h = ref h0 and c = ref c0 in
    let out = Array.make (seq * k) 0. in
    for s = 0 to seq - 1 do
      let nh, nc =
        step ~weight_hh ~weight_ih ~bias_hh ~bias_ih ~input_size !h !c
          (get_input s)
      in
      for j = 0 to k - 1 do
        out.((s * k) + j) <- nh.(j)
      done;
      h := nh;
      c := nc
    done;
    (out, !h, !c)

  let reference () =
    let h0_l0 = [| h0.(0); h0.(1) |] and c0_l0 = [| c0.(0); c0.(1) |] in
    let h0_l1 = [| h0.(2); h0.(3) |] and c0_l1 = [| c0.(2); c0.(3) |] in
    let layer0_out, h_n0, c_n0 =
      run_layer ~weight_hh:weight_hh0 ~weight_ih:weight_ih0 ~bias_hh:bias_hh0
        ~bias_ih:bias_ih0 ~input_size:isz ~h0:h0_l0 ~c0:c0_l0
        ~get_input:(fun s -> Array.sub input (s * isz) isz)
    in
    let layer1_out, h_n1, c_n1 =
      run_layer ~weight_hh:weight_hh1 ~weight_ih:weight_ih1 ~bias_hh:bias_hh1
        ~bias_ih:bias_ih1 ~input_size:k ~h0:h0_l1 ~c0:c0_l1 ~get_input:(fun s ->
          Array.sub layer0_out (s * k) k)
    in
    (layer1_out, [| h_n0; h_n1 |], [| c_n0; c_n1 |])

  let%expect_test
      "a two-layer stacked LSTM graph agrees with a plain OCaml reference on \
       all three outputs" =
    let inputs = List.combine graph.Graph_ir.Graph.inputs input_tensors in
    let env =
      Err.or_raise ~pp_error:Eval_direct.pp_error
        (Eval_direct.run graph ~inputs)
    in
    let out_id, hn_id, cn_id =
      match graph.Graph_ir.Graph.outputs with
      | [ out; hn; cn ] -> (out, hn, cn)
      | _ -> assert false
    in
    let out_t = Tensor_id.Map.find out_id env in
    let hn_t = Tensor_id.Map.find hn_id env in
    let cn_t = Tensor_id.Map.find cn_id env in
    let expected_out, expected_hn, expected_cn = reference () in
    let max_diff = ref 0. in
    for s = 0 to seq - 1 do
      for j = 0 to k - 1 do
        let p = Tensor.read out_t (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:s ~w:0 ~c:j) in
        max_diff :=
          Float.max !max_diff (Float.abs (p -. expected_out.((s * k) + j)))
      done
    done;
    for layer = 0 to num_layers - 1 do
      for j = 0 to k - 1 do
        let hp =
          Tensor.read hn_t (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:layer ~w:0 ~c:j)
        in
        let cp =
          Tensor.read cn_t (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:layer ~w:0 ~c:j)
        in
        max_diff :=
          Float.max !max_diff (Float.abs (hp -. expected_hn.(layer).(j)));
        max_diff :=
          Float.max !max_diff (Float.abs (cp -. expected_cn.(layer).(j)))
      done
    done;
    Fmt.pr "max_abs_diff=%g@." !max_diff;
    [%expect {| max_abs_diff=6.42713e-09 |}]
end

(* Proves the two out-of-scope-configuration rejections actually fire (not
   just typed and unreachable): a present [reverse] direction, and an empty
   [layers] list. *)
let%expect_test "a reverse direction is rejected, not silently ignored" =
  let build () =
    Graph_builder.(
      build ~name:"lstm_reverse" ~outputs:(fun (out, _, _) -> [ out ])
      @@
      let* input_id = input ~shape:seq_shape ~name:"input" () in
      let* wih =
        input ~shape:(mat_shape ~rows:(4 * k) ~cols:isz) ~name:"wih" ()
      in
      let* whh =
        input ~shape:(mat_shape ~rows:(4 * k) ~cols:k) ~name:"whh" ()
      in
      let* h0_id = input ~shape:state_shape ~name:"h0" () in
      let* c0_id = input ~shape:state_shape ~name:"c0" () in
      let layer : Lstm.Lstm.Layer.t =
        {
          forward = { weight_ih = wih; weight_hh = whh; bias = None };
          reverse = Some { weight_ih = wih; weight_hh = whh; bias = None };
        }
      in
      Graph_builder.lstm params ~input:input_id ~layers:[ layer ] ~h0:h0_id
        ~c0:c0_id ())
  in
  (match build () with
  | Ok _ -> Fmt.pr "unexpectedly built@."
  | Error e -> Fmt.pr "%a@." Graph_builder.pp_error (Err.Error.kind e));
  [%expect {| lstm: bidirectional (reverse direction) is not yet supported |}]

let%expect_test "an empty layer list is rejected, not silently ignored" =
  let build () =
    Graph_builder.(
      build ~name:"lstm_empty" ~outputs:(fun (out, _, _) -> [ out ])
      @@
      let* input_id = input ~shape:seq_shape ~name:"input" () in
      let* h0_id = input ~shape:state_shape ~name:"h0" () in
      let* c0_id = input ~shape:state_shape ~name:"c0" () in
      Graph_builder.lstm params ~input:input_id ~layers:[] ~h0:h0_id ~c0:c0_id
        ())
  in
  (match build () with
  | Ok _ -> Fmt.pr "unexpectedly built@."
  | Error e -> Fmt.pr "%a@." Graph_builder.pp_error (Err.Error.kind e));
  [%expect {| lstm: at least one layer is required |}]
