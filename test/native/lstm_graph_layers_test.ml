(* Stacked-layer and bidirectional coverage for the WIP `Lstm` Native op
   (project step 13 / M3b), split out of lstm_graph_test.ml (which keeps
   the single-layer/direction case) to stay under the tracked file-size
   ceiling. Each module builds one tiny graph through the real
   Graph_builder/Region_computation/Eval_direct pipeline and checks all
   three outputs against an independent plain OCaml reference. *)

let tensor_of_array shape arr =
  Tensor.materialize shape (fun coord -> arr.((Vec6.offset shape coord :> int)))

let k = 2 (* hidden_size *)
let isz = 2 (* input width *)
let seq = 3 (* sequence length *)
let batch = 1
let mat_shape ~rows ~cols = Vec6.shape ~n:rows ~t:1 ~d:1 ~h:1 ~w:1 ~c:cols
let vec_shape ~n = Vec6.shape ~n ~t:1 ~d:1 ~h:1 ~w:1 ~c:1
let state_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:batch ~c:k
let seq_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:seq ~w:batch ~c:isz

let params : Lstm.Lstm.params =
  { hidden_size = k; input_size = isz; batch_first = false }

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

  let params : Lstm.Lstm.params =
    { hidden_size = k; input_size = isz; batch_first = false }

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

(* Q=1, R=2: one bidirectional layer, time-first. Distinct forward/reverse
   weights and asymmetric per-timestep input data (lstm-plan.md §7's
   "Bidirectional ordering" row) make a forward/reverse mixup, or reading
   the reverse final state as the reverse half of [output[L-1,:]] instead
   of original time zero (§5's own warning), visible. *)
module Bidirectional = struct
  let k = 2
  let isz = 2
  let seq = 3
  let batch = 1
  let mat_shape ~rows ~cols = Vec6.shape ~n:rows ~t:1 ~d:1 ~h:1 ~w:1 ~c:cols
  let vec_shape ~n = Vec6.shape ~n ~t:1 ~d:1 ~h:1 ~w:1 ~c:1
  let state_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:batch ~c:k
  let seq_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:seq ~w:batch ~c:isz

  let weight_hh_f =
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

  let weight_ih_f =
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

  let bias_hh_f = [| 0.1; -0.1; 0.2; 0.05; -0.2; 0.1; 0.05; -0.05 |]
  let bias_ih_f = [| 0.05; 0.1; -0.1; 0.2; 0.1; -0.05; -0.1; 0.05 |]

  let weight_hh_r =
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

  let weight_ih_r =
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

  let bias_hh_r = [| -0.05; 0.1; 0.05; -0.1; 0.15; -0.05; 0.1; 0.2 |]
  let bias_ih_r = [| 0.1; -0.05; 0.2; 0.05; -0.1; 0.1; -0.05; 0.15 |]
  let h0 = [| 0.3; -0.4; 0.1; 0.2 |] (* [forward; reverse] *)
  let c0 = [| 0.5; 0.1; -0.2; 0.3 |]
  let input = [| 1.0; -0.5; 0.2; 0.7; -0.3; 0.4 |] (* asymmetric across t *)

  let params : Lstm.Lstm.params =
    { hidden_size = k; input_size = isz; batch_first = false }

  let graph =
    Err.or_raise ~pp_error:Graph_builder.pp_error
      Graph_builder.(
        build ~name:"lstm_bidir" ~outputs:(fun (out, hn, cn) -> [ out; hn; cn ])
        @@
        let* input_id = input ~shape:seq_shape ~name:"input" () in
        let* wihf =
          input ~shape:(mat_shape ~rows:(4 * k) ~cols:isz) ~name:"wihf" ()
        in
        let* whhf =
          input ~shape:(mat_shape ~rows:(4 * k) ~cols:k) ~name:"whhf" ()
        in
        let* bihf = input ~shape:(vec_shape ~n:(4 * k)) ~name:"bihf" () in
        let* bhhf = input ~shape:(vec_shape ~n:(4 * k)) ~name:"bhhf" () in
        let* wihr =
          input ~shape:(mat_shape ~rows:(4 * k) ~cols:isz) ~name:"wihr" ()
        in
        let* whhr =
          input ~shape:(mat_shape ~rows:(4 * k) ~cols:k) ~name:"whhr" ()
        in
        let* bihr = input ~shape:(vec_shape ~n:(4 * k)) ~name:"bihr" () in
        let* bhhr = input ~shape:(vec_shape ~n:(4 * k)) ~name:"bhhr" () in
        let* h0_id = input ~shape:state_shape ~name:"h0" () in
        let* c0_id = input ~shape:state_shape ~name:"c0" () in
        let layer : Lstm.Lstm.Layer.t =
          {
            forward =
              { weight_ih = wihf; weight_hh = whhf; bias = Some (bihf, bhhf) };
            reverse =
              Some
                { weight_ih = wihr; weight_hh = whhr; bias = Some (bihr, bhhr) };
          }
        in
        Graph_builder.lstm params ~input:input_id ~layers:[ layer ] ~h0:h0_id
          ~c0:c0_id ())

  let input_tensors =
    [
      tensor_of_array seq_shape input;
      tensor_of_array (mat_shape ~rows:(4 * k) ~cols:isz) weight_ih_f;
      tensor_of_array (mat_shape ~rows:(4 * k) ~cols:k) weight_hh_f;
      tensor_of_array (vec_shape ~n:(4 * k)) bias_ih_f;
      tensor_of_array (vec_shape ~n:(4 * k)) bias_hh_f;
      tensor_of_array (mat_shape ~rows:(4 * k) ~cols:isz) weight_ih_r;
      tensor_of_array (mat_shape ~rows:(4 * k) ~cols:k) weight_hh_r;
      tensor_of_array (vec_shape ~n:(4 * k)) bias_ih_r;
      tensor_of_array (vec_shape ~n:(4 * k)) bias_hh_r;
      tensor_of_array state_shape h0;
      tensor_of_array state_shape c0;
    ]

  let sigmoid x = 1. /. (1. +. exp (-.x))
  let get2 arr ~cols row col = arr.((row * cols) + col)

  let gate ~weight_hh ~weight_ih ~bias_hh ~bias_ih r j ~h ~inp =
    let row = (r * k) + j in
    let hh = ref 0. in
    for jj = 0 to k - 1 do
      hh := !hh +. (get2 weight_hh ~cols:k row jj *. h.(jj))
    done;
    let ih = ref 0. in
    for jj = 0 to isz - 1 do
      ih := !ih +. (get2 weight_ih ~cols:isz row jj *. inp.(jj))
    done;
    !hh +. !ih +. bias_hh.(row) +. bias_ih.(row)

  let step ~weight_hh ~weight_ih ~bias_hh ~bias_ih h c inp =
    let next_h = Array.make k 0. and next_c = Array.make k 0. in
    for j = 0 to k - 1 do
      let i =
        sigmoid (gate ~weight_hh ~weight_ih ~bias_hh ~bias_ih 0 j ~h ~inp)
      in
      let f =
        sigmoid (gate ~weight_hh ~weight_ih ~bias_hh ~bias_ih 1 j ~h ~inp)
      in
      let g = tanh (gate ~weight_hh ~weight_ih ~bias_hh ~bias_ih 2 j ~h ~inp) in
      let o =
        sigmoid (gate ~weight_hh ~weight_ih ~bias_hh ~bias_ih 3 j ~h ~inp)
      in
      next_c.(j) <- (f *. c.(j)) +. (i *. g);
      next_h.(j) <- o *. tanh next_c.(j)
    done;
    (next_h, next_c)

  (* Forward: original time t = s. Reverse: original time t = seq-1-s.
     [out] is indexed by ORIGINAL time regardless of direction. *)
  let run_direction ~weight_hh ~weight_ih ~bias_hh ~bias_ih ~h0 ~c0 ~reverse =
    let h = ref h0 and c = ref c0 in
    let out = Array.make (seq * k) 0. in
    for s = 0 to seq - 1 do
      let t = if reverse then seq - 1 - s else s in
      let inp = Array.sub input (t * isz) isz in
      let nh, nc = step ~weight_hh ~weight_ih ~bias_hh ~bias_ih !h !c inp in
      for j = 0 to k - 1 do
        out.((t * k) + j) <- nh.(j)
      done;
      h := nh;
      c := nc
    done;
    (out, !h, !c)

  let reference () =
    let h0_f = [| h0.(0); h0.(1) |] and c0_f = [| c0.(0); c0.(1) |] in
    let h0_r = [| h0.(2); h0.(3) |] and c0_r = [| c0.(2); c0.(3) |] in
    let fwd_out, h_n_f, c_n_f =
      run_direction ~weight_hh:weight_hh_f ~weight_ih:weight_ih_f
        ~bias_hh:bias_hh_f ~bias_ih:bias_ih_f ~h0:h0_f ~c0:c0_f ~reverse:false
    in
    let rev_out, h_n_r, c_n_r =
      run_direction ~weight_hh:weight_hh_r ~weight_ih:weight_ih_r
        ~bias_hh:bias_hh_r ~bias_ih:bias_ih_r ~h0:h0_r ~c0:c0_r ~reverse:true
    in
    let out = Array.make (seq * 2 * k) 0. in
    for t = 0 to seq - 1 do
      for j = 0 to k - 1 do
        out.((t * 2 * k) + j) <- fwd_out.((t * k) + j);
        out.((t * 2 * k) + k + j) <- rev_out.((t * k) + j)
      done
    done;
    (out, [| h_n_f; h_n_r |], [| c_n_f; c_n_r |])

  let%expect_test
      "a bidirectional LSTM graph agrees with a plain OCaml reference on all \
       three outputs" =
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
    for t = 0 to seq - 1 do
      for c = 0 to (2 * k) - 1 do
        let p = Tensor.read out_t (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:t ~w:0 ~c) in
        max_diff :=
          Float.max !max_diff (Float.abs (p -. expected_out.((t * 2 * k) + c)))
      done
    done;
    for dir = 0 to 1 do
      for j = 0 to k - 1 do
        let hp =
          Tensor.read hn_t (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:dir ~w:0 ~c:j)
        in
        let cp =
          Tensor.read cn_t (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:dir ~w:0 ~c:j)
        in
        max_diff :=
          Float.max !max_diff (Float.abs (hp -. expected_hn.(dir).(j)));
        max_diff :=
          Float.max !max_diff (Float.abs (cp -. expected_cn.(dir).(j)))
      done
    done;
    Fmt.pr "max_abs_diff=%g@." !max_diff;
    [%expect {| max_abs_diff=1.07743e-08 |}]
end

let%expect_test
    "a nonuniform direction count across layers is rejected, not silently \
     ignored" =
  let build () =
    Graph_builder.(
      build ~name:"lstm_nonuniform" ~outputs:(fun (out, _, _) -> [ out ])
      @@
      let* input_id = input ~shape:seq_shape ~name:"input" () in
      let* wih0 =
        input ~shape:(mat_shape ~rows:(4 * k) ~cols:isz) ~name:"wih0" ()
      in
      let* whh0 =
        input ~shape:(mat_shape ~rows:(4 * k) ~cols:k) ~name:"whh0" ()
      in
      let* wih1 =
        input ~shape:(mat_shape ~rows:(4 * k) ~cols:(2 * k)) ~name:"wih1" ()
      in
      let* whh1 =
        input ~shape:(mat_shape ~rows:(4 * k) ~cols:k) ~name:"whh1" ()
      in
      let* h0_id =
        input
          ~shape:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:batch ~c:k)
          ~name:"h0" ()
      in
      let* c0_id =
        input
          ~shape:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:3 ~w:batch ~c:k)
          ~name:"c0" ()
      in
      let layer0 : Lstm.Lstm.Layer.t =
        {
          forward = { weight_ih = wih0; weight_hh = whh0; bias = None };
          reverse = Some { weight_ih = wih0; weight_hh = whh0; bias = None };
        }
      in
      let layer1 : Lstm.Lstm.Layer.t =
        {
          forward = { weight_ih = wih1; weight_hh = whh1; bias = None };
          reverse = None;
        }
      in
      Graph_builder.lstm params ~input:input_id ~layers:[ layer0; layer1 ]
        ~h0:h0_id ~c0:c0_id ())
  in
  (match build () with
  | Ok _ -> Fmt.pr "unexpectedly built@."
  | Error e -> Fmt.pr "%a@." Graph_builder.pp_error (Err.Error.kind e));
  [%expect {| lstm: every layer must have a reverse direction, or none must |}]

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

let%expect_test "a non-positive hidden_size is rejected, not silently ignored" =
  let bad_params : Lstm.Lstm.params =
    { hidden_size = 0; input_size = isz; batch_first = false }
  in
  let build () =
    Graph_builder.(
      build ~name:"lstm_zero_hidden" ~outputs:(fun (out, _, _) -> [ out ])
      @@
      (* Non-positive [hidden_size] is rejected before any operand shape is
         examined, so these shapes are arbitrary valid placeholders, not
         shapes consistent with [bad_params]. *)
      let* input_id = input ~shape:seq_shape ~name:"input" () in
      let* weight_ih_id =
        input ~shape:(mat_shape ~rows:1 ~cols:isz) ~name:"weight_ih" ()
      in
      let* weight_hh_id =
        input ~shape:(mat_shape ~rows:1 ~cols:1) ~name:"weight_hh" ()
      in
      let* h0_id =
        input
          ~shape:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:batch ~c:1)
          ~name:"h0" ()
      in
      let* c0_id =
        input
          ~shape:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:batch ~c:1)
          ~name:"c0" ()
      in
      let layer : Lstm.Lstm.Layer.t =
        {
          forward =
            { weight_ih = weight_ih_id; weight_hh = weight_hh_id; bias = None };
          reverse = None;
        }
      in
      Graph_builder.lstm bad_params ~input:input_id ~layers:[ layer ] ~h0:h0_id
        ~c0:c0_id ())
  in
  (match build () with
  | Ok _ -> Fmt.pr "unexpectedly built@."
  | Error e -> Fmt.pr "%a@." Graph_builder.pp_error (Err.Error.kind e));
  [%expect {| lstm: hidden_size must be positive, got 0 |}]
