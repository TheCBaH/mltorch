(* Batch-first layout coverage for the WIP `Lstm` Native op (project step
   13 / M3b, completing it): input/output on [H=batch, W=seq, C=channel]
   instead of the time-first [H=seq, W=batch, C=channel] the other
   lstm_graph_*_test.ml files use; h0/c0/h_n/c_n stay [H=layer*R+direction,
   W=batch, C=hidden] regardless of layout (lstm-plan.md §2). Uses
   batch=2 (not 1) so a layout bug that mixed up which axis is "batch"
   would blend two batch elements' values together, not merely produce
   the right numbers at swapped-but-otherwise-consistent coordinates. *)

let k = 2
let isz = 2
let seq = 2
let batch = 2
let mat_shape ~rows ~cols = Vec6.shape ~n:rows ~t:1 ~d:1 ~h:1 ~w:1 ~c:cols
let vec_shape ~n = Vec6.shape ~n ~t:1 ~d:1 ~h:1 ~w:1 ~c:1
let state_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:batch ~c:k

(* Batch-first: H=batch, W=seq, C=channel. *)
let seq_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:batch ~w:seq ~c:isz

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

(* [h0]/[c0], flat by batch (W=batch, C=k): distinct per batch element so a
   batch mixup is visible in the final states too. *)
let h0 = [| 0.3; -0.4; 0.1; 0.2 |]
let c0 = [| 0.5; 0.1; -0.2; 0.3 |]

(* Flat in BATCH-FIRST dense order (H=batch outermost, W=seq, C=isz
   innermost): input.(((b*seq)+t)*isz)+i]. Distinct per batch element. *)
let input = [| 1.0; -0.5; 0.2; 0.7; -0.3; 0.4; 0.6; -0.2 |]

let params : Lstm.Lstm.params =
  { hidden_size = k; input_size = isz; batch_first = true }

let graph =
  Err.or_raise ~pp_error:Graph_builder.pp_error
    Graph_builder.(
      build ~name:"lstm_batch_first" ~outputs:(fun (out, hn, cn) ->
          [ out; hn; cn ])
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

let sigmoid x = 1. /. (1. +. exp (-.x))
let get2 arr ~cols row col = arr.((row * cols) + col)

(* One batch element's forward recurrence, independent of every other
   (LSTM never mixes batch elements) -- [get_input t] reads that batch's
   own per-timestep input. *)
let run_one ~h0 ~c0 ~get_input =
  let gate r j ~h ~inp =
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
  in
  let h = ref h0 and c = ref c0 in
  let out = Array.make (seq * k) 0. in
  for t = 0 to seq - 1 do
    let inp = get_input t in
    let next_h = Array.make k 0. and next_c = Array.make k 0. in
    for j = 0 to k - 1 do
      let i = sigmoid (gate 0 j ~h:!h ~inp) in
      let f = sigmoid (gate 1 j ~h:!h ~inp) in
      let g = tanh (gate 2 j ~h:!h ~inp) in
      let o = sigmoid (gate 3 j ~h:!h ~inp) in
      next_c.(j) <- (f *. !c.(j)) +. (i *. g);
      next_h.(j) <- o *. tanh next_c.(j)
    done;
    for j = 0 to k - 1 do
      out.((t * k) + j) <- next_h.(j)
    done;
    h := next_h;
    c := next_c
  done;
  (out, !h, !c)

let reference () =
  Array.init batch (fun b ->
      run_one
        ~h0:[| h0.((b * k) + 0); h0.((b * k) + 1) |]
        ~c0:[| c0.((b * k) + 0); c0.((b * k) + 1) |]
        ~get_input:(fun t ->
          [|
            input.((((b * seq) + t) * isz) + 0);
            input.((((b * seq) + t) * isz) + 1);
          |]))

let%expect_test
    "a batch-first LSTM graph (batch=2) agrees with a plain OCaml reference on \
     all three outputs" =
  let inputs = List.combine graph.Graph_ir.Graph.inputs input_tensors in
  let env =
    Err.or_raise ~pp_error:Eval_direct.pp_error (Eval_direct.run graph ~inputs)
  in
  let out_id, hn_id, cn_id =
    match graph.Graph_ir.Graph.outputs with
    | [ out; hn; cn ] -> (out, hn, cn)
    | _ -> assert false
  in
  let out_t = Tensor_id.Map.find out_id env in
  let hn_t = Tensor_id.Map.find hn_id env in
  let cn_t = Tensor_id.Map.find cn_id env in
  let per_batch = reference () in
  let max_diff = ref 0. in
  for b = 0 to batch - 1 do
    let expected_out, expected_hn, expected_cn = per_batch.(b) in
    for t = 0 to seq - 1 do
      for j = 0 to k - 1 do
        (* Batch-first output: H=batch, W=seq, C=channel. *)
        let p = Tensor.read out_t (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:b ~w:t ~c:j) in
        max_diff :=
          Float.max !max_diff (Float.abs (p -. expected_out.((t * k) + j)))
      done
    done;
    for j = 0 to k - 1 do
      (* h_n/c_n: H=layer*R+direction (=0), W=batch, C=hidden -- always,
         regardless of layout. *)
      let hp = Tensor.read hn_t (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:b ~c:j) in
      let cp = Tensor.read cn_t (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:b ~c:j) in
      max_diff := Float.max !max_diff (Float.abs (hp -. expected_hn.(j)));
      max_diff := Float.max !max_diff (Float.abs (cp -. expected_cn.(j)))
    done
  done;
  Fmt.pr "max_abs_diff=%g@." !max_diff;
  [%expect {| max_abs_diff=8.24738e-09 |}]
