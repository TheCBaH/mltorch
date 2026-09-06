(* End-to-end coverage for the WIP `Lstm` Native op (project step 13 / M3):
   builds one tiny single-layer, single-direction, time-first graph through
   the real Graph_builder/Region_computation/Eval_direct pipeline and checks
   all three outputs (live together, per M3's own exit condition) against a
   plain OCaml reference -- the same cross-check discipline
   test/native/lstm_region_prototype_test.ml used before any graph
   registration existed. Stacked layers, bidirectionality and the M3b
   rejection tests live in lstm_graph_layers_test.ml (kept separate to
   stay under the tracked file-size ceiling); batch-first layout remains
   unimplemented. *)

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

let params : Lstm.Lstm.params =
  { hidden_size = k; input_size = isz; batch_first = false }

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
