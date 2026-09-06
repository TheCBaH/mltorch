(* dispatch: torch.ops.aten.lstm.input (project step 14 / Op_bridge half).
   [Op_bridge_recurrent] always exposes all three outputs (output, h_n, c_n)
   -- there is no broader-graph liveness context at this single-node level,
   unlike [Native_interp_lower_recurrent]'s use-analysis-based [Discard]
   routing (see lstm_importer_test.ml). *)

open Helpers

let k = 2 (* hidden_size *)
let isz = 2 (* input_size *)
let seq = 3
let batch = 1

let weight_hh =
  [
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
  ]

let weight_ih =
  [
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
  ]

let bias_hh = [ 0.1; -0.1; 0.2; 0.05; -0.2; 0.1; 0.05; -0.05 ]
let bias_ih = [ 0.05; 0.1; -0.1; 0.2; 0.1; -0.05; -0.1; 0.05 ]
let h0 = [ 0.3; -0.4 ]
let c0 = [ 0.5; 0.1 ]
let input = [ 1.0; -0.5; 0.2; 0.7; -0.3; 0.4 ]

let lstm_inputs ~has_biases ~num_layers ~train ~bidirectional ~batch_first =
  [
    in_tensor "input";
    in_tensors "hx" [ "h0"; "c0" ];
    in_tensors "params"
      (if has_biases then [ "weight_ih"; "weight_hh"; "bias_ih"; "bias_hh" ]
       else [ "weight_ih"; "weight_hh" ]);
    in_bool "has_biases" has_biases;
    in_int "num_layers" num_layers;
    in_float "dropout" 0.0;
    in_bool "train" train;
    in_bool "bidirectional" bidirectional;
    in_bool "batch_first" batch_first;
  ]

let bindings =
  [
    ("input", float_tensor [ seq; batch; isz ] input);
    ("h0", float_tensor [ 1; batch; k ] h0);
    ("c0", float_tensor [ 1; batch; k ] c0);
    ("weight_ih", float_tensor [ 4 * k; isz ] weight_ih);
    ("weight_hh", float_tensor [ 4 * k; k ] weight_hh);
    ("bias_ih", float_tensor [ 4 * k ] bias_ih);
    ("bias_hh", float_tensor [ 4 * k ] bias_hh);
  ]

(* The real oracle: ATen runs lstm.input, the native side runs the graph
   [Op_bridge_recurrent] builds through [Eval_direct], and
   [Verify.verify_node] compares all three outputs (this bridge always
   exposes all three, so the comparison is exact, not merely leading). *)
let verify_lstm ~inputs ~bindings =
  let env = List.fold_left (fun m (k, t) -> Sm.add k t m) Sm.empty bindings in
  let node =
    PT.Node.make "torch.ops.aten.lstm.input" inputs
      [ targ "out"; targ "hn"; targ "cn" ]
      Sm.empty None (Some "test")
  in
  match
    Interp_verify.dispatch ~verify:true ~ppf:Format.std_formatter env node
  with
  | Error e ->
      Format.printf "dispatch error: %a@." Interp_verify.pp_interp_error
        (Err.Error.kind e)
  | Ok _ -> print_string "aten and native agree\n"

let%expect_test
    "verify: lstm.input agrees with ATen (single layer/direction, time-first, \
     with biases)" =
  verify_lstm
    ~inputs:
      (lstm_inputs ~has_biases:true ~num_layers:1 ~train:false
         ~bidirectional:false ~batch_first:false)
    ~bindings;
  [%expect {| aten and native agree |}]

let%expect_test
    "verify: lstm.input agrees with ATen (single layer/direction, time-first, \
     without biases)" =
  verify_lstm
    ~inputs:
      (lstm_inputs ~has_biases:false ~num_layers:1 ~train:false
         ~bidirectional:false ~batch_first:false)
    ~bindings:
      (List.remove_assoc "bias_ih" (List.remove_assoc "bias_hh" bindings));
  [%expect {| aten and native agree |}]

(* --- typed rejections at the importing boundary (project step 14) --- *)

let dispatch_lstm_print ~inputs =
  dispatch_print ~target:"torch.ops.aten.lstm.input" ~bindings ~inputs
    ~noutputs:3

let%expect_test "dispatch: lstm.input rejects train=true" =
  dispatch_lstm_print
    ~inputs:
      (lstm_inputs ~has_biases:true ~num_layers:1 ~train:true
         ~bidirectional:false ~batch_first:false);
  [%expect {| error: lstm: train=true is not supported |}]

let%expect_test "dispatch: lstm.input rejects a malformed hx list" =
  dispatch_lstm_print
    ~inputs:
      [
        in_tensor "input";
        in_tensors "hx" [ "h0" ] (* only one -- must be exactly 2 *);
        in_tensors "params" [ "weight_ih"; "weight_hh"; "bias_ih"; "bias_hh" ];
        in_bool "has_biases" true;
        in_int "num_layers" 1;
        in_float "dropout" 0.0;
        in_bool "train" false;
        in_bool "bidirectional" false;
        in_bool "batch_first" false;
      ];
  [%expect {| error: lstm: hx must have exactly 2 tensors (h0, c0), got 1 |}]

let%expect_test "dispatch: lstm.input rejects a params list of the wrong arity"
    =
  dispatch_lstm_print
    ~inputs:
      [
        in_tensor "input";
        in_tensors "hx" [ "h0"; "c0" ];
        (* bidirectional=true needs 2 directions' worth (8 tensors with
           biases), but only one direction's 4 are given. *)
        in_tensors "params" [ "weight_ih"; "weight_hh"; "bias_ih"; "bias_hh" ];
        in_bool "has_biases" true;
        in_int "num_layers" 1;
        in_float "dropout" 0.0;
        in_bool "train" false;
        in_bool "bidirectional" true;
        in_bool "batch_first" false;
      ];
  [%expect
    {|
    error: lstm: params has 4 tensors, expected 8 for this num_layers/bidirectional/has_biases configuration |}]
