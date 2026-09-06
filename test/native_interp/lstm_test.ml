(* torch.ops.aten.lstm.input through the serialized path (project step 14 /
   Native_interp_lower_recurrent). Structure only -- numeric agreement with
   ATen lives in test/native_bridge/lstm_test.ml; this file's own subject is
   the use-analysis-based [Discard] routing ([ctx.reads]) that the isolated
   ATen bridge has no context to do, plus the typed rejections shared with
   it via [Lstm.Lstm.Reject]. Single layer/direction, time-first, k=2,
   isz=2, seq=3, batch=1 throughout -- small enough to read the whole graph
   dump, matching layer_norm_test.ml's "x" naming convention (the [program]
   helper's one built-in User_input is always called "x", regardless of
   what the node's own argument name is). *)

open Programs

let lstm_node ~hx ~params ~has_biases ~num_layers ~dropout ~train ~bidirectional
    ~batch_first ~outs =
  let bool_arg name b =
    jstr {|,{"name":"%s","arg":{"as_bool":%b},"kind":1}|} name b
  in
  let int_arg name i =
    jstr {|,{"name":"%s","arg":{"as_int":%d},"kind":1}|} name i
  in
  let float_arg name f =
    jstr {|,{"name":"%s","arg":{"as_float":%.12g},"kind":1}|} name f
  in
  jstr
    {|{"target":"torch.ops.aten.lstm.input","inputs":[{"name":"input","arg":%s,"kind":1},{"name":"hx","arg":%s,"kind":1},{"name":"params","arg":%s,"kind":1}%s%s%s%s%s%s],"outputs":[%s],"metadata":{}}|}
    (as_tensor "x") (as_tensors hx) (as_tensors params)
    (bool_arg "has_biases" has_biases)
    (int_arg "num_layers" num_layers)
    (float_arg "dropout" dropout)
    (bool_arg "train" train)
    (bool_arg "bidirectional" bidirectional)
    (bool_arg "batch_first" batch_first)
    (as_tensors outs)

let relu_node ~self ~out =
  jstr
    {|{"target":"torch.ops.aten.relu.default","inputs":[{"name":"self","arg":%s,"kind":1}],"outputs":[%s],"metadata":{}}|}
    (as_tensor self) (as_tensor out)

let k = 2 (* hidden_size *)
let isz = 2 (* input_size *)
let seq = 3
let batch = 1

let captured =
  [
    ("h0", tensor_meta [ 1; batch; k ]);
    ("c0", tensor_meta [ 1; batch; k ]);
    ("wih", tensor_meta [ 4 * k; isz ]);
    ("whh", tensor_meta [ 4 * k; k ]);
    ("bih", tensor_meta [ 4 * k ]);
    ("bhh", tensor_meta [ 4 * k ]);
  ]

let single_layer ?(reads = []) ~outs () =
  program ~x_sizes:[ seq; batch; isz ] ~extra_tensor_values:captured
    ~params:(List.map fst captured)
    ~nodes:
      (lstm_node ~hx:[ "h0"; "c0" ]
         ~params:[ "wih"; "whh"; "bih"; "bhh" ]
         ~has_biases:true ~num_layers:1 ~dropout:0.0 ~train:false
         ~bidirectional:false ~batch_first:false ~outs
      :: List.map (fun (self, out) -> relu_node ~self ~out) reads)
    ~graph_outputs:[ as_tensor "out" ]
    ()

let dump label json =
  Format.printf "%s@." label;
  match lower json with
  | Error e -> Format.printf "  %a@." Native_interp.pp_error (Err.Error.kind e)
  | Ok l -> Format.printf "%a@." Graph_ir.pp l.Pt2_native_graph.graph

(* Neither [h_n] nor [c_n] is read anywhere (not by a node, not as a graph
   output): both are routed to [Discard], but [output] stays a normal
   bound edge -- the same "materialise, then mark dead" treatment
   max_pool2d_with_indices' indices output gets, generalized here to a
   PER-GRAPH decision instead of a fixed per-op one. *)
let%expect_test "lstm.input discards both state outputs when neither is read" =
  dump "both dead:" (single_layer ~outs:[ "out"; "hn"; "cn" ] ());
  [%expect
    {|
    both dead:
    graph
    inputs:
      [t0 f32 [H=3 W=1 C=2] ->[n4], t1 f32 [C=2] ->[n4] constant,
       t2 f32 [C=2] ->[n4] constant, t3 f32 [W=8 C=2] ->[n0] constant,
       t4 f32 [W=8 C=2] ->[n1] constant, t5 f32 [C=8] ->[n2] constant,
       t6 f32 [C=8] ->[n3] constant]
    nodes:
      group g1 torch.ops.aten.lstm.input:
        n0: [t7 f32 [N=8 T=1 D=1 H=1 W=1 C=2] ->[n4]] =
          permute x=t3 perm=[N<-W, W<-N]
        n1: [t8 f32 [N=8 T=1 D=1 H=1 W=1 C=2] ->[n4]] =
          permute x=t4 perm=[N<-W, W<-N]
        n2: [t9 f32 [N=8 T=1 D=1 H=1 W=1 C=1] ->[n4]] =
          permute x=t5 perm=[N<-C, C<-N]
        n3: [t10 f32 [N=8 T=1 D=1 H=1 W=1 C=1] ->[n4]] =
          permute x=t6 perm=[N<-C, C<-N]
        n4: [t11 f32 [H=3 W=1 C=2], t12 f32 [C=2] ->[n5], t13 f32 [C=2] ->[n6]] =
          lstm
            input=t0
            layers=[{forward={weight_ih=t7 <-n0;
                               weight_hh=t8 <-n1;
                               bias=(t9 <-n2,t10 <-n3)};
                      reverse=none}]
            h0=t1
            c0=t2
            params={hidden_size=2; input_size=2; batch_first=false}
        n5: [] = discard x=t12 <-n4
        n6: [] = discard x=t13 <-n4
    outputs: [t11 f32 [H=3 W=1 C=2] <-n4] |}]

(* [h_n] is read by a downstream node; [c_n] is not. Only [c_n] is routed to
   [Discard] -- the decision is per-output, not all-or-nothing for the
   node. *)
let%expect_test
    "lstm.input keeps a live state output bound, discards the dead one" =
  dump "hn live, cn dead:"
    (single_layer ~reads:[ ("hn", "r") ] ~outs:[ "out"; "hn"; "cn" ] ());
  [%expect
    {|
    hn live, cn dead:
    graph
    inputs:
      [t0 f32 [H=3 W=1 C=2] ->[n4], t1 f32 [C=2] ->[n4] constant,
       t2 f32 [C=2] ->[n4] constant, t3 f32 [W=8 C=2] ->[n0] constant,
       t4 f32 [W=8 C=2] ->[n1] constant, t5 f32 [C=8] ->[n2] constant,
       t6 f32 [C=8] ->[n3] constant]
    nodes:
      group g1 torch.ops.aten.lstm.input:
        n0: [t7 f32 [N=8 T=1 D=1 H=1 W=1 C=2] ->[n4]] =
          permute x=t3 perm=[N<-W, W<-N]
        n1: [t8 f32 [N=8 T=1 D=1 H=1 W=1 C=2] ->[n4]] =
          permute x=t4 perm=[N<-W, W<-N]
        n2: [t9 f32 [N=8 T=1 D=1 H=1 W=1 C=1] ->[n4]] =
          permute x=t5 perm=[N<-C, C<-N]
        n3: [t10 f32 [N=8 T=1 D=1 H=1 W=1 C=1] ->[n4]] =
          permute x=t6 perm=[N<-C, C<-N]
        n4: [t11 f32 [H=3 W=1 C=2], t12 f32 [C=2] ->[n6], t13 f32 [C=2] ->[n5]] =
          lstm
            input=t0
            layers=[{forward={weight_ih=t7 <-n0;
                               weight_hh=t8 <-n1;
                               bias=(t9 <-n2,t10 <-n3)};
                      reverse=none}]
            h0=t1
            c0=t2
            params={hidden_size=2; input_size=2; batch_first=false}
        n5: [] = discard x=t13 <-n4
      n6: [t14 f32 [C=2]] = relu x=t12 <-n4
    outputs: [t11 f32 [H=3 W=1 C=2] <-n4] |}]

(* A graph OUTPUT is a read too, exactly [native_layer_norm]'s own
   [Live_layer_norm_stats] check (layer_norm_test.ml). *)
let%expect_test "lstm.input treats a graph output as a live read" =
  let json =
    program ~x_sizes:[ seq; batch; isz ] ~extra_tensor_values:captured
      ~params:(List.map fst captured)
      ~nodes:
        [
          lstm_node ~hx:[ "h0"; "c0" ]
            ~params:[ "wih"; "whh"; "bih"; "bhh" ]
            ~has_biases:true ~num_layers:1 ~dropout:0.0 ~train:false
            ~bidirectional:false ~batch_first:false ~outs:[ "out"; "hn"; "cn" ];
        ]
      ~graph_outputs:[ as_tensor "out"; as_tensor "hn" ]
      ()
  in
  dump "hn is a graph output:" json;
  [%expect
    {|
    hn is a graph output:
    graph
    inputs:
      [t0 f32 [H=3 W=1 C=2] ->[n4], t1 f32 [C=2] ->[n4] constant,
       t2 f32 [C=2] ->[n4] constant, t3 f32 [W=8 C=2] ->[n0] constant,
       t4 f32 [W=8 C=2] ->[n1] constant, t5 f32 [C=8] ->[n2] constant,
       t6 f32 [C=8] ->[n3] constant]
    nodes:
      group g1 torch.ops.aten.lstm.input:
        n0: [t7 f32 [N=8 T=1 D=1 H=1 W=1 C=2] ->[n4]] =
          permute x=t3 perm=[N<-W, W<-N]
        n1: [t8 f32 [N=8 T=1 D=1 H=1 W=1 C=2] ->[n4]] =
          permute x=t4 perm=[N<-W, W<-N]
        n2: [t9 f32 [N=8 T=1 D=1 H=1 W=1 C=1] ->[n4]] =
          permute x=t5 perm=[N<-C, C<-N]
        n3: [t10 f32 [N=8 T=1 D=1 H=1 W=1 C=1] ->[n4]] =
          permute x=t6 perm=[N<-C, C<-N]
        n4: [t11 f32 [H=3 W=1 C=2], t12 f32 [C=2], t13 f32 [C=2] ->[n5]] =
          lstm
            input=t0
            layers=[{forward={weight_ih=t7 <-n0;
                               weight_hh=t8 <-n1;
                               bias=(t9 <-n2,t10 <-n3)};
                      reverse=none}]
            h0=t1
            c0=t2
            params={hidden_size=2; input_size=2; batch_first=false}
        n5: [] = discard x=t13 <-n4
    outputs: [t11 f32 [H=3 W=1 C=2] <-n4, t12 f32 [C=2] <-n4] |}]

let show label json =
  Format.printf "%-26s" label;
  match lower json with
  | Error e -> Format.printf "%a@." Native_interp.pp_error (Err.Error.kind e)
  | Ok _ -> Format.printf "ok@."

let%expect_test "lstm.input rejects train=true" =
  show "train=true:"
    (program ~x_sizes:[ seq; batch; isz ] ~extra_tensor_values:captured
       ~params:(List.map fst captured)
       ~nodes:
         [
           lstm_node ~hx:[ "h0"; "c0" ]
             ~params:[ "wih"; "whh"; "bih"; "bhh" ]
             ~has_biases:true ~num_layers:1 ~dropout:0.0 ~train:true
             ~bidirectional:false ~batch_first:false ~outs:[ "out"; "hn"; "cn" ];
         ]
       ~graph_outputs:[ as_tensor "out" ]
       ());
  [%expect
    {| train=true:               malformed PT2 graph: lstm: train=true is not supported |}]

let%expect_test "lstm.input rejects a malformed hx list" =
  show "hx has 1:"
    (program ~x_sizes:[ seq; batch; isz ] ~extra_tensor_values:captured
       ~params:(List.map fst captured)
       ~nodes:
         [
           lstm_node ~hx:[ "h0" ]
             ~params:[ "wih"; "whh"; "bih"; "bhh" ]
             ~has_biases:true ~num_layers:1 ~dropout:0.0 ~train:false
             ~bidirectional:false ~batch_first:false ~outs:[ "out"; "hn"; "cn" ];
         ]
       ~graph_outputs:[ as_tensor "out" ]
       ());
  [%expect
    {| hx has 1:                 malformed PT2 graph: lstm: hx must have exactly 2 tensors (h0, c0), got 1 |}]

let%expect_test "lstm.input rejects a params list of the wrong arity" =
  show "bidirectional needs 8, got 4:"
    (program ~x_sizes:[ seq; batch; isz ] ~extra_tensor_values:captured
       ~params:(List.map fst captured)
       ~nodes:
         [
           lstm_node ~hx:[ "h0"; "c0" ]
             ~params:[ "wih"; "whh"; "bih"; "bhh" ]
             ~has_biases:true ~num_layers:1 ~dropout:0.0 ~train:false
             ~bidirectional:true ~batch_first:false ~outs:[ "out"; "hn"; "cn" ];
         ]
       ~graph_outputs:[ as_tensor "out" ]
       ());
  [%expect
    {|
    bidirectional needs 8, got 4:malformed PT2 graph: lstm: params has 4 tensors, expected 8 for this num_layers/bidirectional/has_biases configuration |}]

(* [has_biases]/[num_layers]/[train]/[bidirectional]/[batch_first] have no
   schema default -- omitting one is [`Missing_arg], not a silent false/0,
   the same fix [float_arg]'s own comment gives for [eps]. *)
let%expect_test
    "lstm.input requires has_biases/num_layers/train/bidirectional/batch_first"
    =
  let lstm_node_missing ~omit =
    let bool_arg name b =
      if name = omit then ""
      else jstr {|,{"name":"%s","arg":{"as_bool":%b},"kind":1}|} name b
    in
    let int_arg name i =
      if name = omit then ""
      else jstr {|,{"name":"%s","arg":{"as_int":%d},"kind":1}|} name i
    in
    jstr
      {|{"target":"torch.ops.aten.lstm.input","inputs":[{"name":"input","arg":%s,"kind":1},{"name":"hx","arg":%s,"kind":1},{"name":"params","arg":%s,"kind":1}%s%s,{"name":"dropout","arg":{"as_float":0.0},"kind":1}%s%s%s],"outputs":[%s],"metadata":{}}|}
      (as_tensor "x")
      (as_tensors [ "h0"; "c0" ])
      (as_tensors [ "wih"; "whh"; "bih"; "bhh" ])
      (bool_arg "has_biases" true)
      (int_arg "num_layers" 1) (bool_arg "train" false)
      (bool_arg "bidirectional" false)
      (bool_arg "batch_first" false)
      (as_tensors [ "out"; "hn"; "cn" ])
  in
  let one_node ~omit =
    program ~x_sizes:[ seq; batch; isz ] ~extra_tensor_values:captured
      ~params:(List.map fst captured)
      ~nodes:[ lstm_node_missing ~omit ]
      ~graph_outputs:[ as_tensor "out" ]
      ()
  in
  show "omit has_biases:" (one_node ~omit:"has_biases");
  show "omit num_layers:" (one_node ~omit:"num_layers");
  show "omit train:" (one_node ~omit:"train");
  show "omit bidirectional:" (one_node ~omit:"bidirectional");
  show "omit batch_first:" (one_node ~omit:"batch_first");
  [%expect
    {|
    omit has_biases:          malformed PT2 graph: torch.ops.aten.lstm.input: missing argument "has_biases"
    omit num_layers:          malformed PT2 graph: torch.ops.aten.lstm.input: missing argument "num_layers"
    omit train:               malformed PT2 graph: torch.ops.aten.lstm.input: missing argument "train"
    omit bidirectional:       malformed PT2 graph: torch.ops.aten.lstm.input: missing argument "bidirectional"
    omit batch_first:         malformed PT2 graph: torch.ops.aten.lstm.input: missing argument "batch_first" |}]

(* [materialized_output_names] falls to the default catch-all (all three
   serialized names), so a declared arity that disagrees with the op's real
   3-output arity is caught by the generic [Output_arity] check in
   [Native_interp_lower], the same way it is for every other op -- unlike
   [native_layer_norm], lstm needs no arm-local arity check of its own. *)
let%expect_test "lstm.input checks its output arity generically" =
  show "two outputs:" (single_layer ~outs:[ "out"; "hn" ] ());
  show "four outputs:" (single_layer ~outs:[ "out"; "hn"; "cn"; "extra" ] ());
  [%expect
    {|
    two outputs:              malformed PT2 graph: torch.ops.aten.lstm.input declares 2 outputs but produces 3
    four outputs:             malformed PT2 graph: torch.ops.aten.lstm.input declares 4 outputs but produces 3 |}]
