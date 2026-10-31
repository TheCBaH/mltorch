(* The mask vocabulary through the serialized path: comparisons, `__and__`,
   `where.ScalarOther`, `new_ones`, `tanh`, and `index.Tensor` with two live
   leading indices. Numeric agreement of the Native ops is in
   test/native/mask_ops_test.ml; here each target must lower to exactly one
   node with the right operands and, where the lowering reads metadata (the
   index ranks), the right payload. *)

open Programs

let meta ?(dtype = 7) sizes =
  jstr
    {|{"dtype":%d,"sizes":[%s],"requires_grad":false,"device":{"type":"cpu"},"strides":[{"as_int":1}],"storage_offset":{"as_int":0},"layout":7}|}
    dtype
    (String.concat "," (List.map (fun i -> jstr {|{"as_int":%d}|} i) sizes))

let tin name n = jstr {|{"name":"%s","arg":%s,"kind":1}|} name (as_tensor n)
let int_in name i = jstr {|{"name":"%s","arg":{"as_int":%d},"kind":1}|} name i

let float_in name f =
  jstr {|{"name":"%s","arg":{"as_float":%g},"kind":1}|} name f

let node target inputs out =
  jstr
    {|{"target":"torch.ops.aten.%s","inputs":[%s],"outputs":[%s],"metadata":{}}|}
    target (String.concat "," inputs) (as_tensor out)

(* [x] is the [2,3] float input; [p] a second [2,3] float parameter; [m] a
   [2,3] bool parameter. *)
let prog ?(extra = []) nodes out_sizes =
  program ~x_sizes:[ 2; 3 ] ~params:[ "p"; "m" ]
    ~extra_tensor_values:
      ([
         ("p", meta [ 2; 3 ]);
         ("m", meta ~dtype:12 [ 2; 3 ]);
         ("y", meta out_sizes);
       ]
      @ extra)
    ~nodes
    ~graph_outputs:[ as_tensor "y" ]
    ()

let dump label json =
  Format.printf "%s@." label;
  match lower json with
  | Error e -> Format.printf "  %a@." Native_interp.pp_error (Err.Error.kind e)
  | Ok l ->
      let g = l.Pt2_native_graph.graph in
      List.iter
        (fun (n : Graph_ir.node) ->
          Format.printf "  %a@." (Graph_ir.pp_op g) n.Graph_ir.Node.op)
        g.Graph_ir.Graph.nodes

let%expect_test "comparisons lower to one node each" =
  List.iter
    (fun (label, target, inputs) ->
      dump label (prog [ node target inputs "y" ] [ 2; 3 ]))
    [
      ("eq.Scalar", "eq.Scalar", [ tin "self" "x"; int_in "other" 1 ]);
      ("eq.Tensor", "eq.Tensor", [ tin "self" "x"; tin "other" "p" ]);
      ("ge.Scalar", "ge.Scalar", [ tin "self" "x"; int_in "other" 0 ]);
      ("gt.Scalar", "gt.Scalar", [ tin "self" "x"; float_in "other" 0.5 ]);
      ("le.Tensor", "le.Tensor", [ tin "self" "x"; tin "other" "p" ]);
      ("lt.Scalar", "lt.Scalar", [ tin "self" "x"; int_in "other" 2 ]);
      ("ne.Scalar", "ne.Scalar", [ tin "self" "x"; int_in "other" 1 ]);
      ("ne.Tensor", "ne.Tensor", [ tin "self" "x"; tin "other" "p" ]);
    ];
  [%expect
    {|
    eq.Scalar
      eq_scalar x=t0 scalar=1
    eq.Tensor
      eq_tensor a=t0 b=t1
    ge.Scalar
      ge_scalar x=t0 scalar=0
    gt.Scalar
      gt_scalar x=t0 scalar=0.5
    le.Tensor
      le_tensor a=t0 b=t1
    lt.Scalar
      lt_scalar x=t0 scalar=2
    ne.Scalar
      ne_scalar x=t0 scalar=1
    ne.Tensor
      ne_tensor a=t0 b=t1 |}]

let%expect_test "boolean and selection ops" =
  dump "__and__.Tensor"
    (prog
       [ node "__and__.Tensor" [ tin "self" "m"; tin "other" "m" ] "y" ]
       [ 2; 3 ]);
  dump "where.ScalarOther"
    (prog
       [
         node "where.ScalarOther"
           [
             tin "condition" "m";
             tin "self" "x";
             float_in "other" (-3.4028234663852886e+38);
           ]
           "y";
       ]
       [ 2; 3 ]);
  dump "tanh" (prog [ node "tanh.default" [ tin "self" "x" ] "y" ] [ 2; 3 ]);
  [%expect
    {|
    __and__.Tensor
      bitwise_and a=t2 b=t2
    where.ScalarOther
      where_scalar_other condition=t2 x=t0 scalar=-3.40282e+38
    tanh
      tanh x=t0 |}]

let new_ones_node
    ?(dtype = {|,{"name":"dtype","arg":{"as_scalar_type":12},"kind":1}|})
    ?(extra = "") size =
  jstr
    {|{"target":"torch.ops.aten.new_ones.default","inputs":[%s,{"name":"size","arg":{"as_ints":[%s]},"kind":1}%s%s],"outputs":[%s],"metadata":{}}|}
    (tin "self" "x") size dtype extra (as_tensor "y")

let%expect_test "new_ones: bool or float32, scalar or vector; nothing else" =
  let go label ?dtype ?extra size out =
    dump label (prog [ new_ones_node ?dtype ?extra size ] out)
  in
  go "bool scalar" "" [];
  go "float32 [2,3]"
    ~dtype:{|,{"name":"dtype","arg":{"as_scalar_type":7},"kind":1}|} "2,3"
    [ 2; 3 ];
  go "no dtype" ~dtype:"" "" [];
  go "int64" ~dtype:{|,{"name":"dtype","arg":{"as_scalar_type":5},"kind":1}|} ""
    [];
  go "pin_memory true"
    ~extra:{|,{"name":"pin_memory","arg":{"as_bool":true},"kind":1}|} "" [];
  [%expect
    {|
    bool scalar
      new_ones params={shape=[C=1]; fmt=bool}
    float32 [2,3]
      new_ones params={shape=[W=2 C=3]; fmt=f32}
    no dtype
      malformed PT2 graph: torch.ops.aten.new_ones.default: dtype is not supported
    int64
      malformed PT2 graph: torch.ops.aten.new_ones.default: dtype is not supported
    pin_memory true
      malformed PT2 graph: torch.ops.aten.new_ones.default.pin_memory is not a bool |}]

(* `mask[batch_idx, kv_idx]`: a [2,3] bool self, a rank-4 batch index and a
   rank-4 position index. The importer carries all three ranks. *)
let index_node indices =
  jstr
    {|{"target":"torch.ops.aten.index.Tensor","inputs":[%s,{"name":"indices","arg":%s,"kind":1}],"outputs":[%s],"metadata":{}}|}
    (tin "self" "m") indices (as_tensor "y")

let%expect_test "index.Tensor with two live leading indices lowers to one pair"
    =
  let long = 5 in
  let extra =
    [
      ("i0", meta ~dtype:long [ 1; 1; 1; 1 ]);
      ("i1", meta ~dtype:long [ 1; 1; 1; 3 ]);
    ]
  in
  let prog nodes =
    program ~x_sizes:[ 2; 3 ] ~params:[ "m"; "i0"; "i1" ]
      ~extra_tensor_values:
        ([
           ("m", meta ~dtype:12 [ 2; 3 ]); ("y", meta ~dtype:12 [ 1; 1; 1; 3 ]);
         ]
        @ extra)
      ~nodes
      ~graph_outputs:[ as_tensor "y" ]
      ()
  in
  dump "as_tensors pair" (prog [ index_node (as_tensors [ "i0"; "i1" ]) ]);
  dump "optional pair"
    (prog
       [
         index_node
           {|{"as_optional_tensors":[{"as_tensor":{"name":"i0"}},{"as_tensor":{"name":"i1"}}]}|};
       ]);
  dump "a single live entry is left to the single-gather arm"
    (prog
       [
         index_node
           {|{"as_optional_tensors":[{"as_none":true},{"as_tensor":{"name":"i1"}}]}|};
       ]);
  [%expect
    {|
    as_tensors pair
      index_pair
        self=t1
        index0=t2
        index1=t3
        params={self_rank=2 index0_rank=4 index1_rank=4}
    optional pair
      index_pair
        self=t1
        index0=t2
        index1=t3
        params={self_rank=2 index0_rank=4 index1_rank=4}
    a single live entry is left to the single-gather arm
      index.Tensor: a multi-axis index at axis C would overwrite self's own axis W, which must have extent 1 (got 2) |}]

(* An explicit float32 [dtype] on a reduction-family op changes no value in a
   float32 engine; any other dtype would, and is still refused. *)
let%expect_test "softmax.int accepts dtype=float32 and refuses the rest" =
  let softmax dtype =
    node "softmax.int"
      [
        tin "self" "x";
        int_in "dim" (-1);
        jstr {|{"name":"dtype","arg":{"as_scalar_type":%d},"kind":1}|} dtype;
      ]
      "y"
  in
  dump "float32" (prog [ softmax 7 ] [ 2; 3 ]);
  dump "float64" (prog [ softmax 8 ] [ 2; 3 ]);
  dump "int64" (prog [ softmax 5 ] [ 2; 3 ]);
  [%expect
    {|
    float32
      softmax x=t0 params={axis=C}
    float64
      malformed PT2 graph: torch.ops.aten.softmax.int: dtype is not supported
    int64
      malformed PT2 graph: torch.ops.aten.softmax.int: dtype is not supported |}]

let%expect_test "argmax and the int32 cast, as TinyCLIP's pooling writes them" =
  let argmax inputs = node "argmax.default" inputs "y" in
  dump "argmax dim=-1"
    (prog [ argmax [ tin "self" "x"; int_in "dim" (-1) ] ] [ 2 ]);
  dump "argmax dim=0, keepdim"
    (prog
       [
         argmax
           [
             tin "self" "x";
             int_in "dim" 0;
             {|{"name":"keepdim","arg":{"as_bool":true},"kind":1}|};
           ];
       ]
       [ 1; 3 ]);
  dump "argmax with no dim (the flattened form)"
    (prog [ argmax [ tin "self" "x" ] ] []);
  let to_copy dtype =
    node "_to_copy.default"
      [
        tin "self" "x";
        jstr {|{"name":"dtype","arg":{"as_scalar_type":%d},"kind":1}|} dtype;
      ]
      "y"
  in
  dump "to int32" (prog [ to_copy 4 ] [ 2; 3 ]);
  dump "to int16 is still refused" (prog [ to_copy 3 ] [ 2; 3 ]);
  [%expect
    {|
    argmax dim=-1
      argmax x=t0 params={axis=C; keepdim=false}
    argmax dim=0, keepdim
      argmax x=t0 params={axis=W; keepdim=true}
    argmax with no dim (the flattened form)
      malformed PT2 graph: torch.ops.aten.argmax.default: missing argument "dim"
    to int32
      to_copy x=t0 target=int
    to int16 is still refused
      malformed PT2 graph: torch.ops.aten._to_copy.default: dtype is not supported |}]

let%expect_test
    "exp, and t as a matrix transpose that is the identity below rank 2" =
  dump "exp" (prog [ node "exp.default" [ tin "self" "x" ] "y" ] [ 2; 3 ]);
  dump "t of a [2,3]"
    (prog [ node "t.default" [ tin "self" "x" ] "y" ] [ 3; 2 ]);
  [%expect
    {|
    exp
      exp x=t0
    t of a [2,3]
      permute x=t0 perm=[W<-C, C<-W] |}]

let%expect_test "the T5 bucket vocabulary lowers to one node each" =
  let long_meta = 5 in
  let extra =
    [
      ("a", meta ~dtype:long_meta [ 2; 3 ]);
      ("b", meta ~dtype:long_meta [ 2; 3 ]);
    ]
  in
  let prog nodes out =
    program ~x_sizes:[ 2; 3 ] ~params:[ "m"; "a"; "b" ]
      ~extra_tensor_values:
        ([ ("m", meta ~dtype:12 [ 2; 3 ]); ("y", meta ~dtype:long_meta out) ]
        @ extra)
      ~nodes
      ~graph_outputs:[ as_tensor "y" ]
      ()
  in
  dump "log" (prog [ node "log.default" [ tin "self" "x" ] "y" ] [ 2; 3 ]);
  dump "min.other on int64"
    (prog [ node "min.other" [ tin "self" "a"; tin "other" "b" ] "y" ] [ 2; 3 ]);
  dump "where.self on int64"
    (prog
       [
         node "where.self"
           [ tin "condition" "m"; tin "self" "a"; tin "other" "b" ]
           "y";
       ]
       [ 2; 3 ]);
  dump "full_like int64 15"
    (prog
       [
         node "full_like.default"
           [
             tin "self" "a";
             int_in "fill_value" 15;
             {|{"name":"pin_memory","arg":{"as_bool":false},"kind":1}|};
           ]
           "y";
       ]
       [ 2; 3 ]);
  dump "zeros_like"
    (prog [ node "zeros_like.default" [ tin "self" "a" ] "y" ] [ 2; 3 ]);
  dump "full_like with an explicit dtype is refused"
    (prog
       [
         node "full_like.default"
           [
             tin "self" "a";
             int_in "fill_value" 1;
             {|{"name":"dtype","arg":{"as_scalar_type":7},"kind":1}|};
           ]
           "y";
       ]
       [ 2; 3 ]);
  [%expect
    {|
    log
      log x=t0
    min.other on int64
      min_other a=t2 b=t3
    where.self on int64
      where_self condition=t1 x=t2 y=t3
    full_like int64 15
      full_like x=t2 value=15
    zeros_like
      full_like x=t2 value=0
    full_like with an explicit dtype is refused
      malformed PT2 graph: torch.ops.aten.full_like.default: dtype is not supported |}]

let%expect_test "log1p, and squeeze.default drops every extent-1 axis" =
  dump "log1p" (prog [ node "log1p.default" [ tin "self" "x" ] "y" ] [ 2; 3 ]);
  let squeeze_prog x_sizes out =
    program ~x_sizes
      ~extra_tensor_values:[ ("y", meta out) ]
      ~nodes:[ node "squeeze.default" [ tin "self" "x" ] "y" ]
      ~graph_outputs:[ as_tensor "y" ]
      ()
  in
  dump "[1,2,1,3] -> [2,3]" (squeeze_prog [ 1; 2; 1; 3 ] [ 2; 3 ]);
  dump "[1,1] -> scalar" (squeeze_prog [ 1; 1 ] []);
  dump "[2,3] unchanged" (squeeze_prog [ 2; 3 ] [ 2; 3 ]);
  [%expect
    {|
    log1p
      log1p x=t0
    [1,2,1,3] -> [2,3]
      reshape x=t0 params={shape=[W=2 C=3]}
    [1,1] -> scalar
      reshape x=t0 params={shape=[C=1]}
    [2,3] unchanged
      reshape x=t0 params={shape=[W=2 C=3]} |}]

let%expect_test
    "_weight_norm lowers to one node; group_norm takes a [N, C, L] input" =
  let prog nodes extra out =
    program ~x_sizes:[ 2; 3 ] ~params:[ "g" ]
      ~extra_tensor_values:([ ("g", meta [ 3 ]); ("y", meta out) ] @ extra)
      ~nodes
      ~graph_outputs:[ as_tensor "y" ]
      ()
  in
  dump "_weight_norm dim 1"
    (prog
       [
         node "_weight_norm.default"
           [ tin "v" "x"; tin "g" "g"; int_in "dim" 1 ]
           "y";
       ]
       [] [ 2; 3 ]);
  let gn =
    program ~x_sizes:[ 1; 4; 6 ] ~params:[ "w"; "b" ]
      ~extra_tensor_values:
        [ ("w", meta [ 4 ]); ("b", meta [ 4 ]); ("y", meta [ 1; 4; 6 ]) ]
      ~nodes:
        [
          node "group_norm.default"
            [
              tin "input" "x";
              int_in "num_groups" 2;
              tin "weight" "w";
              tin "bias" "b";
              float_in "eps" 1e-5;
            ]
            "y";
        ]
      ~graph_outputs:[ as_tensor "y" ]
      ()
  in
  dump "group_norm on [1, 4, 6]" gn;
  [%expect
    {|
    _weight_norm dim 1
      weight_norm v=t0 g=t1 params={axis=C}
    group_norm on [1, 4, 6]
      permute x=t0 perm=[W<-C, C<-W]
      group_norm
        x=t3 <-n0
        weight=t1
        bias=t2
        params={batch=H; channel=C; groups=2; eps=1e-05}
      permute x=t4 <-n1 perm=[W<-C, C<-W] |}]

(* esm2's vocabulary: [erf] (its exact GELU written out), [masked_fill] of a
   binary32 tensor, [type_as] an int32 tensor, and an int64 mask times a
   binary32 tensor, which ATen promotes by casting the mask. *)
let%expect_test "erf, masked_fill, type_as int32, and an int64 x binary32 mul" =
  let prog ?(y = meta [ 2; 3 ]) nodes =
    program ~x_sizes:[ 2; 3 ] ~params:[ "m"; "a"; "i" ]
      ~extra_tensor_values:
        [
          ("m", meta ~dtype:12 [ 2; 3 ]);
          ("a", meta ~dtype:5 [ 2; 3 ]);
          ("i", meta ~dtype:4 [ 2; 3 ]);
          ("y", y);
        ]
      ~nodes
      ~graph_outputs:[ as_tensor "y" ]
      ()
  in
  dump "erf" (prog [ node "erf.default" [ tin "self" "x" ] "y" ]);
  dump "masked_fill.Scalar"
    (prog
       [
         node "masked_fill.Scalar"
           [ tin "self" "x"; tin "mask" "m"; float_in "value" 0. ]
           "y";
       ]);
  dump "masked_fill.Scalar of an int64 tensor is refused"
    (prog
       ~y:(meta ~dtype:5 [ 2; 3 ])
       [
         node "masked_fill.Scalar"
           [ tin "self" "a"; tin "mask" "m"; float_in "value" 0. ]
           "y";
       ]);
  dump "type_as an int32 tensor"
    (prog
       ~y:(meta ~dtype:4 [ 2; 3 ])
       [ node "type_as.default" [ tin "self" "a"; tin "other" "i" ] "y" ]);
  dump "mul binary32 by int64"
    (prog [ node "mul.Tensor" [ tin "self" "x"; tin "other" "a" ] "y" ]);
  dump "mul int64 by binary32"
    (prog [ node "mul.Tensor" [ tin "self" "a"; tin "other" "x" ] "y" ]);
  [%expect
    {|
    erf
      erf x=t0
    masked_fill.Scalar
      bitwise_not x=t1
      where_scalar_other condition=t4 <-n0 x=t0 scalar=0
    masked_fill.Scalar of an int64 tensor is refused
      malformed PT2 graph: torch.ops.aten.masked_fill.Scalar: dtype is not supported
    type_as an int32 tensor
      to_copy x=t2 target=int
    mul binary32 by int64
      to_copy x=t2 target=float
      mul a=t0 b=t4 <-n0
    mul int64 by binary32
      to_copy x=t2 target=float
      mul a=t4 <-n0 b=t0 |}]

(* The values: [masked_fill] writes [value] where the mask holds and keeps
   [self] elsewhere; the promoted [mul] multiplies by the int64 values. *)
let%expect_test "masked_fill and the promoted mul, evaluated" =
  let prog target inputs =
    program ~x_sizes:[ 2; 3 ] ~params:[ "m"; "a" ]
      ~extra_tensor_values:
        [
          ("m", meta ~dtype:12 [ 2; 3 ]);
          ("a", meta ~dtype:5 [ 2; 3 ]);
          ("y", meta [ 2; 3 ]);
        ]
      ~nodes:[ node target inputs "y" ]
      ~graph_outputs:[ as_tensor "y" ]
      ()
  in
  let eval label json =
    match lower json with
    | Error e ->
        Format.printf "%s: %a@." label Native_interp.pp_error (Err.Error.kind e)
    | Ok l -> (
        let g = l.Pt2_native_graph.graph in
        let value id =
          let sg = Tensor_id.Map.find id g.Graph_ir.Graph.tensors in
          let shape = sg.Tensor_sig.shape in
          let flat c = (Vec6.offset shape c :> int) in
          match sg.Tensor_sig.fmt with
          | Payload.Fmt Payload.Bool ->
              Tensor.materialize_bool shape (fun c -> flat c mod 2 = 0)
          | Payload.Fmt Payload.I64 ->
              Tensor.materialize_i64 shape (fun c -> Int64.of_int (flat c - 2))
          | _ -> Tensor.materialize shape (fun c -> float_of_int (flat c + 1))
        in
        let bound =
          List.map (fun id -> (id, value id)) g.Graph_ir.Graph.inputs
        in
        let inputs, constants =
          List.partition
            (fun (id, _) -> Graph_ir.input_kind g id = Graph_common.Input.Input)
            bound
        in
        match Eval_direct.run g ~inputs ~constants with
        | Error e ->
            Format.printf "%s: %a@." label Eval_direct.pp_error
              (Err.Error.kind e)
        | Ok env ->
            let y = Tensor_id.Map.find (List.hd g.Graph_ir.Graph.outputs) env in
            let out = ref [] in
            Vec6.iter (Tensor.packed_shape y) (fun c ->
                out := Tensor.read y c :: !out);
            Format.printf "%s: %a@." label
              Fmt.(list ~sep:sp float)
              (List.rev !out))
  in
  (* x = 1 .. 6, m true at even offsets, a = -2 .. 3 *)
  eval "masked_fill(x, m, -7)"
    (prog "masked_fill.Scalar"
       [ tin "self" "x"; tin "mask" "m"; float_in "value" (-7.) ]);
  eval "x * a" (prog "mul.Tensor" [ tin "self" "x"; tin "other" "a" ]);
  eval "a * x" (prog "mul.Tensor" [ tin "self" "a"; tin "other" "x" ]);
  [%expect
    {|
    masked_fill(x, m, -7): -7 2 -7 4 -7
    6
    x * a: -2 -2 0 4 10
    18
    a * x: -2 -2 0 4 10
    18 |}]
