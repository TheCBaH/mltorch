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
