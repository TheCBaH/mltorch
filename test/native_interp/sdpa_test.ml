(* torch.ops.aten.scaled_dot_product_attention.default through the serialized
   path (op8-impl.md commit 3).

   No model in this repo's zoo serializes this target (F1) -- these fixtures
   ARE the coverage, not a supplement to it. Numeric agreement is checked in
   test/native/compute_test.ml (hand-computed) and test/native_bridge_test.ml
   (the ATen-live-tensor path); this file only asserts what
   [Native_interp.lower] builds and rejects on the SERIALIZED metadata path,
   which decodes rank from the declared [sizes] list rather than from a live
   ATen tensor -- different code from [Op_bridge]'s, and the only place
   op8-impl.md F13's rank-erasure trap (a rank-3 [1,Wq,Wk] mask normalizing to
   the same Native shape as an admissible rank-4 one) can be proven on this
   importer specifically. *)

open Programs

let sdpa_node ?(mask = `Absent) ?dropout_p ?is_causal ?scale ?enable_gqa () =
  let mask_arg =
    match mask with
    | `Absent -> ""
    | `None -> {|,{"name":"attn_mask","arg":{"as_none":true},"kind":1}|}
    | `Tensor ->
        jstr {|,{"name":"attn_mask","arg":%s,"kind":1}|} (as_tensor "m")
  in
  let float_arg name = function
    | None -> ""
    | Some f -> jstr {|,{"name":"%s","arg":{"as_float":%.12g},"kind":1}|} name f
  in
  let bool_arg name = function
    | None -> ""
    | Some b -> jstr {|,{"name":"%s","arg":{"as_bool":%b},"kind":1}|} name b
  in
  jstr
    {|{"target":"torch.ops.aten.scaled_dot_product_attention.default","inputs":[{"name":"query","arg":%s,"kind":1},{"name":"key","arg":%s,"kind":1},{"name":"value","arg":%s,"kind":1}%s%s%s%s%s],"outputs":[%s],"metadata":{}}|}
    (as_tensor "x") (as_tensor "k") (as_tensor "v") mask_arg
    (float_arg "dropout_p" dropout_p)
    (bool_arg "is_causal" is_causal)
    (float_arg "scale" scale)
    (bool_arg "enable_gqa" enable_gqa)
    (as_tensor "y")

(* [x] is query (via [program]'s primary input); key/value/mask are captured
   [params], following [sub.Tensor]'s "other" operand precedent
   (binary_test.ml) -- the only way this helper reaches a second real tensor
   operand. *)
let prog ?(q_sizes = [ 1; 1; 1; 2 ]) ?(k_sizes = [ 1; 1; 2; 2 ])
    ?(v_sizes = [ 1; 1; 2; 2 ]) ?mask_sizes node =
  let mask_extra =
    match mask_sizes with None -> [] | Some s -> [ ("m", tensor_meta s) ]
  in
  program ~x_sizes:q_sizes
    ~extra_tensor_values:
      (("k", tensor_meta k_sizes) :: ("v", tensor_meta v_sizes) :: mask_extra)
    ~params:
      ([ "k"; "v" ] @ match mask_sizes with None -> [] | Some _ -> [ "m" ])
    ~nodes:[ node ]
    ~graph_outputs:[ as_tensor "y" ]
    ()

let dump label json =
  Format.printf "%s@." label;
  match lower json with
  | Error e -> Format.printf "  %a@." Native_interp.pp_error (Err.Error.kind e)
  | Ok l -> Format.printf "%a@." Graph_ir.pp l.Pt2_native_graph.graph

let%expect_test "sdpa: mask absent vs present" =
  dump "mask absent:" (prog (sdpa_node ()));
  dump "mask present:" (prog ~mask_sizes:[ 1; 2 ] (sdpa_node ~mask:`Tensor ()));
  [%expect
    {|
    mask absent:
    graph
    inputs:
      [t0 f32 [C=2] ->[n0], t1 f32 [W=2 C=2] ->[n0] constant,
       t2 f32 [W=2 C=2] ->[n0] constant]
    nodes:
      group g1 torch.ops.aten.scaled_dot_product_attention.default:
        n0: [t3 f32 [C=2]] =
          sdpa query=t0 key=t1 value=t2 mask=none params={scale=default}
    outputs: [t3 f32 [C=2] <-n0]
    mask present:
    graph
    inputs:
      [t0 f32 [C=2] ->[n0], t1 f32 [W=2 C=2] ->[n0] constant,
       t2 f32 [W=2 C=2] ->[n0] constant, t3 f32 [C=2] ->[n0] constant]
    nodes:
      group g1 torch.ops.aten.scaled_dot_product_attention.default:
        n0: [t4 f32 [C=2]] =
          sdpa query=t0 key=t1 value=t2 mask=t3 params={scale=default}
    outputs: [t4 f32 [C=2] <-n0]
    |}]

let%expect_test "sdpa: explicit scale survives the serialized path" =
  dump "scale=0.1:" (prog (sdpa_node ~scale:0.1 ()));
  [%expect
    {|
    scale=0.1:
    graph
    inputs:
      [t0 f32 [C=2] ->[n0], t1 f32 [W=2 C=2] ->[n0] constant,
       t2 f32 [W=2 C=2] ->[n0] constant]
    nodes:
      group g1 torch.ops.aten.scaled_dot_product_attention.default:
        n0: [t3 f32 [C=2]] =
          sdpa query=t0 key=t1 value=t2 mask=none params={scale=explicit(0.1)}
    outputs: [t3 f32 [C=2] <-n0]
    |}]

let%expect_test "sdpa: dropout_p, is_causal, enable_gqa are typed rejections" =
  dump "dropout_p=0.5:" (prog (sdpa_node ~dropout_p:0.5 ()));
  dump "is_causal=true:" (prog (sdpa_node ~is_causal:true ()));
  dump "enable_gqa=true:" (prog (sdpa_node ~enable_gqa:true ()));
  [%expect
    {|
    dropout_p=0.5:
      malformed PT2 graph: sdpa: dropout_p=0.5 is not supported (only 0)
    is_causal=true:
      malformed PT2 graph: sdpa: is_causal=true is not supported
    enable_gqa=true:
      malformed PT2 graph: sdpa: enable_gqa=true is not supported
    |}]

let%expect_test "sdpa: negative or non-finite explicit scale is rejected" =
  dump "scale=-1:" (prog (sdpa_node ~scale:(-1.0) ()));
  [%expect
    {|
    scale=-1:
      malformed PT2 graph: sdpa: explicit scale=-1 is negative; ATen handles this specially and this row does not implement it
    |}]

(* F13, proven on THIS importer: rank is read from the DECLARED [sizes] list
   length, before any right-alignment into the six-axis frame -- a rank-3
   [1,Wq,Wk] mask and an admissible rank-4 [1,1,Wq,Wk] one normalize to the
   SAME Native shape, so only a check on the raw declared rank can tell them
   apart. *)
let%expect_test
    "sdpa: rank is checked on the declared metadata, not the normalized frame" =
  dump "query rank 3:" (prog ~q_sizes:[ 1; 1; 2 ] (sdpa_node ()));
  dump "mask rank 3 [1,Wq,Wk]:"
    (prog ~mask_sizes:[ 1; 1; 2 ] (sdpa_node ~mask:`Tensor ()));
  [%expect
    {|
    query rank 3:
      malformed PT2 graph: x is rank 3, expected 4
    mask rank 3 [1,Wq,Wk]:
      malformed PT2 graph: sdpa: sdpa attn_mask has rank 3, expected 2 or 4
    |}]
