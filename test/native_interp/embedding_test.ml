(* torch.ops.aten.embedding.default through the serialized path, the way the
   Transformers fixtures write it: a [V, D] float32 table (a captured weight),
   an int64 index tensor of rank 1 or 2, and an optional padding_idx. One
   [Embedding] node; the indices' ATen rank is read here and carried because
   the six-axis frame erases it. Numeric agreement with ATen is checked in
   test/native_bridge/embedding_test.ml. *)

open Programs

let embedding_node ?padding_idx () =
  let padding =
    match padding_idx with
    | None -> ""
    | Some p -> jstr {|,{"name":"padding_idx","arg":{"as_int":%d},"kind":1}|} p
  in
  jstr
    {|{"target":"torch.ops.aten.embedding.default","inputs":[{"name":"weight","arg":%s,"kind":1},{"name":"indices","arg":%s,"kind":1}%s],"outputs":[%s],"metadata":{}}|}
    (as_tensor "w") (as_tensor "x") padding (as_tensor "y")

(* [x] is the int64 index input; [w] the table, a captured parameter. *)
let with_long_x json =
  let needle = {|"x":{"dtype":7|} in
  let i =
    let rec go i =
      if String.sub json i (String.length needle) = needle then i else go (i + 1)
    in
    go 0
  in
  String.sub json 0 i ^ {|"x":{"dtype":5|}
  ^ String.sub json
      (i + String.length needle)
      (String.length json - i - String.length needle)

let prog ?padding_idx ~x_sizes ~vocab ~dim () =
  with_long_x
    (program ~x_sizes
       ~extra_tensor_values:
         [
           ("w", tensor_meta [ vocab; dim ]);
           ("y", tensor_meta (x_sizes @ [ dim ]));
         ]
       ~params:[ "w" ]
       ~nodes:[ embedding_node ?padding_idx () ]
       ~graph_outputs:[ as_tensor "y" ]
       ())

let dump label json =
  Format.printf "%s@." label;
  match lower json with
  | Error e -> Format.printf "  %a@." Native_interp.pp_error (Err.Error.kind e)
  | Ok l -> Format.printf "%a@." Graph_ir.pp l.Pt2_native_graph.graph

let%expect_test "embedding.default lowers to one Embedding node" =
  dump "rank-1 indices, default padding_idx:"
    (prog ~x_sizes:[ 16 ] ~vocab:49152 ~dim:576 ());
  dump "rank-2 indices (1, 16), padding_idx 2:"
    (prog ~padding_idx:2 ~x_sizes:[ 1; 16 ] ~vocab:49152 ~dim:576 ());
  dump "rank-2 indices (16, 16), the T5 bucket table:"
    (prog ~x_sizes:[ 16; 16 ] ~vocab:32 ~dim:8 ());
  [%expect
    {|
    rank-1 indices, default padding_idx:
    graph
    inputs: [t0 i64 [C=16] ->[n0], t1 f32 [W=49152 C=576] ->[n0] constant]
    nodes:
      n0: [t2 f32 [W=16 C=576]] =
        embedding weight=t1 indices=t0 params={indices_rank=1 padding_idx=-1}
    outputs: [t2 f32 [W=16 C=576] <-n0]
    rank-2 indices (1, 16), padding_idx 2:
    graph
    inputs: [t0 i64 [C=16] ->[n0], t1 f32 [W=49152 C=576] ->[n0] constant]
    nodes:
      n0: [t2 f32 [W=16 C=576]] =
        embedding weight=t1 indices=t0 params={indices_rank=2 padding_idx=2}
    outputs: [t2 f32 [W=16 C=576] <-n0]
    rank-2 indices (16, 16), the T5 bucket table:
    graph
    inputs: [t0 i64 [W=16 C=16] ->[n0], t1 f32 [W=32 C=8] ->[n0] constant]
    nodes:
      n0: [t2 f32 [H=16 W=16 C=8]] =
        embedding weight=t1 indices=t0 params={indices_rank=2 padding_idx=-1}
    outputs: [t2 f32 [H=16 W=16 C=8] <-n0] |}]
