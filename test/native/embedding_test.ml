(* `embedding.default` at the [Graph_ir]/[Eval_direct] level and through the
   Symbolic -> Kernel route.

   The table is [V=5, D=3] with element (v, d) = 10*v + d, so every value names
   the row and column it came from. Direct evaluation is strict: an index
   outside [0, V) -- negative included -- is an error row. The symbolic route
   shares [index.Tensor]'s gather, which also takes [-V, -1] by wrapping; the
   last test states that difference rather than hiding it. *)

open Graph_ir
open Graph_direct_fixtures

let vocab = 5
let dim = 3
let weight_shape = s 1 1 1 1 vocab dim

let weight =
  Tensor.materialize weight_shape (fun c ->
      float_of_int
        ((10 * Dim.to_int (Vec6.get c Axis.W)) + Dim.to_int (Vec6.get c Axis.C)))

let build ?(padding_idx = -1) ?(indices_fmt = Payload.(Fmt I64)) ~indices_shape
    ~indices_rank () =
  Graph_builder.(
    build ~name:"embedding" ~outputs:(fun r -> [ r ])
    @@
    let* weight = input ~shape:weight_shape ~name:"weight" () in
    let* indices =
      constant ~shape:indices_shape ~fmt:indices_fmt ~name:"indices" ()
    in
    embedding ~name:"out"
      {
        Embedding.Embedding.indices_rank = Rank.of_int indices_rank;
        padding_idx = Aten_int.Index.of_int padding_idx;
      }
      ~weight ~indices)

let ids (g : graph) =
  match g.Graph.inputs with [ w; i ] -> (w, i) | _ -> assert false

let run_direct g indices =
  let open Err.Syntax in
  let w_id, i_id = ids g in
  let* env =
    lift_eval
      (Eval_direct.run g
         ~inputs:[ (w_id, weight) ]
         ~constants:[ (i_id, indices) ])
  in
  tensor_of_name g env "out"

let i64_tensor shape values =
  Tensor.materialize_i64 shape (fun c ->
      let flat =
        List.fold_left
          (fun acc a ->
            (acc * (Vec6.get shape a :> int)) + Dim.to_int (Vec6.get c a))
          0 Axis.all
      in
      List.nth values flat)

(* [Tensor.pp] stops after eight elements; a lookup is judged by every row. *)
let pp_all ppf tensor =
  let (Tensor.Tensor t) = tensor in
  Format.fprintf ppf "%a {" Vec6.pp_shape t.Tensor.shape;
  let first = ref true in
  Vec6.iter t.Tensor.shape (fun c ->
      if not !first then Format.fprintf ppf ", ";
      first := false;
      Format.fprintf ppf "%g" (Tensor.read tensor c));
  Format.fprintf ppf "}"

let show label result = Format.printf "%s: %a@." label (pp_result pp_all) result

let rank1 values =
  let n = List.length values in
  let shape = s 1 1 1 1 1 n in
  let result =
    let open Err.Syntax in
    let* g = lift_build (build ~indices_shape:shape ~indices_rank:1 ()) in
    run_direct g (i64_tensor shape values)
  in
  result

let%expect_test "Direct graph: rows are looked up, repeated and at both ends" =
  show "rank-1 [3;0;3;4]" (rank1 [ 3L; 0L; 3L; 4L ]);
  [%expect
    {| rank-1 [3;0;3;4]: [W=4 C=3] {30, 31, 32, 0, 1, 2, 30, 31, 32, 40, 41, 42} |}]

let%expect_test "Direct graph: a rank-2 index grows the output by the dimension"
    =
  let shape = s 1 1 1 1 2 3 in
  let result =
    let open Err.Syntax in
    let* g = lift_build (build ~indices_shape:shape ~indices_rank:2 ()) in
    Format.printf "%a@." Graph_ir.pp g;
    run_direct g (i64_tensor shape [ 0L; 1L; 2L; 4L; 4L; 3L ])
  in
  show "rank-2 [[0;1;2];[4;4;3]]" result;
  [%expect
    {|
    graph
    inputs: [t0 f32 [W=5 C=3] ->[n0], t1 i64 [W=2 C=3] ->[n0] constant]
    nodes:
      n0: [t2 f32 [H=2 W=3 C=3]] =
        embedding weight=t0 indices=t1 params={indices_rank=2 padding_idx=-1}
    outputs: [t2 f32 [H=2 W=3 C=3] <-n0]
    rank-2 [[0;1;2];[4;4;3]]: [H=2 W=3 C=3] {0, 1, 2, 10, 11, 12, 20, 21, 22, 40, 41, 42, 40, 41, 42, 30, 31, 32} |}]

(* The padding row is returned as stored, never zeroed: [padding_idx] steers
   only ATen's backward pass. *)
let%expect_test "Direct graph: padding_idx does not touch the forward lookup" =
  let shape = s 1 1 1 1 1 3 in
  let with_padding p =
    let open Err.Syntax in
    let* g =
      lift_build (build ~padding_idx:p ~indices_shape:shape ~indices_rank:1 ())
    in
    run_direct g (i64_tensor shape [ 2L; 1L; 2L ])
  in
  show "padding_idx = 2" (with_padding 2);
  show "padding_idx = -1" (with_padding (-1));
  [%expect
    {|
    padding_idx = 2: [W=3 C=3] {20, 21, 22, 10, 11, 12, 20, 21, 22}
    padding_idx = -1: [W=3 C=3] {20, 21, 22, 10, 11, 12, 20, 21, 22} |}]

let%expect_test
    "Direct graph: an index outside [0, V) is an error, never a wrap or a clamp"
    =
  let try_index v =
    let shape = s 1 1 1 1 1 2 in
    let result =
      let open Err.Syntax in
      let* g = lift_build (build ~indices_shape:shape ~indices_rank:1 ()) in
      run_direct g (i64_tensor shape [ 1L; v ])
    in
    show (Int64.to_string v) result
  in
  try_index 4L;
  try_index 5L;
  try_index (-1L);
  try_index (-5L);
  try_index Int64.max_int;
  try_index Int64.min_int;
  [%expect
    {|
    4: [W=2 C=3] {10, 11, 12, 40, 41, 42}
    5: embedding index 5 out of range [0, 5)
    -1: embedding index -1 out of range [0, 5)
    -5: embedding index -5 out of range [0, 5)
    9223372036854775807: embedding index 9223372036854775807 out of range [0, 5)
    -9223372036854775808: embedding index -9223372036854775808 out of range [0, 5) |}]

let%expect_test "Direct graph: float indices or a non-f32 table are refused" =
  let shape = s 1 1 1 1 1 2 in
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        (build
           ~indices_fmt:Payload.(Fmt F32)
           ~indices_shape:shape ~indices_rank:1 ())
    in
    let w_id, i_id = ids g in
    let indices = Tensor.materialize shape (fun _ -> 1.) in
    let* env =
      lift_eval
        (Eval_direct.run g
           ~inputs:[ (w_id, weight) ]
           ~constants:[ (i_id, indices) ])
    in
    tensor_of_name g env "out"
  in
  show "f32 indices" result;
  [%expect
    {| f32 indices: embedding: weight must be f32 and indices i64, got weight=f32 indices=f32 |}]

let%expect_test "Graph: shape rules are checked when the node is built" =
  let build_with ~weight_shape ~indices_shape ~indices_rank =
    Graph_builder.(
      build ~name:"bad" ~outputs:(fun r -> [ r ])
      @@
      let* weight = input ~shape:weight_shape ~name:"weight" () in
      let* indices =
        constant ~shape:indices_shape ~fmt:Payload.(Fmt I64) ~name:"indices" ()
      in
      embedding
        {
          Embedding.Embedding.indices_rank = Rank.of_int indices_rank;
          padding_idx = Aten_int.Index.of_int (-1);
        }
        ~weight ~indices)
  in
  let show_build label r =
    Format.printf "%s: %s@." label
      (match Err.payload r with
      | Ok _ -> "built"
      | Error e -> Format.asprintf "%a" Graph_builder.pp_error e)
  in
  show_build "table with a batch axis"
    (build_with ~weight_shape:(s 1 1 1 2 5 3) ~indices_shape:(s 1 1 1 1 1 4)
       ~indices_rank:1);
  (* The six-axis frame erases a leading extent-1 axis: [4] and [1, 4] are the
     same frame, so a rank claim of 2 over it is a [1, 4] index, not a mistake. *)
  show_build "a [1, 4] index (claimed rank 2)"
    (build_with ~weight_shape ~indices_shape:(s 1 1 1 1 1 4) ~indices_rank:2);
  show_build "indices claimed rank 1 but rank 2"
    (build_with ~weight_shape ~indices_shape:(s 1 1 1 1 2 3) ~indices_rank:1);
  [%expect
    {|
    table with a batch axis: embedding weight must be a [V, D] matrix, got [H=2 W=5 C=3]
    a [1, 4] index (claimed rank 2): built
    indices claimed rank 1 but rank 2: index.Tensor: index declared rank 1, but its own axis W has extent 2 (must be 1, outside a rank-1 tensor's own real axes) |}]

let%expect_test "Graph: Embedding survives a JSON round trip" =
  let g =
    match
      Err.payload
        (build ~padding_idx:7 ~indices_shape:(s 1 1 1 1 1 4) ~indices_rank:1 ())
    with
    | Ok g -> g
    | Error _ -> assert false
  in
  let json =
    match Graph_json.encode_graph ~format:Jsont.Indent g with
    | Ok j -> j
    | Error _ -> assert false
  in
  let g2 =
    match Graph_json.decode_graph json with
    | Ok g2 -> g2
    | Error e -> Err.or_raise ~pp_error:Graph_json.pp_error (Error e)
  in
  let printed g = Format.asprintf "%a" Graph_ir.pp g in
  Format.printf "round-trips identically: %b@."
    (String.equal (printed g) (printed g2));
  Format.printf "%s@." (printed g2);
  [%expect
    {|
    round-trips identically: true
    graph
    inputs: [t0 f32 [W=5 C=3] ->[n0], t1 i64 [C=4] ->[n0] constant]
    nodes:
      n0: [t2 f32 [W=4 C=3]] =
        embedding weight=t0 indices=t1 params={indices_rank=1 padding_idx=7}
    outputs: [t2 f32 [W=4 C=3] <-n0] |}]

(* Symbolic -> Kernel on the same graphs. For every valid index it agrees with
   Direct; the one difference is the negative index, which the shared gather
   wraps and Direct rejects. *)
let kernel_result g indices =
  let w_id, i_id = ids g in
  let kernel =
    Kernel_adapt.of_stage_program (Eval_symbolic.run g)
    |> Err.or_raise ~pp_error:Kernel_adapt.pp_error
  in
  let result =
    Kernel_eval.run kernel ~bind:(fun id ->
        if Tensor_id.equal id w_id then Some weight
        else if Tensor_id.equal id i_id then Some indices
        else None)
  in
  match Err.payload result with
  | Ok map ->
      Format.asprintf "%a" pp_all
        (Tensor_id.Map.find (List.hd g.Graph.outputs) map)
  | Error e -> Format.asprintf "error: %a" Kernel_eval.pp_error e

let%expect_test "Symbolic -> Kernel agrees with Direct on valid indices" =
  let shape = s 1 1 1 1 2 3 in
  let g =
    match Err.payload (build ~indices_shape:shape ~indices_rank:2 ()) with
    | Ok g -> g
    | Error _ -> assert false
  in
  let indices = i64_tensor shape [ 0L; 1L; 2L; 4L; 4L; 3L ] in
  Format.printf "kernel: %s@." (kernel_result g indices);
  (match Err.payload (run_direct g indices) with
  | Ok t -> Format.printf "direct: %a@." pp_all t
  | Error _ -> print_endline "direct: error");
  [%expect
    {|
    kernel: [H=2 W=3 C=3] {0, 1, 2, 10, 11, 12, 20, 21, 22, 40, 41, 42, 40, 41, 42, 30, 31, 32}
    direct: [H=2 W=3 C=3] {0, 1, 2, 10, 11, 12, 20, 21, 22, 40, 41, 42, 40, 41, 42, 30, 31, 32} |}]

let%expect_test
    "known difference: the symbolic gather wraps -1, Direct rejects it" =
  let shape = s 1 1 1 1 1 2 in
  let g =
    match Err.payload (build ~indices_shape:shape ~indices_rank:1 ()) with
    | Ok g -> g
    | Error _ -> assert false
  in
  let indices = i64_tensor shape [ 1L; -1L ] in
  Format.printf "kernel, index -1: %s@." (kernel_result g indices);
  show "direct, index -1" (run_direct g indices);
  Format.printf "kernel, index 5: %s@."
    (kernel_result g (i64_tensor shape [ 1L; 5L ]));
  [%expect
    {|
    kernel, index -1: [W=2 C=3] {10, 11, 12, 40, 41, 42}
    direct, index -1: embedding index -1 out of range [0, 5)
    kernel, index 5: error: gather index 5 out of range [-5, 4] |}]
