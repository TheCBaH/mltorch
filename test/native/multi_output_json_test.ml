(* Multi-output and variadic ops through the codec: Unbind,
   Split_with_sizes, Group_norm (with and without affine), and Concat. Split
   from graph_json_test.ml. *)

open Graph_json_fixtures

let%expect_test "graph with Unbind op: encode → decode → pretty-print" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"unbind_c" ~outputs:(fun ids -> ids)
          @@
          let* x = input ~shape:(s 1 1 1 2 4 3) ~name:"x" () in
          unbind ~name:"slice" { Split.Unbind.axis = Axis.C } x)
    in
    let* json = encode_graph g in
    decode_graph json
  in
  Format.printf "%a@." (pp_result pp_decoded) result;
  [%expect
    {|
    decoded:
    graph
    inputs: [t0 f32 [H=2 W=4 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [W=2 C=4], t2 f32 [W=2 C=4], t3 f32 [W=2 C=4]] =
        unbind x=t0 params={axis=C}
    outputs:
      [t1 f32 [W=2 C=4] <-n0, t2 f32 [W=2 C=4] <-n0, t3 f32 [W=2 C=4] <-n0] |}]

(* Two DIFFERENT window sizes, so a codec that dropped or reordered a [sizes]
   entry is visible in the shapes as well as the params -- and, like
   [Unbind]'s test above, every output edge (not just the first) has to
   survive. *)
let%expect_test "graph with Split_with_sizes op: encode → decode → pretty-print"
    =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"split_with_sizes" ~outputs:(fun ids -> ids)
          @@
          let* x = input ~shape:(s 1 1 1 2 5 3) ~name:"x" () in
          split_with_sizes ~name:"pieces"
            { Split.Split_with_sizes.axis = Axis.W; sizes = [ 2; 3 ] }
            x)
    in
    let* json = encode_graph g in
    decode_graph json
  in
  Format.printf "%a@." (pp_result pp_decoded) result;
  [%expect
    {|
    decoded:
    graph
    inputs: [t0 f32 [H=2 W=5 C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [H=2 W=2 C=3], t2 f32 [H=2 W=3 C=3]] =
        split_with_sizes x=t0 params={axis=W sizes=[2, 3]}
    outputs: [t1 f32 [H=2 W=2 C=3] <-n0, t2 f32 [H=2 W=3 C=3] <-n0] |}]

(* [groups] round-trips as a plain int inside [params], not [Op_config.Pos.t]'s
   own codec, and the weight/bias PAIR states are exercised (both present,
   both absent) since both options are independently reachable through
   [Graph_ir]. *)
let%expect_test "graph with Group_norm op: encode → decode → pretty-print" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"group_norm" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s 1 1 1 2 2 4) ~name:"x" () in
          let* w = constant ~shape:(s1c 4) ~name:"w" () in
          let* b = constant ~shape:(s1c 4) ~name:"b" () in
          group_norm ~name:"out"
            {
              Norm.GroupNorm.channel = Axis.C;
              groups = Op_config.Pos.of_int 2;
              eps = 1e-5;
            }
            ~x ~weight:w ~bias:b ())
    in
    let* json = encode_graph g in
    decode_graph json
  in
  Format.printf "%a@." (pp_result pp_decoded) result;
  [%expect
    {|
    decoded:
    graph
    inputs:
      [t0 f32 [H=2 W=2 C=4] ->[n0], t1 f32 [C=4] ->[n0] constant,
       t2 f32 [C=4] ->[n0] constant]
    nodes:
      n0: [t3 f32 [H=2 W=2 C=4]] =
        group_norm x=t0 weight=t1 bias=t2 params={channel=C; groups=2; eps=1e-05}
    outputs: [t3 f32 [H=2 W=2 C=4] <-n0] |}]

(* No weight, no bias: the option-absent codec path, not exercised by the
   fixture above. *)
let%expect_test
    "graph with Group_norm op, no affine: encode → decode → pretty-print" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"group_norm_bare" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s 1 1 1 2 2 4) ~name:"x" () in
          group_norm ~name:"out"
            {
              Norm.GroupNorm.channel = Axis.C;
              groups = Op_config.Pos.of_int 2;
              eps = 1e-5;
            }
            ~x ())
    in
    let* json = encode_graph g in
    decode_graph json
  in
  Format.printf "%a@." (pp_result pp_decoded) result;
  [%expect
    {|
    decoded:
    graph
    inputs: [t0 f32 [H=2 W=2 C=4] ->[n0]]
    nodes:
      n0: [t1 f32 [H=2 W=2 C=4]] =
        group_norm
          x=t0
          weight=none
          bias=none
          params={channel=C; groups=2; eps=1e-05}
    outputs: [t1 f32 [H=2 W=2 C=4] <-n0] |}]

(* The [xs] field is the first variadic-OPERAND list codec any op has -- every
   earlier op's [Tensor_ref.jsont] usage is for a single-arity field or (in
   [Unbind]'s case) a single-input, multi-OUTPUT op, never a JSON array of
   operand refs. *)
let%expect_test "graph with Concat op: encode → decode → pretty-print" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"cat" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 2) ~name:"a" () in
          let* b = input ~shape:(s1c 3) ~name:"b" () in
          let* c = input ~shape:(s1c 1) ~name:"c" () in
          concat ~name:"out" { Concat.Concat.axis = Axis.C } [ a; b; c ])
    in
    let* json = encode_graph g in
    decode_graph json
  in
  Format.printf "%a@." (pp_result pp_decoded) result;
  [%expect
    {|
    decoded:
    graph
    inputs: [t0 f32 [C=2] ->[n0], t1 f32 [C=3] ->[n0], t2 f32 [C=1] ->[n0]]
    nodes:
      n0: [t3 f32 [C=6]] = concat xs=[t0, t1, t2] params={axis=C}
    outputs: [t3 f32 [C=6] <-n0] |}]
