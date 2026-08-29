(* Per-op encode -> decode round trips: Add, Conv2d (with and without bias),
   Permute, the *_scalar/Clamp cluster, Hardtanh/Clone, and the Group 5
   activations. Split from graph_json_test.ml. *)

open Graph_ir
open Graph_json_fixtures

let%expect_test "op Add: encode → JSON" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"add" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 3) ~name:"a" () in
          let* b = input ~shape:(s1c 3) ~name:"b" () in
          add ~name:"out" a b)
    in
    let* node = first_node g in
    let* graph_json = encode_graph g in
    let* op_json = encode_op node.Node.op in
    Err.return (graph_json, op_json)
  in
  Format.printf "%a@." (pp_result pp_graph_json_and_op) result;
  [%expect
    {|
    graph JSON:
    {
      "inputs": [
        0,
        1
      ],
      "nodes": [
        {
          "id": 0,
          "op": {
            "Add": {
              "a": 0,
              "b": 1
            }
          },
          "outputs": [
            2
          ]
        }
      ],
      "root": {
        "id": 0,
        "items": [
          {
            "Node": 0
          }
        ]
      },
      "outputs": [
        2
      ],
      "tensors": [
        {
          "fmt": "f32",
          "id": 0,
          "shape": [
            1,
            1,
            1,
            1,
            1,
            3
          ]
        },
        {
          "fmt": "f32",
          "id": 1,
          "shape": [
            1,
            1,
            1,
            1,
            1,
            3
          ]
        },
        {
          "fmt": "f32",
          "id": 2,
          "shape": [
            1,
            1,
            1,
            1,
            1,
            3
          ]
        }
      ]
    }
    op JSON:
    {
      "Add": {
        "a": 0,
        "b": 1
      }
    } |}]

let%expect_test "op Conv2d: encode → decode → pretty-print graph" =
  let conv_params =
    Conv.Conv2d.
      {
        h = conv_axis ~kernel:3 ~stride:1 ~pad:1;
        w = conv_axis ~kernel:3 ~stride:1 ~pad:1;
        in_channels = Dim.extent 8;
        groups = Op_config.Pos.of_int 1;
      }
  in
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"conv" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s 1 1 1 4 4 8) ~name:"x" () in
          let* w = input ~shape:(s 16 1 1 3 3 8) ~name:"w" () in
          let* b = input ~shape:(s1c 16) ~name:"b" () in
          conv2d ~name:"y" conv_params ~x ~weight:w ~bias:b ())
    in
    let* json = encode_graph g in
    let* g2 = decode_graph json in
    Err.return (json, g2)
  in
  Format.printf "%a@." (pp_result pp_json_and_graph) result;
  [%expect
    {|
    JSON:
    {
      "inputs": [
        0,
        1,
        2
      ],
      "nodes": [
        {
          "id": 0,
          "op": {
            "Conv2d": {
              "bias": 2,
              "params": {
                "h": {
                  "kernel": 3,
                  "stride": 1,
                  "pad_before": 1,
                  "pad_after": 1,
                  "dilation": 1
                },
                "w": {
                  "kernel": 3,
                  "stride": 1,
                  "pad_before": 1,
                  "pad_after": 1,
                  "dilation": 1
                },
                "in_channels": 8,
                "groups": 1
              },
              "weight": 1,
              "x": 0
            }
          },
          "outputs": [
            3
          ]
        }
      ],
      "root": {
        "id": 0,
        "items": [
          {
            "Node": 0
          }
        ]
      },
      "outputs": [
        3
      ],
      "tensors": [
        {
          "fmt": "f32",
          "id": 0,
          "shape": [
            1,
            1,
            1,
            4,
            4,
            8
          ]
        },
        {
          "fmt": "f32",
          "id": 1,
          "shape": [
            16,
            1,
            1,
            3,
            3,
            8
          ]
        },
        {
          "fmt": "f32",
          "id": 2,
          "shape": [
            1,
            1,
            1,
            1,
            1,
            16
          ]
        },
        {
          "fmt": "f32",
          "id": 3,
          "shape": [
            1,
            1,
            1,
            4,
            4,
            16
          ]
        }
      ]
    }
    decoded graph:
    graph
    inputs:
      [t0 f32 [H=4 W=4 C=8] ->[n0], t1 f32 [N=16 T=1 D=1 H=3 W=3 C=8] ->[n0],
       t2 f32 [C=16] ->[n0]]
    nodes:
      n0: [t3 f32 [H=4 W=4 C=16]] =
        conv2d
          x=t0
          weight=t1
          bias=t2
          params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 in_channels=8;
                 groups=1}
    outputs: [t3 f32 [H=4 W=4 C=16] <-n0] |}]

let%expect_test "op Conv2d no bias: optional field absent in JSON" =
  let conv_params =
    Conv.Conv2d.
      {
        h = conv_axis ~kernel:1 ~stride:1 ~pad:0;
        w = conv_axis ~kernel:1 ~stride:1 ~pad:0;
        in_channels = Dim.extent 4;
        groups = Op_config.Pos.of_int 1;
      }
  in
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"conv_nb" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s1c 4) ~name:"x" () in
          let* w = input ~shape:(s 4 1 1 1 1 4) ~name:"w" () in
          conv2d ~name:"y" conv_params ~x ~weight:w ())
    in
    let* json = encode_graph g in
    let* g2 = decode_graph json in
    Err.return (g, g2)
  in
  Format.printf "%a@." (pp_result pp_original_and_graph) result;
  [%expect
    {|
    original:
    graph
    inputs: [t0 f32 [C=4] ->[n0], t1 f32 [N=4 T=1 D=1 H=1 W=1 C=4] ->[n0]]
    nodes:
      n0: [t2 f32 [C=4]] =
        conv2d
          x=t0
          weight=t1
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=4;
                 groups=1}
    outputs: [t2 f32 [C=4] <-n0]
    decoded:
    graph
    inputs: [t0 f32 [C=4] ->[n0], t1 f32 [N=4 T=1 D=1 H=1 W=1 C=4] ->[n0]]
    nodes:
      n0: [t2 f32 [C=4]] =
        conv2d
          x=t0
          weight=t1
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=4;
                 groups=1}
    outputs: [t2 f32 [C=4] <-n0] |}]

let%expect_test "op Permute: encode → decode" =
  let perm = Axis.[ (N, N); (T, T); (D, D); (H, W); (W, C); (C, H) ] in
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"perm" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s 1 1 1 2 3 4) ~name:"x" () in
          permute ~name:"y" perm x)
    in
    let* json = encode_graph g in
    decode_graph json
  in
  Format.printf "%a@." (pp_result pp_decoded_graph) result;
  [%expect
    {|
    decoded graph:
    graph
    inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
    nodes:
      n0: [t1 f32 [H=3 W=4 C=2]] = permute x=t0 perm=[H<-W, W<-C, C<-H]
    outputs: [t1 f32 [H=3 W=4 C=2] <-n0] |}]

(* The scalar-parameter and clamp ops carry JSON shapes the other ops do not: a
   bare [scalar] number, and two INDEPENDENTLY optional bounds. Round-tripping
   the whole hardsigmoid chain covers a present/absent bound in each position at
   once. The 0.1 scalar is not f32-exact, so this also pins that the builder's
   narrowing and [f32_jsont] agree — a double-precision 0.1 would come back as a
   different literal. *)
let%expect_test "ops Add_scalar/Clamp/Div_scalar/Mul_scalar: encode → decode" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"hardsigmoid" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s1c 3) ~name:"x" () in
          let* s = add_scalar 0.1 x in
          let* lo = clamp { Pointwise.Clamp.min = Some 0.; max = None } s in
          let* hi = clamp { Pointwise.Clamp.min = None; max = Some 6. } lo in
          let* d = div_scalar 6. hi in
          mul_scalar ~name:"out" 2. d)
    in
    let* node = first_node g in
    let* op_json = encode_op node.Node.op in
    let* json = encode_graph g in
    let+ decoded = decode_graph json in
    (op_json, decoded)
  in
  Format.printf "%a@."
    (pp_result (fun ppf (op_json, decoded) ->
         Format.fprintf ppf "add_scalar op JSON:@.%s@.%a" op_json
           pp_decoded_graph decoded))
    result;
  [%expect
    {|
    add_scalar op JSON:
    {
      "Add_scalar": {
        "x": 0,
        "scalar": 0.10000000149011612
      }
    }
    decoded graph:
    graph
    inputs: [t0 f32 [C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [C=3] ->[n1]] = add_scalar x=t0 scalar=0.1
      n1: [t2 f32 [C=3] ->[n2]] = clamp x=t1 <-n0 params={min=0; max=none}
      n2: [t3 f32 [C=3] ->[n3]] = clamp x=t2 <-n1 params={min=none; max=6}
      n3: [t4 f32 [C=3] ->[n4]] = div_scalar x=t3 <-n2 scalar=6
      n4: [t5 f32 [C=3]] = mul_scalar x=t4 <-n3 scalar=2
    outputs: [t5 f32 [C=3] <-n4] |}]

let%expect_test "ops Hardtanh/Clone: encode → decode" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"hardtanh_clone" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s1c 3) ~name:"x" () in
          let* h =
            hardtanh { Pointwise.Hardtanh.min_val = -1.; max_val = 1. } x
          in
          clone ~name:"out" h)
    in
    let* json = encode_graph g in
    decode_graph json
  in
  Format.printf "%a@." (pp_result pp_decoded_graph) result;
  [%expect
    {|
    decoded graph:
    graph
    inputs: [t0 f32 [C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [C=3] ->[n1]] = hardtanh x=t0 params={min_val=-1; max_val=1}
      n1: [t2 f32 [C=3]] = clone x=t1 <-n0
    outputs: [t2 f32 [C=3] <-n1] |}]

(* Silu/Sigmoid/Gelu/Hardsigmoid/Hardswish carry no [params] -- byte-identical
   in JSON shape to Relu/Sqrt/Clone -- so this is a codec sanity check
   (op5-impl), not new codec coverage: chained so all five round-trip through
   one graph. *)
let%expect_test "ops Silu/Sigmoid/Gelu/Hardsigmoid/Hardswish: encode → decode" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"group5" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s1c 3) ~name:"x" () in
          let* s = silu x in
          let* g = sigmoid s in
          let* e = gelu Pointwise.Gelu.Exact g in
          let* h = hardsigmoid e in
          hardswish ~name:"out" h)
    in
    let* json = encode_graph g in
    decode_graph json
  in
  Format.printf "%a@." (pp_result pp_decoded_graph) result;
  [%expect
    {|
    decoded graph:
    graph
    inputs: [t0 f32 [C=3] ->[n0]]
    nodes:
      n0: [t1 f32 [C=3] ->[n1]] = silu x=t0
      n1: [t2 f32 [C=3] ->[n2]] = sigmoid x=t1 <-n0
      n2: [t3 f32 [C=3] ->[n3]] = gelu x=t2 <-n1 approximate=none
      n3: [t4 f32 [C=3] ->[n4]] = hardsigmoid x=t3 <-n2
      n4: [t5 f32 [C=3]] = hardswish x=t4 <-n3
    outputs: [t5 f32 [C=3] <-n4] |}]
