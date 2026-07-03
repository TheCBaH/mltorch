(* Expect tests for Graph_json: JSON serialization/deserialization of native
   ops, graphs, and tensor payloads. Each test either:
   - builds a value, pretty-prints it, encodes to JSON, then decodes the JSON
     and pretty-prints the result (verifying roundtrip);
   - or starts from a JSON literal and decodes then pretty-prints. *)

open Graph_ir

type error =
  [ `Build of Graph_builder.error | `Json of string | `Message of string ]

let pp_error ppf : [< error ] -> unit = function
  | `Build e -> Graph_builder.pp_error ppf e
  | `Json msg | `Message msg -> Format.pp_print_string ppf msg

let pp_result pp_ok =
  Fmt.result ~ok:pp_ok ~error:(fun ppf e -> pp_error ppf e.Core.Error.kind)

let lift_build (r : ('a, Graph_builder.error) Core.result) :
    ('a, error) Core.result =
  Core.map_error (fun e -> `Build e) r

let lift_json (r : ('a, string) result) : ('a, error) Core.result =
  match r with Ok x -> Core.return x | Error e -> Core.fail (`Json e)

let s n t d h w c = Vec6.shape ~n ~t ~d ~h ~w ~c
let s1c c = s 1 1 1 1 1 c
let chan c = Dim.to_int (Vec6.get c Axis.C)

let conv_axis ~kernel ~stride ~pad : Conv.Conv2d.axis_window =
  {
    kernel = Dim.extent kernel;
    stride = Op_config.Pos.of_int stride;
    pad_before = Op_config.Nonneg.of_int pad;
    pad_after = Op_config.Nonneg.of_int pad;
    dilation = Op_config.Pos.of_int 1;
  }

let encode_graph g = lift_json (Graph_json.encode_graph ~format:Jsont.Indent g)
let decode_graph s = lift_json (Graph_json.decode_graph s)

let encode_tensor ?max_elts t =
  lift_json (Graph_json.encode_tensor ~format:Jsont.Indent ?max_elts t)

let decode_tensor s = lift_json (Graph_json.decode_tensor s)

let encode_op op =
  lift_json
    (Jsont_bytesrw.encode_string ~format:Jsont.Indent Graph_ir.op_jsont op)

let first_node g =
  match g.Graph.nodes with
  | node :: _ -> Core.return node
  | [] -> Core.fail (`Message "expected graph to contain at least one node")

let pp_graph_json_and_op ppf (graph_json, op_json) =
  Format.fprintf ppf "graph JSON:@.%s@.op JSON:@.%s" graph_json op_json

let pp_json_and_graph ppf (json, g) =
  Format.fprintf ppf "JSON:@.%s@.decoded graph:@.%a" json Graph_ir.pp g

let pp_original_and_graph ppf (g, decoded) =
  Format.fprintf ppf "original:@.%a@.decoded:@.%a" Graph_ir.pp g Graph_ir.pp
    decoded

let pp_decoded_graph ppf g =
  Format.fprintf ppf "decoded graph:@.%a" Graph_ir.pp g

let pp_decoded ppf g = Format.fprintf ppf "decoded:@.%a" Graph_ir.pp g

let pp_original_and_tensor ppf (original, decoded) =
  Format.fprintf ppf "original: %a@.decoded:  %a" Tensor.pp original Tensor.pp
    decoded

let pp_full_and_elided ppf (full_json, elided_json) =
  Format.fprintf ppf "full (8 elts):@.%s@.elided (max_elts=4):@.%s" full_json
    elided_json

(* ---- ops ------------------------------------------------------------------ *)

let%expect_test "op Add: encode → JSON" =
  let result =
    let open Core.Syntax in
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
    Core.return (graph_json, op_json)
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
      "name": "add",
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
      "outputs": [
        2
      ],
      "tensors": [
        {
          "fmt": "f32",
          "id": 0,
          "name": "a",
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
          "name": "b",
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
          "name": "out",
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
    let open Core.Syntax in
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
    Core.return (json, g2)
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
      "name": "conv",
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
      "outputs": [
        3
      ],
      "tensors": [
        {
          "fmt": "f32",
          "id": 0,
          "name": "x",
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
          "name": "w",
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
          "name": "b",
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
          "name": "y",
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
    graph conv
    inputs:
      [t0 x:f32 [H=4 W=4 C=8], t1 w:f32 [N=16 T=1 D=1 H=3 W=3 C=8],
       t2 b:f32 [C=16]]
    nodes:
      n0: [t3 y:f32 [H=4 W=4 C=16]] =
        conv2d
          x=t0(x)
          weight=t1(w)
          bias=t2(b)
          params={h={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 w={kernel=3; stride=1; pad_before=1; pad_after=1; dilation=1};
                 in_channels=8;
                 groups=1}
    outputs: [t3 y:f32 [H=4 W=4 C=16]] |}]

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
    let open Core.Syntax in
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
    Core.return (g, g2)
  in
  Format.printf "%a@." (pp_result pp_original_and_graph) result;
  [%expect
    {|
    original:
    graph conv_nb
    inputs: [t0 x:f32 [C=4], t1 w:f32 [N=4 T=1 D=1 H=1 W=1 C=4]]
    nodes:
      n0: [t2 y:f32 [C=4]] =
        conv2d
          x=t0(x)
          weight=t1(w)
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=4;
                 groups=1}
    outputs: [t2 y:f32 [C=4]]
    decoded:
    graph conv_nb
    inputs: [t0 x:f32 [C=4], t1 w:f32 [N=4 T=1 D=1 H=1 W=1 C=4]]
    nodes:
      n0: [t2 y:f32 [C=4]] =
        conv2d
          x=t0(x)
          weight=t1(w)
          bias=none
          params={h={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 w={kernel=1; stride=1; pad_before=0; pad_after=0; dilation=1};
                 in_channels=4;
                 groups=1}
    outputs: [t2 y:f32 [C=4]] |}]

let%expect_test "op Permute: encode → decode" =
  let perm = Axis.[ (N, N); (T, T); (D, D); (H, W); (W, C); (C, H) ] in
  let result =
    let open Core.Syntax in
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
    graph perm
    inputs: [t0 x:f32 [H=2 W=3 C=4]]
    nodes:
      n0: [t1 y:f32 [H=3 W=4 C=2]] = permute x=t0(x) perm=[H<-W, W<-C, C<-H]
    outputs: [t1 y:f32 [H=3 W=4 C=2]] |}]

let%expect_test "graph with Mean op: encode → decode → pretty-print" =
  let result =
    let open Core.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"mean_hw" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s 1 1 1 7 7 64) ~name:"x" () in
          mean ~name:"out"
            Reduce.Mean.{ dims = [ Axis.H; Axis.W ]; keepdim = false }
            x)
    in
    let* json = encode_graph g in
    decode_graph json
  in
  Format.printf "%a@." (pp_result pp_decoded) result;
  [%expect
    {|
    decoded:
    graph mean_hw
    inputs: [t0 x:f32 [H=7 W=7 C=64]]
    nodes:
      n0: [t1 out:f32 [C=64]] = mean x=t0(x) params={dims=[H, W]; keepdim=false}
    outputs: [t1 out:f32 [C=64]] |}]

let%expect_test "nested subgraph: encode → decode → pretty-print" =
  let result =
    let open Core.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"outer" ~outputs:(fun outs -> outs)
          @@
          let* x = input ~shape:(s1c 4) ~name:"x" () in
          let* y = input ~shape:(s1c 4) ~name:"y" () in
          let* sg =
            subgraph ~name:"add_relu"
              (let* a = input ~shape:(s1c 4) ~name:"a" () in
               let* b = input ~shape:(s1c 4) ~name:"b" () in
               let* t = add ~name:"sum" a b in
               let* r = relu ~name:"r" t in
               return [ r ])
          in
          invoke ~names:[ "out" ] sg [ x; y ])
    in
    let* json = encode_graph g in
    let* g2 = decode_graph json in
    Core.return (g, g2)
  in
  Format.printf "%a@." (pp_result pp_original_and_graph) result;
  [%expect
    {|
    original:
    graph outer
    inputs: [t0 x:f32 [C=4], t1 y:f32 [C=4]]
    nodes:
      n2: [t6 out:f32 [C=4]] = subgraph add_relu args=[t0(x), t1(y)]
        graph add_relu
        inputs: [t2 a:f32 [C=4], t3 b:f32 [C=4]]
        nodes:
          n0: [t4 sum:f32 [C=4]] = add a=t2(a) b=t3(b)
          n1: [t5 r:f32 [C=4]] = relu x=t4(sum)
        outputs: [t5 r:f32 [C=4]]
    outputs: [t6 out:f32 [C=4]]
    decoded:
    graph outer
    inputs: [t0 x:f32 [C=4], t1 y:f32 [C=4]]
    nodes:
      n2: [t6 out:f32 [C=4]] = subgraph add_relu args=[t0(x), t1(y)]
        graph add_relu
        inputs: [t2 a:f32 [C=4], t3 b:f32 [C=4]]
        nodes:
          n0: [t4 sum:f32 [C=4]] = add a=t2(a) b=t3(b)
          n1: [t5 r:f32 [C=4]] = relu x=t4(sum)
        outputs: [t5 r:f32 [C=4]]
    outputs: [t6 out:f32 [C=4]] |}]

(* ---- tensor payload ------------------------------------------------------- *)

let%expect_test "tensor: encode with Array payload" =
  let t = Tensor.materialize (s1c 4) (fun c -> float_of_int (chan c)) in
  Format.printf "%a@." (pp_result Format.pp_print_string) (encode_tensor t);
  [%expect
    {|
    {
      "data": {
        "Array": [
          0,
          1,
          2,
          3
        ]
      },
      "fmt": "f32",
      "quant": null,
      "shape": [
        1,
        1,
        1,
        1,
        1,
        4
      ]
    } |}]

let%expect_test "tensor: encode → decode → verify values" =
  let original =
    Tensor.materialize (s1c 4) (fun c -> float_of_int (chan c) *. 0.5)
  in
  let result =
    let open Core.Syntax in
    let* json = encode_tensor original in
    let* decoded = decode_tensor json in
    Core.return (original, decoded)
  in
  Format.printf "%a@." (pp_result pp_original_and_tensor) result;
  [%expect
    {|
    original: tensor f32 [C=4] {0, 0.5, 1, 1.5}
    decoded:  tensor f32 [C=4] {0, 0.5, 1, 1.5} |}]

let%expect_test "tensor: payload elided when numel exceeds max_elts" =
  let t = Tensor.materialize (s1c 8) (fun c -> float_of_int (chan c)) in
  let result =
    let open Core.Syntax in
    let* full_json = encode_tensor t in
    let* elided_json = encode_tensor ~max_elts:4 t in
    Core.return (full_json, elided_json)
  in
  Format.printf "%a@." (pp_result pp_full_and_elided) result;
  [%expect
    {|
    full (8 elts):
    {
      "data": {
        "Array": [
          0,
          1,
          2,
          3,
          4,
          5,
          6,
          7
        ]
      },
      "fmt": "f32",
      "quant": null,
      "shape": [
        1,
        1,
        1,
        1,
        1,
        8
      ]
    }
    elided (max_elts=4):
    {
      "data": {
        "None": null
      },
      "fmt": "f32",
      "quant": null,
      "shape": [
        1,
        1,
        1,
        1,
        1,
        8
      ]
    } |}]

let%expect_test "tensor: decode JSON literal" =
  let json =
    {|{"data":{"Array":[1,2,3]},"fmt":"f32","quant":null,"shape":[1,1,1,1,1,3]}|}
  in
  Format.printf "%a@." (pp_result Tensor.pp) (decode_tensor json);
  [%expect {| tensor f32 [C=3] {1, 2, 3} |}]

let%expect_test "tensor: f32 special values roundtrip" =
  let t =
    Tensor.materialize (s1c 3) (fun c ->
        match chan c with 0 -> Float.nan | 1 -> 1.0 /. 0.0 | _ -> -0.0)
  in
  Format.printf "%a@." (pp_result Format.pp_print_string) (encode_tensor t);
  [%expect
    {|
    {
      "data": {
        "Array": [
          "0x7fc00000",
          "0x7f800000",
          "0x80000000"
        ]
      },
      "fmt": "f32",
      "quant": null,
      "shape": [
        1,
        1,
        1,
        1,
        1,
        3
      ]
    } |}]
