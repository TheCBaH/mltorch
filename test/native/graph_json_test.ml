(* Expect tests for Graph_json: JSON serialization/deserialization of native
   ops, graphs, and tensor payloads. Each test either:
   - builds a value, pretty-prints it, encodes to JSON, then decodes the JSON
     and pretty-prints the result (verifying roundtrip);
   - or starts from a JSON literal and decodes then pretty-prints. *)

open Graph_ir

let s n t d h w c = Vec6.shape ~n ~t ~d ~h ~w ~c
let s1c c = s 1 1 1 1 1 c
let chan c = Dim.to_int (Vec6.get c Axis.C)
let pf fmt = Format.printf fmt

let encode_graph_exn g =
  match Graph_json.encode_graph ~format:Jsont.Indent g with
  | Ok s -> s
  | Error e -> failwith e

let decode_graph_exn s =
  match Graph_json.decode_graph s with Ok g -> g | Error e -> failwith e

let encode_tensor_exn ?max_elts t =
  match Graph_json.encode_tensor ~format:Jsont.Indent ?max_elts t with
  | Ok s -> s
  | Error e -> failwith e

let decode_tensor_exn s =
  match Graph_json.decode_tensor s with Ok t -> t | Error e -> failwith e

(* ---- ops ------------------------------------------------------------------ *)

let%expect_test "op Add: encode → JSON" =
  let g =
    Graph_builder.(
      build ~name:"add" ~outputs:(fun r -> [ r ])
      @@
      let* a = input ~shape:(s1c 3) ~name:"a" () in
      let* b = input ~shape:(s1c 3) ~name:"b" () in
      add ~name:"out" a b)
  in
  let node = List.hd g.Graph.nodes in
  pf "graph JSON:@.%s@." (encode_graph_exn g);
  (match
     Jsont_bytesrw.encode_string ~format:Jsont.Indent Graph_ir.op_jsont
       node.Node.op
   with
  | Ok s -> pf "op JSON:@.%s@." s
  | Error e -> pf "error: %s@." e);
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
        kernel = Op_config.Hw.{ h = Dim.extent 3; w = Dim.extent 3 };
        in_channels = Dim.extent 8;
        stride =
          Op_config.Hw.
            { h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int 1 };
        pad =
          Op_config.Hw.
            { h = Op_config.Nonneg.of_int 1; w = Op_config.Nonneg.of_int 1 };
      }
  in
  let g =
    Graph_builder.(
      build ~name:"conv" ~outputs:(fun r -> [ r ])
      @@
      let* x = input ~shape:(s 1 1 1 4 4 8) ~name:"x" () in
      let* w = input ~shape:(s 16 1 1 3 3 8) ~name:"w" () in
      let* b = input ~shape:(s1c 16) ~name:"b" () in
      conv2d ~name:"y" conv_params ~x ~weight:w ~bias:b ())
  in
  let json = encode_graph_exn g in
  pf "JSON:@.%s@." json;
  let g2 = decode_graph_exn json in
  pf "decoded graph:@.%a@." Graph_ir.pp g2;
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
                "kernel": {
                  "h": 3,
                  "w": 3
                },
                "in_channels": 8,
                "pad": {
                  "h": 1,
                  "w": 1
                },
                "stride": {
                  "h": 1,
                  "w": 1
                }
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
          params={kernel={h=3; w=3};
                 in_channels=8;
                 stride={h=1; w=1};
                 pad={h=1; w=1}}
    outputs: [t3 y:f32 [H=4 W=4 C=16]] |}]

let%expect_test "op Conv2d no bias: optional field absent in JSON" =
  let conv_params =
    Conv.Conv2d.
      {
        kernel = Op_config.Hw.{ h = Dim.extent 1; w = Dim.extent 1 };
        in_channels = Dim.extent 4;
        stride =
          Op_config.Hw.
            { h = Op_config.Pos.of_int 1; w = Op_config.Pos.of_int 1 };
        pad =
          Op_config.Hw.
            { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
      }
  in
  let g =
    Graph_builder.(
      build ~name:"conv_nb" ~outputs:(fun r -> [ r ])
      @@
      let* x = input ~shape:(s1c 4) ~name:"x" () in
      let* w = input ~shape:(s 4 1 1 1 1 4) ~name:"w" () in
      conv2d ~name:"y" conv_params ~x ~weight:w ())
  in
  let json = encode_graph_exn g in
  let g2 = decode_graph_exn json in
  pf "original:@.%a@." Graph_ir.pp g;
  pf "decoded:@.%a@." Graph_ir.pp g2;
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
          params={kernel={h=1; w=1};
                 in_channels=4;
                 stride={h=1; w=1};
                 pad={h=0; w=0}}
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
          params={kernel={h=1; w=1};
                 in_channels=4;
                 stride={h=1; w=1};
                 pad={h=0; w=0}}
    outputs: [t2 y:f32 [C=4]] |}]

let%expect_test "op Permute: encode → decode" =
  let perm = Axis.[ (N, N); (T, T); (D, D); (H, W); (W, C); (C, H) ] in
  let g =
    Graph_builder.(
      build ~name:"perm" ~outputs:(fun r -> [ r ])
      @@
      let* x = input ~shape:(s 1 1 1 2 3 4) ~name:"x" () in
      permute ~name:"y" perm x)
  in
  let json = encode_graph_exn g in
  let g2 = decode_graph_exn json in
  pf "decoded graph:@.%a@." Graph_ir.pp g2;
  [%expect
    {|
    decoded graph:
    graph perm
    inputs: [t0 x:f32 [H=2 W=3 C=4]]
    nodes:
      n0: [t1 y:f32 [H=3 W=4 C=2]] = permute x=t0(x) perm=[H<-W, W<-C, C<-H]
    outputs: [t1 y:f32 [H=3 W=4 C=2]] |}]

let%expect_test "graph with Mean op: encode → decode → pretty-print" =
  let g =
    Graph_builder.(
      build ~name:"mean_hw" ~outputs:(fun r -> [ r ])
      @@
      let* x = input ~shape:(s 1 1 1 7 7 64) ~name:"x" () in
      mean ~name:"out"
        Reduce.Mean.{ dims = [ Axis.H; Axis.W ]; keepdim = false }
        x)
  in
  let json = encode_graph_exn g in
  let g2 = decode_graph_exn json in
  pf "decoded:@.%a@." Graph_ir.pp g2;
  [%expect
    {|
    decoded:
    graph mean_hw
    inputs: [t0 x:f32 [H=7 W=7 C=64]]
    nodes:
      n0: [t1 out:f32 [C=64]] = mean x=t0(x) params={dims=[H, W]; keepdim=false}
    outputs: [t1 out:f32 [C=64]] |}]

let%expect_test "nested subgraph: encode → decode → pretty-print" =
  let g =
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
  let json = encode_graph_exn g in
  let g2 = decode_graph_exn json in
  pf "original:@.%a@." Graph_ir.pp g;
  pf "decoded:@.%a@." Graph_ir.pp g2;
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
  pf "%s@." (encode_tensor_exn t);
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
  let json = encode_tensor_exn original in
  let decoded = decode_tensor_exn json in
  pf "original: %a@." Tensor.pp original;
  pf "decoded:  %a@." Tensor.pp decoded;
  [%expect
    {|
    original: tensor f32 [C=4] {0, 0.5, 1, 1.5}
    decoded:  tensor f32 [C=4] {0, 0.5, 1, 1.5} |}]

let%expect_test "tensor: payload elided when numel exceeds max_elts" =
  let t = Tensor.materialize (s1c 8) (fun c -> float_of_int (chan c)) in
  pf "full (8 elts):@.%s@." (encode_tensor_exn t);
  pf "elided (max_elts=4):@.%s@." (encode_tensor_exn ~max_elts:4 t);
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
  pf "%a@." Tensor.pp (decode_tensor_exn json);
  [%expect {| tensor f32 [C=3] {1, 2, 3} |}]

let%expect_test "tensor: f32 special values roundtrip" =
  let t =
    Tensor.materialize (s1c 3) (fun c ->
        match chan c with
        | 0 -> Float.nan
        | 1 -> 1.0 /. 0.0 (* +inf *)
        | _ -> -0.0)
  in
  pf "%s@." (encode_tensor_exn t);
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
