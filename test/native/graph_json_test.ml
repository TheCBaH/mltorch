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

let pp_result pp_ok = Core.Pretty.err_result ~ok:pp_ok ~error:pp_error

let lift_build (r : ('a, Graph_builder.error) Err.t) : ('a, error) Err.t =
  Err.map_error (fun e -> `Build e) r

let lift_json (r : ('a, string) result) : ('a, error) Err.t =
  match r with Ok x -> Err.return x | Error e -> Err.fail (`Json e)

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
  | node :: _ -> Err.return node
  | [] -> Err.fail (`Message "expected graph to contain at least one node")

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

let%expect_test "graph: captured input kind survives JSON roundtrip" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"captured" ~outputs:(fun r -> [ r ])
          @@
          let* x = input ~shape:(s1c 2) ~name:"x" () in
          let* weight = constant ~shape:(s1c 2) ~name:"weight" () in
          add x weight)
    in
    let* json = encode_graph g in
    let* decoded = decode_graph json in
    match List.nth decoded.Graph.inputs 1 |> Graph_ir.input_kind decoded with
    | Input.Constant -> Err.return (String.contains json 'c')
    | Input.Input -> Err.fail (`Message "constant decoded as input")
  in
  Format.printf "%a@."
    (pp_result (fun ppf has_constants ->
         Format.fprintf ppf "input_constants=%b" has_constants))
    result;
  [%expect {| input_constants=true |}]

(* ---- ops ------------------------------------------------------------------ *)

(* [op_name] and [op_jsont]'s case tag are one vocabulary, asserted rather than
   claimed: the tag is recovered from the encoding itself (a single-key object,
   [Json_util.single ~case]) and compared with [op_name] for every op the
   fixture builds. Changing one without the other fails here.

   The ops are listed explicitly because [op_registry] is private and its [OP]
   signature carries no sample payload, so there is no way to derive a fixture
   per registry entry — a registry-driven version of this test cannot be
   written from outside the module. [Discard] is included for the opposite
   reason: it owns no registry module at all and is the one arm that has to be
   spelled out in the implementation. *)
(* The encoding is one object with exactly one member, so the tag is the first
   quoted string in it. Read textually rather than through a codec, so the
   comparison stays independent of whatever [op_jsont] does inside the case. *)
let case_tag json =
  match String.index_opt json '"' with
  | None -> "<no tag>"
  | Some i -> (
      match String.index_from_opt json (i + 1) '"' with
      | None -> "<no tag>"
      | Some j -> String.sub json (i + 1) (j - i - 1))

let%expect_test "op_name agrees with the JSON case tag, Discard included" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"names" ~outputs:(fun r -> [ r ])
          @@
          let* a = input ~shape:(s1c 3) ~name:"a" () in
          let* b = input ~shape:(s1c 3) ~name:"b" () in
          let* sum = add ~name:"sum" a b in
          let* prod = mul ~name:"prod" sum b in
          let* act = relu ~name:"act" prod in
          let* dead = sub ~name:"dead" act a in
          let* () = discard dead in
          relu ~name:"out" act)
    in
    Err.List.map
      (fun (node : Graph_ir.node) ->
        let+ json = encode_op node.Node.op in
        (Graph_ir.op_name node.Node.op, case_tag json))
      g.Graph.nodes
  in
  (match result with
  | Error e -> Format.printf "%a@." pp_error (Err.Error.kind e)
  | Ok rows ->
      List.iter
        (fun (name, tag) ->
          Printf.printf "%-10s tag=%-10s agree=%b\n" name tag (name = tag))
        rows);
  [%expect
    {|
    Add        tag=Add        agree=true
    Mul        tag=Mul        agree=true
    Relu       tag=Relu       agree=true
    Sub        tag=Sub        agree=true
    Discard    tag=Discard    agree=true
    Relu       tag=Relu       agree=true
    |}]

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
let%expect_test "ops Add_scalar/Clamp/Div_scalar: encode → decode" =
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
          div_scalar ~name:"out" 6. hi)
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
      n3: [t4 f32 [C=3]] = div_scalar x=t3 <-n2 scalar=6
    outputs: [t4 f32 [C=3] <-n3] |}]

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

let%expect_test "graph with Mean op: encode → decode → pretty-print" =
  let result =
    let open Err.Syntax in
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
    graph
    inputs: [t0 f32 [H=7 W=7 C=64] ->[n0]]
    nodes:
      n0: [t1 f32 [C=64]] = mean x=t0 params={dims=[H, W]; keepdim=false}
    outputs: [t1 f32 [C=64] <-n0] |}]

let%expect_test "nested group: encode → decode → pretty-print" =
  let result =
    let open Err.Syntax in
    let* g =
      lift_build
        Graph_builder.(
          build ~name:"outer" ~outputs:(fun outs -> outs)
          @@
          let* x = input ~shape:(s1c 4) ~name:"x" () in
          let* y = input ~shape:(s1c 4) ~name:"y" () in
          group ~label:"add_relu"
            (let* t = add ~name:"sum" x y in
             let* r = relu ~name:"r" t in
             return [ r ]))
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
    inputs: [t0 f32 [C=4] ->[n0], t1 f32 [C=4] ->[n0]]
    nodes:
      group g1 add_relu:
        n0: [t2 f32 [C=4] ->[n1]] = add a=t0 b=t1
        n1: [t3 f32 [C=4]] = relu x=t2 <-n0
    outputs: [t3 f32 [C=4] <-n1]
    decoded:
    graph
    inputs: [t0 f32 [C=4] ->[n0], t1 f32 [C=4] ->[n0]]
    nodes:
      group g1 add_relu:
        n0: [t2 f32 [C=4] ->[n1]] = add a=t0 b=t1
        n1: [t3 f32 [C=4]] = relu x=t2 <-n0
    outputs: [t3 f32 [C=4] <-n1] |}]

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
    let open Err.Syntax in
    let* json = encode_tensor original in
    let* decoded = decode_tensor json in
    Err.return (original, decoded)
  in
  Format.printf "%a@." (pp_result pp_original_and_tensor) result;
  [%expect
    {|
    original: tensor f32 [C=4] {0, 0.5, 1, 1.5}
    decoded:  tensor f32 [C=4] {0, 0.5, 1, 1.5} |}]

let%expect_test "tensor: payload elided when numel exceeds max_elts" =
  let t = Tensor.materialize (s1c 8) (fun c -> float_of_int (chan c)) in
  let result =
    let open Err.Syntax in
    let* full_json = encode_tensor t in
    let* elided_json = encode_tensor ~max_elts:4 t in
    Err.return (full_json, elided_json)
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
