(* Graph-structural JSON codec properties: constant-vs-input roundtrip, op
   tag agreement with [op_name], and graphs with a reduction (Mean) or a
   nested group. Split from graph_json_test.ml. *)

open Graph_ir
open Graph_json_fixtures

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
          let* _slices = unbind { Split.Unbind.axis = Axis.C } act in
          let* _pieces =
            split_with_sizes
              { Split.Split_with_sizes.axis = Axis.C; sizes = [ 1; 2 ] }
              act
          in
          let* selected =
            select { Split.Select.axis = Axis.C; index = 0 } act
          in
          let* stacked =
            stack { Concat.Stack.axis = Axis.C } [ selected; selected ]
          in
          let* () = discard stacked in
          let* padded =
            pad
              {
                Pad.Pad.pads = [ (Axis.C, { Pad.Pad.before = 1; after = 0 }) ];
                mode = Pad.Pad.Constant 0.;
              }
              act
          in
          let* narrowed =
            slice
              { Split.Slice.axis = Axis.C; start = 1; stop = 3; step = pos 1 }
              padded
          in
          let* normed =
            layer_norm
              { Norm.LayerNorm.dims = [ Axis.C ]; eps = 1e-6 }
              ~x:narrowed ()
          in
          let* grouped =
            group_norm
              {
                Norm.GroupNorm.channel = Axis.C;
                groups = Op_config.Pos.of_int 2;
                eps = 1e-6;
              }
              ~x:normed ()
          in
          relu ~name:"out" grouped)
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
    Unbind     tag=Unbind     agree=true
    SplitWithSizes tag=SplitWithSizes agree=true
    Select     tag=Select     agree=true
    Stack      tag=Stack      agree=true
    Discard    tag=Discard    agree=true
    Pad        tag=Pad        agree=true
    Slice      tag=Slice      agree=true
    Layer_norm tag=Layer_norm agree=true
    Group_norm tag=Group_norm agree=true
    Relu       tag=Relu       agree=true
    |}]

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
