(* [Me_source]: the PT2 source view (.ai/model_explorer_design.md).

   Three things here are invisible in a rendering and are why the module is
   shaped as it is: that the input/constant split comes from the SIGNATURE
   rather than from the exporter's name prefixes, that a module level is taken
   relative to its parent, and that an SSA name nothing defines is named rather
   than dropped — a missing edge is a picture that is wrong, not one that is
   incomplete. *)

module ME = Model_explorer
module PT = Pytorch_types
module SM = Schema_runtime.String_map

let limits = Me_limits.Limits.untrusted

let pp_err ppf r =
  Core.Pretty.core_result ~ok:(Fmt.any "ok") ~error:Me_source.pp_error ppf r

(* --- a graph to project --- *)

let tensor name = PT.Argument.Tensor { PT.TensorArgument.name }
let arg ?kind name a = { PT.NamedArgument.name; arg = a; kind }

let meta stack =
  match stack with
  | None -> SM.empty
  | Some s -> SM.add "nn_module_stack" s SM.empty

let node ?stack target inputs outputs =
  {
    PT.Node.target;
    inputs;
    outputs = List.map tensor outputs;
    metadata = meta stack;
    is_hop_single_tensor_return = None;
    name = None;
  }

let tensor_meta name sizes m =
  SM.add name
    {
      PT.TensorMeta.dtype = PT.ScalarType.FLOAT;
      sizes = List.map (fun i -> PT.SymInt.Int i) sizes;
      requires_grad = false;
      device = { PT.Device.type_ = "cpu"; index = None };
      strides = [];
      storage_offset = PT.SymInt.Int 0;
      layout = PT.Layout.Strided;
    }
    m

(* [x] is a user input and [p_conv_weight] a parameter; the two differ ONLY in
   the signature, which is the whole point of reading it. *)
let graph_module ?(nodes_of = fun () -> []) ?(outputs = [ "relu" ]) () =
  let nodes = nodes_of () in
  {
    PT.GraphModule.graph =
      {
        PT.Graph.inputs = [ tensor "x"; tensor "p_conv_weight" ];
        outputs = List.map tensor outputs;
        nodes;
        tensor_values =
          tensor_meta "x" [ 1; 3; 8; 8 ]
            (tensor_meta "conv" [ 1; 4; 8; 8 ] SM.empty);
        sym_int_values = SM.empty;
        sym_bool_values = SM.empty;
        is_single_tensor_return = true;
        custom_obj_values = SM.empty;
        sym_float_values = SM.empty;
      };
    signature =
      {
        PT.GraphSignature.input_specs =
          [
            PT.InputSpec.User_input { PT.UserInputSpec.arg = tensor "x" };
            PT.InputSpec.Parameter
              {
                PT.InputToParameterSpec.arg =
                  { PT.TensorArgument.name = "p_conv_weight" };
                parameter_name = "conv.weight";
              };
          ];
        output_specs = [];
      };
    module_call_graph = [];
    metadata = SM.empty;
    treespec_namedtuple_fields = SM.empty;
  }

let stack =
  "L__self__,,torchvision.models.resnet.ResNet;L__self__layer1,layer1,torch.nn.modules.container.Sequential;L__self__layer1.0,layer1.0,torchvision.models.resnet.BasicBlock;L__self__layer1.0.conv1,layer1.0.conv1,torch.nn.modules.conv.Conv2d"

let two_nodes () =
  [
    node ~stack "torch.ops.aten.convolution.default"
      [
        arg "input" (tensor "x");
        arg "weight" (tensor "p_conv_weight");
        arg "bias" (PT.Argument.None true);
        arg "stride" (PT.Argument.Ints [ 2; 2 ]);
      ]
      [ "conv" ];
    node "torch.ops.aten.relu.default" [ arg "self" (tensor "conv") ] [ "relu" ];
  ]

let show gm =
  match Me_source.graph ~limits gm with
  | Error e -> Format.printf "%a@." pp_err (Error e)
  | Ok g ->
      Printf.printf "graph %s\n" g.ME.Graph.id;
      List.iter
        (fun (n : ME.GraphNode.t) ->
          Printf.printf "  %-18s %-38s ns=%-22s in=[%s]\n" n.ME.GraphNode.id
            n.ME.GraphNode.label n.ME.GraphNode.namespace
            (String.concat " "
               (List.map
                  (fun (e : ME.IncomingEdge.t) ->
                    e.ME.IncomingEdge.sourceNodeId ^ ":"
                    ^ e.ME.IncomingEdge.sourceNodeOutputId ^ "->"
                    ^ e.ME.IncomingEdge.targetNodeInputId)
                  (Option.value n.ME.GraphNode.incomingEdges ~default:[]))))
        g.ME.Graph.nodes

let%expect_test "nodes, boundaries and edges" =
  show (graph_module ~nodes_of:two_nodes ());
  [%expect
    {|
    graph pt2/root
      in:x               input                                  ns=                       in=[]
      const:p_conv_weight constant                               ns=                       in=[]
      root#0             torch.ops.aten.convolution.default     ns=layer1/0/conv1         in=[in:x:0->input const:p_conv_weight:0->weight]
      root#1             torch.ops.aten.relu.default            ns=                       in=[root#0:0->self]
      out:relu           output                                 ns=                       in=[root#1:0->0] |}]

let%expect_test "the input/constant split comes from the signature" =
  (* Both names would be indistinguishable to a rule reading the exporter's
     [p_] prefix; here the SPEC decides, so a parameter the exporter happened to
     name [x2] would still be a constant. *)
  (match Me_source.graph ~limits (graph_module ~nodes_of:two_nodes ()) with
  | Error e -> Format.printf "%a@." pp_err (Error e)
  | Ok g ->
      List.iter
        (fun (n : ME.GraphNode.t) ->
          if n.ME.GraphNode.label = "input" || n.ME.GraphNode.label = "constant"
          then Printf.printf "%s -> %s\n" n.ME.GraphNode.id n.ME.GraphNode.label)
        g.ME.Graph.nodes);
  [%expect {|
    in:x -> input
    const:p_conv_weight -> constant |}]

(* --- the module hierarchy --- *)

let%expect_test "a level is relative to its parent" =
  (* [nn_module_stack]'s [name] field is the FULL dotted path, so the naive
     join is [layer1/layer1.0/layer1.0.conv1] — every level repeating its whole
     prefix. The outermost entry has an empty name and contributes no level. *)
  Format.printf "%a@."
    (Core.Pretty.core_result
       ~ok:(Fmt.list ~sep:(Fmt.any "/") Fmt.string)
       ~error:Me_source.pp_error)
    (Ok (Me_source.module_levels stack));
  [%expect {| layer1/0/conv1 |}]

let%expect_test "a level that does not extend its parent is kept whole" =
  (* Dropping a prefix that is not there would be guessing at a hierarchy the
     exporter did not write. *)
  List.iter
    (fun s ->
      Printf.printf "%-46s -> %s\n" s
        (String.concat "/" (Me_source.module_levels s)))
    [
      "a,,C;b,other,D";
      (* no comma at all: the entry names nothing and is skipped *)
      "malformed";
      (* every level empty: the node sits at the root *)
      "a,,C";
    ];
  [%expect
    {|
    a,,C;b,other,D                                 -> other
    malformed                                      ->
    a,,C                                           -> |}]

let%expect_test "a module name containing a slash cannot move a node" =
  (* The renderer splits [namespace] on ['/'], so an unencoded name would put
     the node one level deeper than the stack says. Invisible in any model whose
     modules happen to be plainly named. *)
  let stack = "a,,C;b,we/ird,D" in
  let gm =
    graph_module
      ~nodes_of:(fun () ->
        [
          node ~stack "torch.ops.aten.relu.default"
            [ arg "self" (tensor "x") ]
            [ "relu" ];
        ])
      ()
  in
  show gm;
  [%expect
    {|
    graph pt2/root
      in:x               input                                  ns=                       in=[]
      const:p_conv_weight constant                               ns=                       in=[]
      root#0             torch.ops.aten.relu.default            ns=we%2Fird               in=[in:x:0->self]
      out:relu           output                                 ns=                       in=[root#0:0->0] |}]

(* --- the ids the navigation pairs against --- *)

let%expect_test "node_ids is the left-pane universe" =
  (* [Me_pt2] needs these and cannot derive them from the sidecar, which is
     indexed by NATIVE node. Total, so a projection failure does not take the
     mapping with it. *)
  List.iter print_endline
    (Me_source.node_ids
       (graph_module ~nodes_of:two_nodes ()).PT.GraphModule.graph);
  [%expect {|
    root#0
    root#1 |}]

(* --- failures --- *)

let%expect_test "an SSA name nothing defines is named, not dropped" =
  let gm =
    graph_module
      ~nodes_of:(fun () ->
        [
          node "torch.ops.aten.relu.default"
            [ arg "self" (tensor "absent") ]
            [ "relu" ];
        ])
      ()
  in
  Format.printf "%a@." pp_err (Me_source.graph ~limits gm);
  [%expect {| SSA value "absent" has no producer |}]

let%expect_test "a sym_int operand draws no edge" =
  (* It names an entry in [sym_int_values], not a node output. Drawing one would
     assert a dataflow dependency the graph does not have — and would then fail
     as an unknown producer, which is the wrong error for a correct graph. *)
  let gm =
    graph_module
      ~nodes_of:(fun () ->
        [
          node "torch.ops.aten.expand.default"
            [
              arg "self" (tensor "x");
              arg "size" (PT.Argument.Sym_ints [ PT.SymIntArgument.Name "s0" ]);
            ]
            [ "relu" ];
        ])
      ()
  in
  show gm;
  [%expect
    {|
    graph pt2/root
      in:x               input                                  ns=                       in=[]
      const:p_conv_weight constant                               ns=                       in=[]
      root#0             torch.ops.aten.expand.default          ns=                       in=[in:x:0->self]
      out:relu           output                                 ns=                       in=[root#0:0->0] |}]

let%expect_test "the ceilings, checked before the walks" =
  let tight =
    Core.or_raise Me_limits.pp_error
      (Me_limits.Limits.create ~max_nodes_per_graph:2 limits)
  in
  Format.printf "nodes %a@." pp_err
    (Me_source.graph ~limits:tight (graph_module ~nodes_of:two_nodes ()));
  let tight_e =
    Core.or_raise Me_limits.pp_error
      (Me_limits.Limits.create ~max_edges_per_graph:1 limits)
  in
  Format.printf "edges %a@." pp_err
    (Me_source.graph ~limits:tight_e (graph_module ~nodes_of:two_nodes ()));
  [%expect
    {|
    nodes source graph nodes = 5 is over the ceiling
    edges source graph edges = 4 is over the ceiling |}]

let%expect_test "an over-long argument render is capped and says so" =
  let tight =
    Core.or_raise Me_limits.pp_error
      (Me_limits.Limits.create ~max_attr_chars:4 limits)
  in
  (match
     Me_source.graph ~limits:tight (graph_module ~nodes_of:two_nodes ())
   with
  | Error e -> Format.printf "%a@." pp_err (Error e)
  | Ok g ->
      List.iter
        (fun (n : ME.GraphNode.t) ->
          if n.ME.GraphNode.id = "root#0" then
            List.iter
              (fun (a : ME.NodeAttribute.t) ->
                Printf.printf "%s=%s\n" a.ME.NodeAttribute.key
                  (match a.ME.NodeAttribute.value with
                  | ME.NodeAttributeValue.Str s -> s
                  | _ -> "<non-string>"))
              (Option.value n.ME.GraphNode.attrs ~default:[]))
        g.ME.Graph.nodes);
  [%expect
    {|
    input=x
    weight=p_co
    weight_truncated=true
    bias=none
    stride=[2,2
    stride_truncated=true |}]

(* --- output metadata --- *)

let%expect_test "shape and dtype come from tensor_values where they exist" =
  (match Me_source.graph ~limits (graph_module ~nodes_of:two_nodes ()) with
  | Error e -> Format.printf "%a@." pp_err (Error e)
  | Ok g ->
      List.iter
        (fun (n : ME.GraphNode.t) ->
          List.iter
            (fun (m : ME.MetadataItem.t) ->
              Printf.printf "%-18s #%s %s\n" n.ME.GraphNode.id
                m.ME.MetadataItem.id
                (String.concat " "
                   (List.map
                      (fun (kv : ME.KeyValue.t) ->
                        kv.ME.KeyValue.key ^ "=" ^ kv.ME.KeyValue.value)
                      m.ME.MetadataItem.attrs)))
            (Option.value n.ME.GraphNode.outputsMetadata ~default:[]))
        g.ME.Graph.nodes);
  [%expect
    {|
    in:x               #0 ssa=x shape=[1,3,8,8] dtype=FLOAT
    const:p_conv_weight #0 ssa=p_conv_weight
    root#0             #0 ssa=conv shape=[1,4,8,8] dtype=FLOAT
    root#1             #0 ssa=relu |}]
