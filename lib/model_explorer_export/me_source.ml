(* Project the exported program's own graph. See the .mli. *)

module ME = Model_explorer
module PT = Pytorch_types
module A = Pytorch_types.Argument
module SM = Schema_runtime.String_map

type error =
  [ Me_ids.error | Me_limits.over_limit_error | `Unknown_producer of string ]

let pp_error fmt : [< error ] -> unit = function
  | #Me_ids.error as e -> Me_ids.pp_error fmt e
  | `Over_limit o -> Me_limits.Over_limit.pp fmt o
  | `Unknown_producer name -> Fmt.pf fmt "SSA value %S has no producer" name

let over_limit = Me_limits.check ~scope:Me_limits.Scope.Source_graph
let tname (t : PT.TensorArgument.t) = t.PT.TensorArgument.name

(* Which SSA values an argument READS.

   Total over [Argument.t]. A constructor added upstream that carried a tensor
   would otherwise lose its edges silently, and a missing edge is a picture that
   is WRONG rather than one that is incomplete — the reader cannot tell the
   difference by looking. *)
let rec tensor_names (a : A.t) =
  match a with
  | A.Tensor t -> [ tname t ]
  | Tensors ts -> List.map tname ts
  | Nested_tensors tss -> List.concat_map (List.map tname) tss
  | Optional_tensor o -> optional_tensor_name o
  | Optional_tensors os -> List.concat_map optional_tensor_name os
  | String_to_argument m -> SM.fold (fun _ v acc -> acc @ tensor_names v) m []
  (* Values and non-tensor references. A [sym_int] names an entry in
     [sym_int_values] and a graph argument a nested body; neither is a node
     output here, and drawing an edge for one would assert a dataflow
     dependency the graph does not have. *)
  | None _ | Int _ | Ints _ | Int_lists _ | Float _ | Floats _ | Float_lists _
  | String _ | Strings _ | Bool _ | Bools _ | Sym_int _ | Sym_ints _
  | Sym_bool _ | Sym_bools _ | Sym_float _ | Sym_floats _ | Scalar_type _
  | Memory_format _ | Layout _ | Device _ | Graph _ | Custom_obj _ | Operator _
  | Complex _ ->
      []

and optional_tensor_name (o : PT.OptionalTensorArgument.t) =
  match o with
  | PT.OptionalTensorArgument.Tensor t -> [ tname t ]
  | None _ -> []

(* --- rendering an argument ---------------------------------------------- *)

(* [Fmt.comma] carries a BREAK HINT, and an attribute value is a single line the
   renderer displays verbatim: a wrapped shape puts a newline inside it, and the
   cap then measures a string that also has to pay for the break. Every
   separator here is a literal. *)
let comma = Fmt.any ","
let pp_seq pp fmt xs = Fmt.pf fmt "[%a]" (Fmt.list ~sep:comma pp) xs
let pp_float fmt f = Fmt.pf fmt "%g" f
let pp_quoted fmt s = Fmt.pf fmt "%S" s

let pp_layout fmt (l : PT.Layout.t) =
  Fmt.string fmt
    (match l with
    | PT.Layout.Unknown -> "unknown"
    | SparseCoo -> "sparse_coo"
    | SparseCsr -> "sparse_csr"
    | SparseCsc -> "sparse_csc"
    | SparseBsr -> "sparse_bsr"
    | SparseBsc -> "sparse_bsc"
    | Mkldnn -> "mkldnn"
    | Strided -> "strided")

let pp_memory_format fmt (m : PT.MemoryFormat.t) =
  Fmt.string fmt
    (match m with
    | PT.MemoryFormat.Unknown -> "unknown"
    | ContiguousFormat -> "contiguous"
    | ChannelsLast -> "channels_last"
    | ChannelsLast3d -> "channels_last_3d"
    | PreserveFormat -> "preserve")

let pp_device fmt (d : PT.Device.t) =
  match d.PT.Device.index with
  | None -> Fmt.string fmt d.PT.Device.type_
  | Some i -> Fmt.pf fmt "%s:%d" d.PT.Device.type_ i

let pp_sym_int_arg fmt (s : PT.SymIntArgument.t) =
  match s with
  | PT.SymIntArgument.Name n -> Fmt.string fmt n
  | Int i -> Fmt.int fmt i

let pp_sym_bool_arg fmt (s : PT.SymBoolArgument.t) =
  match s with
  | PT.SymBoolArgument.Name n -> Fmt.string fmt n
  | Bool b -> Fmt.bool fmt b

let pp_sym_float_arg fmt (s : PT.SymFloatArgument.t) =
  match s with
  | PT.SymFloatArgument.Name n -> Fmt.string fmt n
  | Float f -> pp_float fmt f

let pp_optional_tensor fmt (o : PT.OptionalTensorArgument.t) =
  match o with
  | PT.OptionalTensorArgument.Tensor t -> Fmt.string fmt (tname t)
  | None _ -> Fmt.string fmt "none"

let pp_sym_int fmt (s : PT.SymInt.t) =
  match s with
  | PT.SymInt.Expr e -> Fmt.string fmt e.PT.SymExpr.expr_str
  | Int i -> Fmt.int fmt i

(* Total, and reached through [Me_build.bounded], so a deeply nested argument
   stops the printer at the ceiling rather than being built whole and cut. A
   nested graph renders as its NAME: its body is a source view of its own and
   inlining it here would put a whole subgraph in one attribute. *)
let rec pp_arg fmt (a : A.t) =
  match a with
  | A.None _ -> Fmt.string fmt "none"
  | Tensor t -> Fmt.string fmt (tname t)
  | Tensors ts -> pp_seq Fmt.string fmt (List.map tname ts)
  | Nested_tensors tss ->
      pp_seq (pp_seq Fmt.string) fmt (List.map (List.map tname) tss)
  | Optional_tensor o -> pp_optional_tensor fmt o
  | Optional_tensors os -> pp_seq pp_optional_tensor fmt os
  | Int i -> Fmt.int fmt i
  | Ints is -> pp_seq Fmt.int fmt is
  | Int_lists iss -> pp_seq (pp_seq Fmt.int) fmt iss
  | Float f -> pp_float fmt f
  | Floats fs -> pp_seq pp_float fmt fs
  | Float_lists fss -> pp_seq (pp_seq pp_float) fmt fss
  | String s -> pp_quoted fmt s
  | Strings ss -> pp_seq pp_quoted fmt ss
  | Bool b -> Fmt.bool fmt b
  | Bools bs -> pp_seq Fmt.bool fmt bs
  | Sym_int s -> pp_sym_int_arg fmt s
  | Sym_ints ss -> pp_seq pp_sym_int_arg fmt ss
  | Sym_bool s -> pp_sym_bool_arg fmt s
  | Sym_bools ss -> pp_seq pp_sym_bool_arg fmt ss
  | Sym_float s -> pp_sym_float_arg fmt s
  | Sym_floats ss -> pp_seq pp_sym_float_arg fmt ss
  | Scalar_type st -> Fmt.string fmt (Pt2_dtype.scalar_type_name st)
  | Memory_format m -> pp_memory_format fmt m
  | Layout l -> pp_layout fmt l
  | Device d -> pp_device fmt d
  | Graph g -> Fmt.pf fmt "graph %s" g.PT.GraphArgument.name
  | Custom_obj c ->
      Fmt.pf fmt "%s : %s" c.PT.CustomObjArgument.name
        c.PT.CustomObjArgument.class_fqn
  | Operator o -> Fmt.pf fmt "operator %s" o
  | Complex c ->
      Fmt.pf fmt "%g+%gi" c.PT.ComplexValue.real c.PT.ComplexValue.imag
  | String_to_argument m ->
      Fmt.pf fmt "{%a}"
        (Fmt.list ~sep:comma (fun fmt (k, v) -> Fmt.pf fmt "%s=%a" k pp_arg v))
        (SM.bindings m)

(* --- the module hierarchy ------------------------------------------------ *)

(* [nn_module_stack] is [";"]-separated levels, each [path,name,class], with the
   OUTERMOST level naming the whole module and carrying an empty [name] — so it
   contributes no level, exactly as [Me_build] drops the root group's own
   component.

   Each [name] is the FULL dotted path ([layer1.1.conv1]), so a level is taken
   RELATIVE to its parent: [layer1 / 1 / conv1] rather than
   [layer1 / layer1.1 / layer1.1.conv1], which repeats the whole prefix at every
   depth for no added information. A level that does not extend its parent is
   kept whole, since dropping a prefix that is not there would be guessing. *)
let module_levels stack =
  let name entry =
    match String.split_on_char ',' entry with _ :: n :: _ -> n | _ -> ""
  in
  let relative parent n =
    let p = parent ^ "." in
    let lp = String.length p in
    if String.length n > lp && String.sub n 0 lp = p then
      String.sub n lp (String.length n - lp)
    else n
  in
  let rec go parent = function
    | [] -> []
    | entry :: rest -> (
        match name entry with
        | "" -> go parent rest
        | n -> relative parent n :: go n rest)
  in
  go "" (String.split_on_char ';' stack)

let namespace ~limits (n : PT.Node.t) =
  match SM.find_opt "nn_module_stack" n.PT.Node.metadata with
  | None -> Err.return ""
  | Some stack -> (
      match module_levels stack with
      | [] -> Err.return ""
      | levels -> Me_ids.pt2_namespace ~limits levels)

(* --- the projection ------------------------------------------------------ *)

(* The exported graph is root-scoped: the lowerer writes [graph_path = []] and
   the sidecar's origins render against that path, so these ids are exactly the
   ones the import comparison pairs. *)
let path = Pt2_native_graph.Graph_path.root

let node_ids (g : PT.Graph.t) =
  List.mapi (fun i _ -> Me_ids.pt2_node path i) g.PT.Graph.nodes

(* Which boundary an input SSA name belongs to, from the SIGNATURE. Guessing it
   from the exporter's [p_]/[b_]/[c_] name prefixes would be inventing a rule
   the schema already states. *)
let input_kinds (s : PT.GraphSignature.t) : (string, [ `In | `Const ]) Hashtbl.t
    =
  let table = Hashtbl.create 128 in
  List.iter
    (fun (spec : PT.InputSpec.t) ->
      match spec with
      | PT.InputSpec.User_input u ->
          List.iter
            (fun n -> Hashtbl.replace table n `In)
            (tensor_names u.PT.UserInputSpec.arg)
      | Parameter p ->
          Hashtbl.replace table (tname p.PT.InputToParameterSpec.arg) `Const
      | Buffer b ->
          Hashtbl.replace table (tname b.PT.InputToBufferSpec.arg) `Const
      | Tensor_constant c ->
          Hashtbl.replace table
            (tname c.PT.InputToTensorConstantSpec.arg)
            `Const
      | Custom_obj c ->
          Hashtbl.replace table
            c.PT.InputToCustomObjSpec.arg.PT.CustomObjArgument.name `Const
      (* A token is an effect ordering edge the caller supplies, and a constant
         input is a value rather than a tensor; neither is a weight. *)
      | Token t ->
          Hashtbl.replace table t.PT.InputTokenSpec.arg.PT.TokenArgument.name
            `In
      | Constant_input c ->
          Hashtbl.replace table c.PT.InputToConstantInputSpec.name `Const)
    s.PT.GraphSignature.input_specs;
  table

(* [Me_ids.pt2_boundary] also names the output side. An input is never that, so
   the widening is written here rather than as a dead [`Out] arm wherever an
   input kind is displayed. *)
let boundary_kind : [ `In | `Const ] -> [ `In | `Const | `Out ] = function
  | `In -> `In
  | `Const -> `Const

let shape_of (g : PT.Graph.t) name =
  match SM.find_opt name g.PT.Graph.tensor_values with
  | None -> None
  | Some meta ->
      Some
        ( Core.Pretty.to_string (pp_seq pp_sym_int) meta.PT.TensorMeta.sizes,
          Pt2_dtype.scalar_type_name meta.PT.TensorMeta.dtype )

let kv key value = ME.KeyValue.create ~key ~value

let attr key value =
  ME.NodeAttribute.create ~key ~value:(ME.NodeAttributeValue.Str value)

let output_metadata g slot name =
  let attrs =
    kv "ssa" name
    ::
    (match shape_of g name with
    | None -> []
    | Some (shape, dtype) -> [ kv "shape" shape; kv "dtype" dtype ])
  in
  ME.MetadataItem.create ~id:(string_of_int slot) ~attrs

let graph ~limits (gm : PT.GraphModule.t) =
  let open Err.Syntax in
  let g = gm.PT.GraphModule.graph in
  let nodes = g.PT.Graph.nodes in
  let inputs = List.concat_map tensor_names g.PT.Graph.inputs in
  let outputs = List.concat_map tensor_names g.PT.Graph.outputs in
  (* The ceiling comes first, before the walks that are linear in it. *)
  let node_count =
    List.length nodes + List.length inputs + List.length outputs
  in
  let* () =
    over_limit Me_limits.Field.Nodes node_count
      ~ceiling:limits.Me_limits.Limits.max_nodes_per_graph
  in
  let kinds = input_kinds gm.PT.GraphModule.signature in
  (* Where each SSA value comes from. A graph input resolves to its pinned
     boundary node, so an edge from a parameter is an ordinary edge rather than
     a case every consumer has to know about. *)
  let producer = Hashtbl.create 256 in
  let* () =
    Err.List.iter
      (fun name ->
        let kind = Option.value (Hashtbl.find_opt kinds name) ~default:`In in
        let+ id = Me_ids.pt2_boundary ~limits (boundary_kind kind) name in
        Hashtbl.replace producer name (id, 0))
      inputs
  in
  List.iteri
    (fun index (n : PT.Node.t) ->
      let id = Me_ids.pt2_node path index in
      List.iteri
        (fun slot name -> Hashtbl.replace producer name (id, slot))
        (List.concat_map tensor_names n.PT.Node.outputs))
    nodes;
  let resolve name =
    match Hashtbl.find_opt producer name with
    | Some p -> Err.return p
    | None -> Err.fail (`Unknown_producer name)
  in
  let* input_nodes =
    Err.List.map
      (fun name ->
        let kind = Option.value (Hashtbl.find_opt kinds name) ~default:`In in
        let+ id = Me_ids.pt2_boundary ~limits (boundary_kind kind) name in
        ME.GraphNode.create ~id
          ~label:(match kind with `In -> "input" | `Const -> "constant")
          ~namespace:"" ~incomingEdges:[]
          ~outputsMetadata:[ output_metadata g 0 name ]
          ())
      inputs
  in
  let* op_nodes =
    Err.List.map
      (fun (index, (n : PT.Node.t)) ->
        let* ns = namespace ~limits n in
        let+ incoming =
          Err.List.map
            (fun (arg_name, ssa) ->
              let+ src, slot = resolve ssa in
              ME.IncomingEdge.create ~sourceNodeId:src
                ~sourceNodeOutputId:(string_of_int slot)
                ~targetNodeInputId:arg_name ())
            (List.concat_map
               (fun (a : PT.NamedArgument.t) ->
                 List.map
                   (fun ssa -> (a.PT.NamedArgument.name, ssa))
                   (tensor_names a.PT.NamedArgument.arg))
               n.PT.Node.inputs)
        in
        (* One attribute per named argument, and a SEPARATE [_truncated] flag
           where the cap stopped the printer — appending an ellipsis would put
           the value past the very bound it reports. *)
        let attrs =
          List.concat_map
            (fun (a : PT.NamedArgument.t) ->
              let name = a.PT.NamedArgument.name in
              let text, capped =
                Me_build.bounded ~max:limits.Me_limits.Limits.max_attr_chars
                  pp_arg a.PT.NamedArgument.arg
              in
              attr name text
              :: (if capped then [ attr (name ^ "_truncated") "true" ] else []))
            n.PT.Node.inputs
        in
        ME.GraphNode.create
          ~id:(Me_ids.pt2_node path index)
          ~label:n.PT.Node.target ~namespace:ns ~incomingEdges:incoming
          ~outputsMetadata:
            (List.mapi (output_metadata g)
               (List.concat_map tensor_names n.PT.Node.outputs))
          ~attrs ())
      (List.mapi (fun i n -> (i, n)) nodes)
  in
  let* output_nodes =
    Err.List.map
      (fun name ->
        let* id = Me_ids.pt2_boundary ~limits `Out name in
        let+ src, slot = resolve name in
        ME.GraphNode.create ~id ~label:"output" ~namespace:""
          ~incomingEdges:
            [
              ME.IncomingEdge.create ~sourceNodeId:src
                ~sourceNodeOutputId:(string_of_int slot) ~targetNodeInputId:"0"
                ();
            ]
          ~outputsMetadata:[] ())
      outputs
  in
  let all = input_nodes @ op_nodes @ output_nodes in
  let edges =
    List.fold_left
      (fun acc (n : ME.GraphNode.t) ->
        acc
        + List.length (Option.value n.ME.GraphNode.incomingEdges ~default:[]))
      0 all
  in
  let+ () =
    over_limit Me_limits.Field.Edges edges
      ~ceiling:limits.Me_limits.Limits.max_edges_per_graph
  in
  ME.Graph.create ~id:(Me_ids.pt2_graph path) ~nodes:all ()
