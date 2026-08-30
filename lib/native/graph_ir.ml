(* See graph_ir.mli. Ids are plain ints under the hood (sealed [private int] by the
   interface); the builder owns their allocation so they are unique across the
   whole graph tree and deterministic per top-level build. [op] is parametrised
   over the embedded-graph type, so only the two small records [Node.t]/[Graph.t]
   need the recursive module group. *)

module Tensor_id = Tensor_id
module Node_id = Graph_common.Node_id
module Input = Graph_common.Input

type tensor_ref = Tensor_id.t

type op =
  (* Constructors kept in global alphabetical order (see graph_ir.mli). Each
     each op carries its own payload record (params + operand refs),
     defined in that op's module; the shared serialise / dataflow / pp logic is
     driven from [op_registry] below, not a per-constructor match. *)
  | Add of Pointwise.Add.t
  | Add_scalar of Pointwise.Add_scalar.t
  | Adaptive_avg_pool2d of Pool.AdaptiveAvgPool2d.t
  | Amax of Reduce.Amax.t
  | Avg_pool2d of Pool.AvgPool2d.t
  | Batch_norm of Norm.BatchNorm.t
  | Bmm of Matmul.Bmm.t
  | Clamp of Pointwise.Clamp.t
  | Clone of Pointwise.Clone.t
  | Concat of Concat.Concat.t
  | Conv2d of Conv.Conv2d.t
  | Conv2d_padding of Conv.Conv2d_padding.t
  | Convolution of Conv.Convolution.t
  | Div of Pointwise.Div.t
  | Div_scalar of Pointwise.Div_scalar.t
  | Discard of { x : tensor_ref }
  | Expand of Pointwise.Expand.t
  | Gelu of Pointwise.Gelu.t
  | Group_norm of Norm.GroupNorm.t
  | Hardsigmoid of Pointwise.Hardsigmoid.t
  | Hardswish of Pointwise.Hardswish.t
  | Hardtanh of Pointwise.Hardtanh.t
  | Layer_norm of Norm.LayerNorm.t
  | Linear of Linear.Linear.t
  | Max_pool2d of Pool.MaxPool2d.t
  | Max_pool2d_with_indices of Pool.MaxPool2dWithIndices.t
  | Mean of Reduce.Mean.t
  | Mul of Pointwise.Mul.t
  | Mul_scalar of Pointwise.Mul_scalar.t
  | Pad of Pad.Pad.t
  | Permute of Permute.Permute.t
  | Pow of Pointwise.Pow.t
  | Relu of Pointwise.Relu.t
  | Reshape of Reshape.Reshape.t
  | Rms_norm of Norm.RmsNorm.t
  | Sdpa of Attention.Sdpa.t
  | Select of Split.Select.t
  | Sigmoid of Pointwise.Sigmoid.t
  | Silu of Pointwise.Silu.t
  | Softmax of Reduce.Softmax.t
  | Slice of Split.Slice.t
  | Split_with_sizes of Split.Split_with_sizes.t
  | Sqrt of Pointwise.Sqrt.t
  | Stack of Concat.Stack.t
  | Sub of Pointwise.Sub.t
  | Sum of Reduce.Sum.t
  | Unbind of Split.Unbind.t
  | Upsample_bilinear2d of Resize.Bilinear2d.t
  | Upsample_nearest2d of Resize.Nearest2d.t
  | Vector_norm of Reduce.Vector_norm.t

(* A module ALIAS, so field access [n.Node.outputs] still resolves — the fields
   belong to the module this names. OCaml will not let a parameterised record be
   re-exported at a specialised instance ("they have different arities"), so the
   monomorphic form is [type node] below rather than [Node.t]. That costs
   nothing: outside this file nothing names [Node.t] as a type. *)
module Node = Graph_common.Node
module Group_id = Graph_common.Group_id
module Group = Graph_common.Group
module Graph = Graph_common.Graph

type node = op Node.t
type graph = op Graph.t

let nodes = Graph_common.nodes

module Printer = struct
  type t = {
    tensor : Format.formatter -> Tensor_id.t -> unit;
    node : Format.formatter -> Node_id.t -> unit;
  }
end

let input_kind = Graph_common.input_kind

(* Per-op interface. Each op module supplies the name, codec, dataflow accessors
   and printer for its OWN payload; [inject]/[project] (added by the wrappers in
   [op_registry], since they name the variant) splice that payload in and out of
   [op]. The common code below folds the registry instead of matching every
   constructor, so adding an op needs only its module plus one registry entry. *)
module type OP = sig
  type t

  val name : string
  val jsont : t Jsont.t
  val operands : t -> tensor_ref list
  val map_operands : (tensor_ref -> tensor_ref) -> t -> t
  val pp : tensor_ref Fmt.t -> Format.formatter -> t -> unit
  val inject : t -> op
  val project : op -> t option
end

(* In global alphabetical order, mirroring the [op] constructors. *)
let op_registry : (module OP) list =
  [
    (module struct
      include Pointwise.Add

      let inject t = Add t
      let project = function Add t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Add_scalar

      let inject t = Add_scalar t
      let project = function Add_scalar t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pool.AdaptiveAvgPool2d

      let inject t = Adaptive_avg_pool2d t
      let project = function Adaptive_avg_pool2d t -> Some t | _ -> None
    end : OP);
    (module struct
      include Reduce.Amax

      let inject t = Amax t
      let project = function Amax t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pool.AvgPool2d

      let inject t = Avg_pool2d t
      let project = function Avg_pool2d t -> Some t | _ -> None
    end : OP);
    (module struct
      include Norm.BatchNorm

      let inject t = Batch_norm t
      let project = function Batch_norm t -> Some t | _ -> None
    end : OP);
    (module struct
      include Matmul.Bmm

      let inject t = Bmm t
      let project = function Bmm t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Clamp

      let inject t = Clamp t
      let project = function Clamp t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Clone

      let inject t = Clone t
      let project = function Clone t -> Some t | _ -> None
    end : OP);
    (module struct
      include Concat.Concat

      let inject t = Concat t
      let project = function Concat t -> Some t | _ -> None
    end : OP);
    (module struct
      include Conv.Conv2d

      let inject t = Conv2d t
      let project = function Conv2d t -> Some t | _ -> None
    end : OP);
    (module struct
      include Conv.Conv2d_padding

      let inject t = Conv2d_padding t
      let project = function Conv2d_padding t -> Some t | _ -> None
    end : OP);
    (module struct
      include Conv.Convolution

      let inject t = Convolution t
      let project = function Convolution t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Div

      let inject t = Div t
      let project = function Div t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Div_scalar

      let inject t = Div_scalar t
      let project = function Div_scalar t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Expand

      let inject t = Expand t
      let project = function Expand t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Gelu

      let inject t = Gelu t
      let project = function Gelu t -> Some t | _ -> None
    end : OP);
    (module struct
      include Norm.GroupNorm

      let inject t = Group_norm t
      let project = function Group_norm t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Hardsigmoid

      let inject t = Hardsigmoid t
      let project = function Hardsigmoid t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Hardswish

      let inject t = Hardswish t
      let project = function Hardswish t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Hardtanh

      let inject t = Hardtanh t
      let project = function Hardtanh t -> Some t | _ -> None
    end : OP);
    (module struct
      include Norm.LayerNorm

      let inject t = Layer_norm t
      let project = function Layer_norm t -> Some t | _ -> None
    end : OP);
    (module struct
      include Linear.Linear

      let inject t = Linear t
      let project = function Linear t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pool.MaxPool2d

      let inject t = Max_pool2d t
      let project = function Max_pool2d t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pool.MaxPool2dWithIndices

      let inject t = Max_pool2d_with_indices t
      let project = function Max_pool2d_with_indices t -> Some t | _ -> None
    end : OP);
    (module struct
      include Reduce.Mean

      let inject t = Mean t
      let project = function Mean t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Mul

      let inject t = Mul t
      let project = function Mul t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Mul_scalar

      let inject t = Mul_scalar t
      let project = function Mul_scalar t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pad.Pad

      let inject t = Pad t
      let project = function Pad t -> Some t | _ -> None
    end : OP);
    (module struct
      include Permute.Permute

      let inject t = Permute t
      let project = function Permute t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Pow

      let inject t = Pow t
      let project = function Pow t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Relu

      let inject t = Relu t
      let project = function Relu t -> Some t | _ -> None
    end : OP);
    (module struct
      include Reshape.Reshape

      let inject t = Reshape t
      let project = function Reshape t -> Some t | _ -> None
    end : OP);
    (module struct
      include Norm.RmsNorm

      let inject t = Rms_norm t
      let project = function Rms_norm t -> Some t | _ -> None
    end : OP);
    (module struct
      include Attention.Sdpa

      let inject t = Sdpa t
      let project = function Sdpa t -> Some t | _ -> None
    end : OP);
    (module struct
      include Split.Select

      let inject t = Select t
      let project = function Select t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Sigmoid

      let inject t = Sigmoid t
      let project = function Sigmoid t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Silu

      let inject t = Silu t
      let project = function Silu t -> Some t | _ -> None
    end : OP);
    (module struct
      include Reduce.Softmax

      let inject t = Softmax t
      let project = function Softmax t -> Some t | _ -> None
    end : OP);
    (module struct
      include Split.Slice

      let inject t = Slice t
      let project = function Slice t -> Some t | _ -> None
    end : OP);
    (module struct
      include Split.Split_with_sizes

      let inject t = Split_with_sizes t
      let project = function Split_with_sizes t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Sqrt

      let inject t = Sqrt t
      let project = function Sqrt t -> Some t | _ -> None
    end : OP);
    (module struct
      include Concat.Stack

      let inject t = Stack t
      let project = function Stack t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Sub

      let inject t = Sub t
      let project = function Sub t -> Some t | _ -> None
    end : OP);
    (module struct
      include Reduce.Sum

      let inject t = Sum t
      let project = function Sum t -> Some t | _ -> None
    end : OP);
    (module struct
      include Split.Unbind

      let inject t = Unbind t
      let project = function Unbind t -> Some t | _ -> None
    end : OP);
    (module struct
      include Resize.Bilinear2d

      let inject t = Upsample_bilinear2d t
      let project = function Upsample_bilinear2d t -> Some t | _ -> None
    end : OP);
    (module struct
      include Resize.Nearest2d

      let inject t = Upsample_nearest2d t
      let project = function Upsample_nearest2d t -> Some t | _ -> None
    end : OP);
    (module struct
      include Reduce.Vector_norm

      let inject t = Vector_norm t
      let project = function Vector_norm t -> Some t | _ -> None
    end : OP);
  ]

(* Apply [f] to whichever registry module owns the current op. *)
let registry_pick (f : (module OP) -> 'a option) : 'a =
  Option.get (List.find_map f op_registry)

let operands : op -> tensor_ref list = function
  | Discard { x } -> [ x ]
  | op ->
      registry_pick (fun (module M : OP) ->
          Option.map M.operands (M.project op))

let map_operands (f : tensor_ref -> tensor_ref) : op -> op = function
  | Discard { x } -> Discard { x = f x }
  | op ->
      registry_pick (fun (module M : OP) ->
          Option.map (fun t -> M.inject (M.map_operands f t)) (M.project op))

(* [M.name] is the same string [op_jsont] uses as its case tag, deliberately:
   an exported graph that labelled its nodes differently from the way the IR
   serialises them would give two names to one op, and the first question asked
   of an exported node is which IR op it came from. [Discard] is spelled out
   because it owns no registry module — it is handled inline wherever the
   registry is folded (see the [op] declaration) — and it takes its JSON case
   tag for the same one-vocabulary reason. *)
let op_name : op -> string = function
  | Discard _ -> "Discard"
  | op ->
      registry_pick (fun (module M : OP) ->
          Option.map (fun (_ : M.t) -> M.name) (M.project op))

(* Producer/consumer/id lookup for the pretty-printer, precomputed once per
   dump instead of rescanning [g.Graph.nodes] for every operand/output/group
   item printed — that rescan is O(nodes) per lookup, so an unindexed dump (or
   repeated single-node dumps, e.g. one per evaluated node in `native_graph
   eval --verbose`) is quadratic in graph size. [source] pins the index to the
   graph it was built from ([==], since ids are only unique within one graph
   and restart from zero in another), so a caller that reuses a stale index
   against a different graph fails loudly instead of printing plausible but
   wrong producer/consumer annotations. *)
module Index = struct
  type t = {
    source : graph;
    producers : Node_id.t Tensor_id.Map.t;
    consumers : Node_id.t list Tensor_id.Map.t;
    by_id : node Node_id.Map.t;
  }

  let add_consumer node_id id consumers =
    Tensor_id.Map.update id
      (function None -> Some [ node_id ] | Some ids -> Some (node_id :: ids))
      consumers

  let make (g : graph) : t =
    let producers, consumers, by_id =
      List.fold_left
        (fun (producers, consumers, by_id) (node : node) ->
          let producers =
            List.fold_left
              (fun m id -> Tensor_id.Map.add id node.Node.id m)
              producers node.Node.outputs
          in
          let consumers =
            List.fold_left
              (fun m id -> add_consumer node.Node.id id m)
              consumers (operands node.Node.op)
          in
          let by_id = Node_id.Map.add node.Node.id node by_id in
          (producers, consumers, by_id))
        (Tensor_id.Map.empty, Tensor_id.Map.empty, Node_id.Map.empty)
        g.Graph.nodes
    in
    {
      source = g;
      producers;
      consumers = Tensor_id.Map.map (List.sort_uniq Node_id.compare) consumers;
      by_id;
    }

  let producer_of t id = Tensor_id.Map.find_opt id t.producers

  let consumers_of t id =
    Option.value (Tensor_id.Map.find_opt id t.consumers) ~default:[]

  let node_by_id t id = Node_id.Map.find_opt id t.by_id

  let assert_matches t (g : graph) =
    if t.source != g then
      invalid_arg
        "Graph_ir.Index: index was built from a different graph (ids are only \
         unique within one graph)"
end

(* ---- deterministic pretty-printer -------------------------------------- *)

let pp_list pp = Fmt.brackets (Fmt.list ~sep:Fmt.comma pp)
let pp_packed_fmt fmt (Payload.Fmt f) = Fmt.pf fmt "%a" Payload.pp_fmt f

let pp_tensor_sig fmt (sg : Tensor_sig.t) =
  match sg.quant with
  | None ->
      Fmt.pf fmt "@[<h>%a %a@]" pp_packed_fmt sg.fmt Vec6.pp_shape sg.shape
  | Some q ->
      Fmt.pf fmt "@[<h>%a[%a] %a@]" pp_packed_fmt sg.fmt Quant.pp q
        Vec6.pp_shape sg.shape

let tensor_sig_opt (g : graph) id =
  try Some (Tensor_id.Map.find id g.Graph.tensors) with Not_found -> None

(* [p.tensor]/[p.node] already have a printer's shape, so they feed %a
   directly — no unit-lambda needed to defer them. *)
let pp_tensor_annotation printer fmt id =
  Option.iter
    (fun (p : Printer.t) -> Fmt.pf fmt " @[<h>{%a}@]" p.tensor id)
    printer

let pp_node_annotation printer fmt id =
  Option.iter
    (fun (p : Printer.t) -> Fmt.pf fmt " @[<h>{%a}@]" p.node id)
    printer

let pp_producer_annotation (index : Index.t) fmt id =
  Option.iter
    (fun node_id -> Fmt.pf fmt " <-%a" Node_id.pp node_id)
    (Index.producer_of index id)

let pp_consumers_annotation (index : Index.t) fmt id =
  match Index.consumers_of index id with
  | [] -> ()
  | ids -> Fmt.pf fmt " ->%a" (pp_list Node_id.pp) ids

let pp_tensor_ref ?printer ~index (g : graph) fmt id =
  match tensor_sig_opt g id with
  | None -> Tensor_id.pp fmt id
  | Some _ ->
      Fmt.pf fmt "%a%a%a" Tensor_id.pp id
        (pp_tensor_annotation printer)
        id
        (pp_producer_annotation index)
        id

let pp_tensor_def ?printer ~index (g : graph) fmt id =
  match tensor_sig_opt g id with
  | None -> Tensor_id.pp fmt id
  | Some sg ->
      Fmt.pf fmt "@[<h>%a %a%a%a@]" Tensor_id.pp id pp_tensor_sig sg
        (pp_tensor_annotation printer)
        id
        (pp_consumers_annotation index)
        id

let pp_tensor_defs ?printer ~index g = pp_list (pp_tensor_def ?printer ~index g)

(* Graph-level outputs aren't printed under the node that made them, so unlike
   [pp_tensor_def] they also carry a producer backref (would be a redundant
   self-reference on a node's own output list). *)
let pp_graph_output_def ?printer ~index (g : graph) fmt id =
  Fmt.pf fmt "%a%a"
    (pp_tensor_def ?printer ~index g)
    id
    (pp_producer_annotation index)
    id

let pp_graph_output_defs ?printer ~index g =
  pp_list (pp_graph_output_def ?printer ~index g)

let pp_op_header ?printer ~index g fmt (node : node) =
  Fmt.pf fmt "@[<h>%a%a: %a =@]" Node_id.pp node.Node.id
    (pp_node_annotation printer)
    node.Node.id
    (pp_tensor_defs ?printer ~index g)
    node.outputs

(* Op printing needs only a way to render an operand reference, so the primitive
   takes one. That lets a caller without a graph — a recipe describing nodes it
   has not spliced yet — print an op by falling back to bare ids. *)
let pp_op_with ~pp_ref fmt : op -> unit = function
  | Discard { x } -> Fmt.pf fmt "@[<hv 2>discard@ x=%a@]" pp_ref x
  | op ->
      registry_pick (fun (module M : OP) ->
          Option.map (fun t -> M.pp pp_ref fmt t) (M.project op))

let pp_op ?printer ~index (g : graph) fmt op =
  pp_op_with ~pp_ref:(pp_tensor_ref ?printer ~index g) fmt op

let pp_graph_header fmt (_g : graph) = Fmt.pf fmt "graph"

let pp_graph_inputs ?printer ~index fmt (g : graph) =
  let pp_input fmt id =
    match input_kind g id with
    | Input.Constant ->
        Fmt.pf fmt "%a constant" (pp_tensor_def ?printer ~index g) id
    | Input.Input -> pp_tensor_def ?printer ~index g fmt id
  in
  Fmt.pf fmt "@[<hv 2>inputs:@ %a@]" (pp_list pp_input) g.Graph.inputs

let pp_graph_outputs ?printer ~index fmt (g : graph) =
  Fmt.pf fmt "@[<hv 2>outputs:@ %a@]"
    (pp_graph_output_defs ?printer ~index g)
    g.Graph.outputs

let pp_node_header ?printer ~index g fmt node =
  Fmt.pf fmt "@[<hv 2>%a@ %a@]"
    (pp_op_header ?printer ~index g)
    node (pp_op ?printer ~index g) node.Node.op

let rec pp_graph ?printer ~index fmt (g : graph) =
  Fmt.pf fmt "@[<v>%a@,%a@,%a@,%a@]" pp_graph_header g
    (pp_graph_inputs ?printer ~index)
    g
    (pp_root ?printer ~index g)
    g.Graph.root
    (pp_graph_outputs ?printer ~index)
    g

and pp_root ?printer ~index g fmt (root : Group.t) =
  let pp_item fmt = function
    | Group.Group child -> pp_group ?printer ~index g fmt child
    | Group.Node id -> (
        match Index.node_by_id index id with
        | Some node -> pp_node_header ?printer ~index g fmt node
        | None -> Fmt.pf fmt "%a <missing>" Node_id.pp id)
  in
  Fmt.pf fmt "@[<v 2>nodes:@,%a@]" (Fmt.list ~sep:Fmt.cut pp_item) root.items

and pp_group ?printer ~index g fmt (group : Group.t) =
  let pp_item fmt = function
    | Group.Group child -> pp_group ?printer ~index g fmt child
    | Group.Node id -> (
        match Index.node_by_id index id with
        | Some node -> pp_node_header ?printer ~index g fmt node
        | None -> Fmt.pf fmt "%a <missing>" Node_id.pp id)
  in
  let label = Option.value group.label ~default:"" in
  Fmt.pf fmt "@[<v 2>group %a%s:@,%a@]" Group_id.pp group.id
    (if String.equal label "" then "" else " " ^ label)
    (Fmt.list ~sep:Fmt.cut pp_item)
    group.items

(* [pp], [pp_with] and [pp_op] dump a whole graph or a single detached op in
   one call, so building the index on entry costs nothing extra; giving them
   an optional [?index] too would leave a leading optional argument in their
   type whenever they're used point-free (e.g. as a "%a" printer with only
   [~printer] applied), which breaks that usage. [pp_node] is the one caller
   that prints repeatedly against the same graph — one call per evaluated
   node in `native_graph eval --verbose` — so it alone takes a precomputed
   index to avoid rebuilding it on every call. *)
let pp_node ?printer ?index g fmt node =
  let index =
    match index with
    | Some index ->
        Index.assert_matches index g;
        index
    | None -> Index.make g
  in
  pp_node_header ?printer ~index g fmt node

let pp_op ?printer (g : graph) fmt op =
  pp_op ?printer ~index:(Index.make g) g fmt op

let pp_with ~printer fmt g = pp_graph ~printer ~index:(Index.make g) fmt g
let pp fmt g = pp_graph ~index:(Index.make g) fmt g

(* ---- JSON codecs ---------------------------------------------------------- *)
(* Groups and graph records are serialised separately from ordinary operations. *)

let tensor_ref_jsont : tensor_ref Jsont.t = Tensor_id.jsont

(* The registry's decode side: one [(case, decoder)] per op. *)
let op_decode_cases : (string * (Jsont.json -> op)) list =
  List.map
    (fun (module M : OP) ->
      (M.name, fun v -> M.inject (Json_util.dec M.jsont v)))
    op_registry

let dec_op (json : Jsont.json) : op =
  let discard_case =
    ( "Discard",
      fun v ->
        let ms = Json_util.req_obj v "Discard" in
        let get k c = Json_util.req_field ms k c "Discard" in
        Discard { x = get "x" tensor_ref_jsont } )
  in
  Json_util.union ~kind:"op" (discard_case :: op_decode_cases) json

let enc_op (op : op) : Jsont.json =
  match op with
  | Discard { x } ->
      Json_util.single ~case:"Discard"
        (Json_util.jobj [ ("x", Json_util.enc tensor_ref_jsont x) ])
  | op ->
      registry_pick (fun (module M : OP) ->
          Option.map
            (fun t -> Json_util.single ~case:M.name (Json_util.enc M.jsont t))
            (M.project op))

let enc_node (node : node) : Jsont.json =
  Json_util.jobj
    [
      ("id", Json_util.jint (Node_id.to_int node.Node.id));
      ("op", enc_op node.Node.op);
      ( "outputs",
        Json_util.jarr
          (List.map (Json_util.enc tensor_ref_jsont) node.Node.outputs) );
    ]

let dec_node (json : Jsont.json) : node =
  let ms = Json_util.req_obj json "node" in
  let get k c = Json_util.req_field ms k c "node" in
  let op_codec = Jsont.map ~kind:"op" ~dec:dec_op ~enc:enc_op Jsont.json in
  Node.
    {
      id = Node_id.of_int (get "id" Jsont.int);
      op = get "op" op_codec;
      outputs = get "outputs" (Jsont.list tensor_ref_jsont);
    }

let rec dec_group (json : Jsont.json) : Group.t =
  let ms = Json_util.req_obj json "group" in
  let get k c = Json_util.req_field ms k c "group" in
  let dec_item json =
    Json_util.union ~kind:"group item"
      [
        ("Group", fun v -> Group.Group (dec_group v));
        ( "Node",
          fun v -> Group.Node (Json_util.dec Jsont.int v |> Node_id.of_int) );
      ]
      json
  in
  Group.
    {
      id = Group_id.of_int (get "id" Jsont.int);
      label = Json_util.opt_field ms "label" Jsont.string;
      items =
        get "items"
          (Jsont.list
             (Jsont.map ~kind:"group item" ~dec:dec_item
                ~enc:(fun _ -> assert false)
                Jsont.json));
    }

let dec_graph (json : Jsont.json) : graph =
  let ms = Json_util.req_obj json "graph" in
  let get k c = Json_util.req_field ms k c "graph" in
  let node_codec =
    Jsont.map ~kind:"node" ~dec:dec_node ~enc:enc_node Jsont.json
  in
  let tensors_list = get "tensors" (Jsont.list Tensor_sig.jsont) in
  let tensors =
    List.fold_left
      (fun m (sg : Tensor_sig.t) -> Tensor_id.Map.add sg.id sg m)
      Tensor_id.Map.empty tensors_list
  in
  let input_kinds =
    (* An absent field and an empty list agree here: the fold's seed IS the
       absent case, so the option only has to supply a list. *)
    Json_util.opt_field ms "input_constants" (Jsont.list Jsont.json)
    |> Option.value ~default:[]
    |> List.fold_left
         (fun kinds json ->
           let cms = Json_util.req_obj json "input constant" in
           let id =
             Json_util.req_field cms "id" tensor_ref_jsont "input constant"
           in
           Tensor_id.Map.add id Input.Constant kinds)
         Tensor_id.Map.empty
  in
  Graph.
    {
      inputs = get "inputs" (Jsont.list tensor_ref_jsont);
      input_kinds;
      nodes = get "nodes" (Jsont.list node_codec);
      root =
        get "root"
          (Jsont.map ~kind:"group" ~dec:dec_group
             ~enc:(fun _ -> assert false)
             Jsont.json);
      outputs = get "outputs" (Jsont.list tensor_ref_jsont);
      tensors;
    }

let rec enc_group (group : Group.t) : Jsont.json =
  let enc_item = function
    | Group.Group child -> Json_util.single ~case:"Group" (enc_group child)
    | Group.Node id ->
        Json_util.single ~case:"Node" (Json_util.jint (Node_id.to_int id))
  in
  Json_util.jobj
    ([
       ("id", Json_util.jint (Group_id.to_int group.id));
       ("items", Json_util.jarr (List.map enc_item group.items));
     ]
    @ Option.fold ~none:[]
        ~some:(fun label -> [ ("label", Json_util.jstr label) ])
        group.label)

let enc_graph (g : graph) : Jsont.json =
  let tensors_json =
    Tensor_id.Map.bindings g.Graph.tensors
    |> List.map (fun (_tid, sg) -> Json_util.enc Tensor_sig.jsont sg)
  in
  let constants =
    List.filter_map
      (fun id ->
        match input_kind g id with
        | Input.Constant ->
            Some (Json_util.jobj [ ("id", Json_util.enc tensor_ref_jsont id) ])
        | Input.Input -> None)
      g.Graph.inputs
  in
  let fields =
    [
      ( "inputs",
        Json_util.jarr
          (List.map (Json_util.enc tensor_ref_jsont) g.Graph.inputs) );
      ("nodes", Json_util.jarr (List.map enc_node g.Graph.nodes));
      ("root", enc_group g.Graph.root);
      ( "outputs",
        Json_util.jarr
          (List.map (Json_util.enc tensor_ref_jsont) g.Graph.outputs) );
      ("tensors", Json_util.jarr tensors_json);
    ]
  in
  Json_util.jobj
    (if constants = [] then fields
     else ("input_constants", Json_util.jarr constants) :: fields)

let op_jsont : op Jsont.t =
  Jsont.map ~kind:"op" ~dec:dec_op ~enc:enc_op Jsont.json

let graph_jsont : graph Jsont.t =
  Jsont.map ~kind:"graph" ~dec:dec_graph ~enc:enc_graph Jsont.json
