(* See graph_ir.mli. Ids are plain ints under the hood (sealed [private int] by the
   interface); the builder owns their allocation so they are unique across the
   whole graph tree and deterministic per top-level build. [op] is parametrised
   over the embedded-graph type, so only the two small records [Node.t]/[Graph.t]
   need the recursive module group. *)

module Tensor_id = Tensor_id

module Node_id = struct
  type t = int

  let of_int x = x
  let to_int x = x
  let pp fmt x = Format.fprintf fmt "n%d" x
end

type tensor_ref = Tensor_id.t

type 'g gop =
  (* Constructors kept in global alphabetical order (see graph_ir.mli). Each
     non-[Subgraph] op carries its own payload record (params + operand refs),
     defined in that op's module; the shared serialise / dataflow / pp logic is
     driven from [op_registry] below, not a per-constructor match. [Subgraph] is
     the exception — it embeds the recursive graph type ['g] — so it stays inline
     and is handled directly wherever the registry is folded. *)
  | Add of Pointwise.Add.t
  | Avg_pool2d of Pool.AvgPool2d.t
  | Bmm of Matmul.Bmm.t
  | Conv2d of Conv.Conv2d.t
  | Conv2d_padding of Conv.Conv2d_padding.t
  | Convolution of Conv.Convolution.t
  | Linear of Linear.Linear.t
  | Max_pool2d of Pool.MaxPool2d.t
  | Mean of Reduce.Mean.t
  | Mul of Pointwise.Mul.t
  | Permute of Permute.Permute.t
  | Relu of Pointwise.Relu.t
  | Rms_norm of Norm.RmsNorm.t
  | Subgraph of { graph : 'g; args : tensor_ref list }

module rec Node : sig
  type t = { id : Node_id.t; op : Graph.t gop; outputs : Tensor_id.t list }
end = struct
  type t = { id : Node_id.t; op : Graph.t gop; outputs : Tensor_id.t list }
end

and Graph : sig
  type t = {
    name : string;
    nodes : Node.t list;
    tensors : Tensor_sig.t Tensor_id.Map.t;
    inputs : Tensor_id.t list;
    outputs : Tensor_id.t list;
  }
end = struct
  type t = {
    name : string;
    nodes : Node.t list;
    tensors : Tensor_sig.t Tensor_id.Map.t;
    inputs : Tensor_id.t list;
    outputs : Tensor_id.t list;
  }
end

type op = Graph.t gop
type node = Node.t
type graph = Graph.t

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

(* In global alphabetical order, mirroring the [gop] constructors. [Subgraph] is
   deliberately absent — it is handled directly by every fold below. *)
let op_registry : (module OP) list =
  [
    (module struct
      include Pointwise.Add

      let inject t = Add t
      let project = function Add t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pool.AvgPool2d

      let inject t = Avg_pool2d t
      let project = function Avg_pool2d t -> Some t | _ -> None
    end : OP);
    (module struct
      include Matmul.Bmm

      let inject t = Bmm t
      let project = function Bmm t -> Some t | _ -> None
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
      include Permute.Permute

      let inject t = Permute t
      let project = function Permute t -> Some t | _ -> None
    end : OP);
    (module struct
      include Pointwise.Relu

      let inject t = Relu t
      let project = function Relu t -> Some t | _ -> None
    end : OP);
    (module struct
      include Norm.RmsNorm

      let inject t = Rms_norm t
      let project = function Rms_norm t -> Some t | _ -> None
    end : OP);
  ]

(* Apply [f] to whichever registry module owns the current op (the one whose
   [project] succeeds). Total for every non-[Subgraph] op, which the callers
   match away first. *)
let registry_pick (f : (module OP) -> 'a option) : 'a =
  Option.get (List.find_map f op_registry)

let operands : op -> tensor_ref list = function
  | Subgraph { args; _ } -> args
  | op ->
      registry_pick (fun (module M : OP) ->
          Option.map M.operands (M.project op))

let map_operands (f : tensor_ref -> tensor_ref) : op -> op = function
  | Subgraph { graph; args } -> Subgraph { graph; args = List.map f args }
  | op ->
      registry_pick (fun (module M : OP) ->
          Option.map (fun t -> M.inject (M.map_operands f t)) (M.project op))

(* ---- deterministic pretty-printer -------------------------------------- *)

let pp_list pp = Fmt.brackets (Fmt.list ~sep:Fmt.comma pp)
let pp_packed_fmt fmt (Payload.Fmt f) = Fmt.pf fmt "%a" Payload.pp_fmt f

let pp_tensor_sig fmt (sg : Tensor_sig.t) =
  match sg.quant with
  | None ->
      Fmt.pf fmt "@[<h>%s:%a %a@]" sg.name pp_packed_fmt sg.fmt Vec6.pp_shape
        sg.shape
  | Some q ->
      Fmt.pf fmt "@[<h>%s:%a[%a] %a@]" sg.name pp_packed_fmt sg.fmt Quant.pp q
        Vec6.pp_shape sg.shape

let tensor_sig_opt (g : graph) id =
  try Some (Tensor_id.Map.find id g.Graph.tensors) with Not_found -> None

let pp_tensor_ref (g : graph) fmt id =
  match tensor_sig_opt g id with
  | None -> Tensor_id.pp fmt id
  | Some (sg : Tensor_sig.t) ->
      Fmt.pf fmt "@[<h>%a(%s)@]" Tensor_id.pp id sg.name

let pp_tensor_def (g : graph) fmt id =
  match tensor_sig_opt g id with
  | None -> Tensor_id.pp fmt id
  | Some sg -> Fmt.pf fmt "@[<h>%a %a@]" Tensor_id.pp id pp_tensor_sig sg

let pp_tensor_defs g = pp_list (pp_tensor_def g)
let pp_tensor_refs g = pp_list (pp_tensor_ref g)

let pp_op_header g fmt (node : node) =
  Fmt.pf fmt "@[<h>%a: %a =@]" Node_id.pp node.Node.id (pp_tensor_defs g)
    node.outputs

let pp_op (g : graph) fmt : op -> unit = function
  | Subgraph { graph; args } ->
      Fmt.pf fmt "@[<hv 2>subgraph@ %s@ args=%a@]" graph.Graph.name
        (pp_tensor_refs g) args
  | op ->
      let pp_ref = pp_tensor_ref g in
      registry_pick (fun (module M : OP) ->
          Option.map (fun t -> M.pp pp_ref fmt t) (M.project op))

let pp_graph_header fmt (g : graph) = Fmt.pf fmt "graph %s" g.Graph.name

let pp_graph_inputs fmt (g : graph) =
  Fmt.pf fmt "@[<hv 2>inputs:@ %a@]" (pp_tensor_defs g) g.Graph.inputs

let pp_graph_outputs fmt (g : graph) =
  Fmt.pf fmt "@[<hv 2>outputs:@ %a@]" (pp_tensor_defs g) g.Graph.outputs

let pp_node_header g fmt node =
  Fmt.pf fmt "@[<hv 2>%a@ %a@]" (pp_op_header g) node (pp_op g) node.Node.op

let rec pp_graph fmt (g : graph) =
  Fmt.pf fmt "@[<v>%a@,%a@,%a@,%a@]" pp_graph_header g pp_graph_inputs g
    (pp_nodes g) g.Graph.nodes pp_graph_outputs g

and pp_nodes g fmt = function
  | [] -> Fmt.pf fmt "nodes: []"
  | nodes ->
      Fmt.pf fmt "@[<v 2>nodes:@,%a@]" (Fmt.list ~sep:Fmt.cut (pp_node g)) nodes

and pp_node g fmt node =
  match node.Node.op with
  | Subgraph { graph; _ } ->
      Fmt.pf fmt "@[<v 2>%a@,%a@]" (pp_node_header g) node pp_graph graph
  | _ -> pp_node_header g fmt node

let pp = pp_graph

(* ---- JSON codecs ---------------------------------------------------------- *)
(* op → node → graph is a mutually recursive type, so the encode/decode
   functions are defined as a single [let rec] group and then wrapped in
   Jsont codecs. *)

let tensor_ref_jsont : tensor_ref Jsont.t = Tensor_id.jsont

(* The registry's decode side: one [(case, decoder)] per op, the decoder reading
   the payload object through that op's own [jsont] and injecting it. [Subgraph]
   is added separately by [dec_op] since its decoder recurses through [dec_graph]. *)
let op_decode_cases : (string * (Jsont.json -> op)) list =
  List.map
    (fun (module M : OP) ->
      (M.name, fun v -> M.inject (Json_util.dec M.jsont v)))
    op_registry

let rec dec_op (json : Jsont.json) : op =
  let subgraph_case =
    ( "Subgraph",
      fun v ->
        let ms = Json_util.req_obj v "Subgraph" in
        let get k c = Json_util.req_field ms k c "Subgraph" in
        let graph_codec =
          Jsont.map ~kind:"graph" ~dec:dec_graph ~enc:enc_graph Jsont.json
        in
        Subgraph
          {
            graph = get "graph" graph_codec;
            args = get "args" (Jsont.list tensor_ref_jsont);
          } )
  in
  Json_util.union ~kind:"op" (subgraph_case :: op_decode_cases) json

and dec_node (json : Jsont.json) : node =
  let ms = Json_util.req_obj json "node" in
  let get k c = Json_util.req_field ms k c "node" in
  let op_codec = Jsont.map ~kind:"op" ~dec:dec_op ~enc:enc_op Jsont.json in
  Node.
    {
      id = Node_id.of_int (get "id" Jsont.int);
      op = get "op" op_codec;
      outputs = get "outputs" (Jsont.list tensor_ref_jsont);
    }

and dec_graph (json : Jsont.json) : graph =
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
  Graph.
    {
      name = get "name" Jsont.string;
      inputs = get "inputs" (Jsont.list tensor_ref_jsont);
      nodes = get "nodes" (Jsont.list node_codec);
      outputs = get "outputs" (Jsont.list tensor_ref_jsont);
      tensors;
    }

and enc_op (op : op) : Jsont.json =
  match op with
  | Subgraph { graph; args } ->
      Json_util.single ~case:"Subgraph"
        (Json_util.jobj
           [
             ( "args",
               Json_util.jarr (List.map (Json_util.enc tensor_ref_jsont) args)
             );
             ("graph", enc_graph graph);
           ])
  | op ->
      registry_pick (fun (module M : OP) ->
          Option.map
            (fun t -> Json_util.single ~case:M.name (Json_util.enc M.jsont t))
            (M.project op))

and enc_node (node : node) : Jsont.json =
  Json_util.jobj
    [
      ("id", Json_util.jint (Node_id.to_int node.Node.id));
      ("op", enc_op node.Node.op);
      ( "outputs",
        Json_util.jarr
          (List.map (Json_util.enc tensor_ref_jsont) node.Node.outputs) );
    ]

and enc_graph (g : graph) : Jsont.json =
  let tensors_json =
    Tensor_id.Map.bindings g.Graph.tensors
    |> List.map (fun (_tid, sg) -> Json_util.enc Tensor_sig.jsont sg)
  in
  Json_util.jobj
    [
      ( "inputs",
        Json_util.jarr
          (List.map (Json_util.enc tensor_ref_jsont) g.Graph.inputs) );
      ("name", Json_util.jstr g.Graph.name);
      ("nodes", Json_util.jarr (List.map enc_node g.Graph.nodes));
      ( "outputs",
        Json_util.jarr
          (List.map (Json_util.enc tensor_ref_jsont) g.Graph.outputs) );
      ("tensors", Json_util.jarr tensors_json);
    ]

let op_jsont : op Jsont.t =
  Jsont.map ~kind:"op" ~dec:dec_op ~enc:enc_op Jsont.json

let graph_jsont : graph Jsont.t =
  Jsont.map ~kind:"graph" ~dec:dec_graph ~enc:enc_graph Jsont.json
