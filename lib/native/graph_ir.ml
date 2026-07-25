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
  let equal = Int.equal
  let compare = Int.compare
  let pp fmt x = Format.fprintf fmt "n%d" x

  module Map = Map.Make (struct
    type nonrec t = t

    let compare = compare
  end)
end

module Input = struct
  type kind = Input | Constant

  let pp_kind fmt = function
    | Input -> Format.pp_print_string fmt "input"
    | Constant -> Format.pp_print_string fmt "constant"
end

type tensor_ref = Tensor_id.t

type op =
  (* Constructors kept in global alphabetical order (see graph_ir.mli). Each
     each op carries its own payload record (params + operand refs),
     defined in that op's module; the shared serialise / dataflow / pp logic is
     driven from [op_registry] below, not a per-constructor match. *)
  | Add of Pointwise.Add.t
  | Avg_pool2d of Pool.AvgPool2d.t
  | Batch_norm of Norm.BatchNorm.t
  | Bmm of Matmul.Bmm.t
  | Conv2d of Conv.Conv2d.t
  | Conv2d_padding of Conv.Conv2d_padding.t
  | Convolution of Conv.Convolution.t
  | Discard of { x : tensor_ref }
  | Linear of Linear.Linear.t
  | Max_pool2d of Pool.MaxPool2d.t
  | Max_pool2d_with_indices of Pool.MaxPool2dWithIndices.t
  | Mean of Reduce.Mean.t
  | Mul of Pointwise.Mul.t
  | Permute of Permute.Permute.t
  | Relu of Pointwise.Relu.t
  | Reshape of Reshape.Reshape.t
  | Rms_norm of Norm.RmsNorm.t

module Node = struct
  type t = { id : Node_id.t; op : op; outputs : Tensor_id.t list }
end

module Group_id = struct
  type t = int

  let of_int x = x
  let to_int x = x
  let pp fmt x = Format.fprintf fmt "g%d" x
end

module Group = struct
  type item = Node of Node_id.t | Group of t
  and t = { id : Group_id.t; label : string option; items : item list }
end

module Graph = struct
  type t = {
    nodes : Node.t list;
    root : Group.t;
    tensors : Tensor_sig.t Tensor_id.Map.t;
    inputs : Tensor_id.t list;
    input_kinds : Input.kind Tensor_id.Map.t;
    outputs : Tensor_id.t list;
  }
end

type node = Node.t
type graph = Graph.t

let nodes g = g.Graph.nodes

module Printer = struct
  type t = {
    tensor : Format.formatter -> Tensor_id.t -> unit;
    node : Format.formatter -> Node_id.t -> unit;
  }
end

let input_kind g id =
  Option.value
    (Tensor_id.Map.find_opt id g.Graph.input_kinds)
    ~default:Input.Input

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
      include Reshape.Reshape

      let inject t = Reshape t
      let project = function Reshape t -> Some t | _ -> None
    end : OP);
    (module struct
      include Norm.RmsNorm

      let inject t = Rms_norm t
      let project = function Rms_norm t -> Some t | _ -> None
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

let pp_tensor_annotation printer fmt id =
  match printer with
  | None -> ()
  | Some (printer : Printer.t) ->
      Fmt.pf fmt " @[<h>{%a}@]" (fun fmt () -> printer.tensor fmt id) ()

let pp_node_annotation printer fmt id =
  match printer with
  | None -> ()
  | Some (printer : Printer.t) ->
      Fmt.pf fmt " @[<h>{%a}@]" (fun fmt () -> printer.node fmt id) ()

let pp_tensor_ref ?printer (g : graph) fmt id =
  match tensor_sig_opt g id with
  | None -> Tensor_id.pp fmt id
  | Some _ ->
      Fmt.pf fmt "%a%a" Tensor_id.pp id (pp_tensor_annotation printer) id

let pp_tensor_def ?printer (g : graph) fmt id =
  match tensor_sig_opt g id with
  | None -> Tensor_id.pp fmt id
  | Some sg ->
      Fmt.pf fmt "@[<h>%a %a%a@]" Tensor_id.pp id pp_tensor_sig sg
        (pp_tensor_annotation printer)
        id

let pp_tensor_defs ?printer g = pp_list (pp_tensor_def ?printer g)

let pp_op_header ?printer g fmt (node : node) =
  Fmt.pf fmt "@[<h>%a%a: %a =@]" Node_id.pp node.Node.id
    (pp_node_annotation printer)
    node.Node.id
    (pp_tensor_defs ?printer g)
    node.outputs

let pp_op ?printer (g : graph) fmt : op -> unit = function
  | Discard { x } ->
      Fmt.pf fmt "@[<hv 2>discard@ x=%a@]" (pp_tensor_ref ?printer g) x
  | op ->
      let pp_ref = pp_tensor_ref ?printer g in
      registry_pick (fun (module M : OP) ->
          Option.map (fun t -> M.pp pp_ref fmt t) (M.project op))

let pp_graph_header fmt (_g : graph) = Fmt.pf fmt "graph"

let pp_graph_inputs ?printer fmt (g : graph) =
  let pp_input fmt id =
    match input_kind g id with
    | Input.Input -> pp_tensor_def ?printer g fmt id
    | Input.Constant -> Fmt.pf fmt "%a constant" (pp_tensor_def ?printer g) id
  in
  Fmt.pf fmt "@[<hv 2>inputs:@ %a@]" (pp_list pp_input) g.Graph.inputs

let pp_graph_outputs ?printer fmt (g : graph) =
  Fmt.pf fmt "@[<hv 2>outputs:@ %a@]"
    (pp_tensor_defs ?printer g)
    g.Graph.outputs

let pp_node_header ?printer g fmt node =
  Fmt.pf fmt "@[<hv 2>%a@ %a@]" (pp_op_header ?printer g) node
    (pp_op ?printer g) node.Node.op

let pp_node = pp_node_header

let node_by_id (g : graph) id =
  List.find_opt
    (fun (node : node) -> Node_id.equal node.Node.id id)
    g.Graph.nodes

let rec pp_graph ?printer fmt (g : graph) =
  Fmt.pf fmt "@[<v>%a@,%a@,%a@,%a@]" pp_graph_header g
    (pp_graph_inputs ?printer) g (pp_root ?printer g) g.Graph.root
    (pp_graph_outputs ?printer)
    g

and pp_root ?printer g fmt (root : Group.t) =
  let pp_item fmt = function
    | Group.Node id -> (
        match node_by_id g id with
        | Some node -> pp_node_header ?printer g fmt node
        | None -> Fmt.pf fmt "%a <missing>" Node_id.pp id)
    | Group.Group child -> pp_group ?printer g fmt child
  in
  Fmt.pf fmt "@[<v 2>nodes:@,%a@]" (Fmt.list ~sep:Fmt.cut pp_item) root.items

and pp_group ?printer g fmt (group : Group.t) =
  let pp_item fmt = function
    | Group.Node id -> (
        match node_by_id g id with
        | Some node -> pp_node_header ?printer g fmt node
        | None -> Fmt.pf fmt "%a <missing>" Node_id.pp id)
    | Group.Group child -> pp_group ?printer g fmt child
  in
  let label = Option.value group.label ~default:"" in
  Fmt.pf fmt "@[<v 2>group %a%s:@,%a@]" Group_id.pp group.id
    (if String.equal label "" then "" else " " ^ label)
    (Fmt.list ~sep:Fmt.cut pp_item)
    group.items

let pp_with ~printer = pp_graph ~printer
let pp fmt graph = pp_graph fmt graph

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
        ( "Node",
          fun v -> Group.Node (Json_util.dec Jsont.int v |> Node_id.of_int) );
        ("Group", fun v -> Group.Group (dec_group v));
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
    match Json_util.opt_field ms "input_constants" (Jsont.list Jsont.json) with
    | None -> Tensor_id.Map.empty
    | Some constants ->
        List.fold_left
          (fun kinds json ->
            let cms = Json_util.req_obj json "input constant" in
            let id =
              Json_util.req_field cms "id" tensor_ref_jsont "input constant"
            in
            Tensor_id.Map.add id Input.Constant kinds)
          Tensor_id.Map.empty constants
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
    | Group.Node id ->
        Json_util.single ~case:"Node" (Json_util.jint (Node_id.to_int id))
    | Group.Group child -> Json_util.single ~case:"Group" (enc_group child)
  in
  Json_util.jobj
    ([
       ("id", Json_util.jint (Group_id.to_int group.id));
       ("items", Json_util.jarr (List.map enc_item group.items));
     ]
    @
    match group.label with
    | None -> []
    | Some label -> [ ("label", Json_util.jstr label) ])

let enc_graph (g : graph) : Jsont.json =
  let tensors_json =
    Tensor_id.Map.bindings g.Graph.tensors
    |> List.map (fun (_tid, sg) -> Json_util.enc Tensor_sig.jsont sg)
  in
  let constants =
    List.filter_map
      (fun id ->
        match input_kind g id with
        | Input.Input -> None
        | Input.Constant ->
            Some (Json_util.jobj [ ("id", Json_util.enc tensor_ref_jsont id) ]))
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
