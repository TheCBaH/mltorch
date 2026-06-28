(* See graph_ir.mli. Ids are plain ints under the hood (sealed [private int] by the
   interface); the builder owns their allocation so they are unique across the
   whole graph tree and deterministic per top-level build. [op] is parametrised
   over the embedded-graph type, so only the two small records [Node.t]/[Graph.t]
   need the recursive module group. *)

module Tensor_id = struct
  type t = int

  let of_int x = x
  let to_int x = x
  let equal = Int.equal
  let compare = Int.compare
  let pp fmt x = Format.fprintf fmt "t%d" x

  module Map = Map.Make (struct
    type nonrec t = t

    let compare = compare
  end)
end

module Node_id = struct
  type t = int

  let of_int x = x
  let to_int x = x
  let pp fmt x = Format.fprintf fmt "n%d" x
end

type tensor_ref = Tensor_id.t

type 'g gop =
  (* Constructors kept in global alphabetical order (see graph_ir.mli). *)
  | Add of { a : tensor_ref; b : tensor_ref }
  | Avg_pool2d of { params : Pool.AvgPool2d.params; x : tensor_ref }
  | Bmm of { input : tensor_ref; mat2 : tensor_ref }
  | Conv2d of {
      params : Conv.Conv2d.params;
      x : tensor_ref;
      weight : tensor_ref;
      bias : tensor_ref option;
    }
  | Linear of {
      params : Linear.Linear.params;
      x : tensor_ref;
      weight : tensor_ref;
      bias : tensor_ref option;
    }
  | Max_pool2d of { params : Pool.MaxPool2d.params; x : tensor_ref }
  | Mean of { params : Reduce.Mean.params; x : tensor_ref }
  | Mul of { a : tensor_ref; b : tensor_ref }
  | Permute of { perm : Permute.Permute.perm; x : tensor_ref }
  | Relu of { x : tensor_ref }
  | Rms_norm of {
      params : Norm.RmsNorm.params;
      x : tensor_ref;
      weight : tensor_ref option;
    }
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

let operands : op -> tensor_ref list = function
  | Add { a; b } -> [ a; b ]
  | Avg_pool2d { x; _ } -> [ x ]
  | Bmm { input; mat2 } -> [ input; mat2 ]
  | Conv2d { x; weight; bias; _ } -> [ x; weight ] @ Option.to_list bias
  | Linear { x; weight; bias; _ } -> [ x; weight ] @ Option.to_list bias
  | Max_pool2d { x; _ } -> [ x ]
  | Mean { x; _ } -> [ x ]
  | Mul { a; b } -> [ a; b ]
  | Permute { x; _ } -> [ x ]
  | Relu { x } -> [ x ]
  | Rms_norm { x; weight; _ } -> x :: Option.to_list weight
  | Subgraph { args; _ } -> args

(* Inline records (the [of { ... }] payloads) can't be captured as a value nor
   updated with [{ r with ... }], so each arm reconstructs explicitly. *)
let map_operands (f : tensor_ref -> tensor_ref) : op -> op = function
  | Add { a; b } -> Add { a = f a; b = f b }
  | Avg_pool2d { params; x } -> Avg_pool2d { params; x = f x }
  | Bmm { input; mat2 } -> Bmm { input = f input; mat2 = f mat2 }
  | Conv2d { params; x; weight; bias } ->
      Conv2d { params; x = f x; weight = f weight; bias = Option.map f bias }
  | Linear { params; x; weight; bias } ->
      Linear { params; x = f x; weight = f weight; bias = Option.map f bias }
  | Max_pool2d { params; x } -> Max_pool2d { params; x = f x }
  | Mean { params; x } -> Mean { params; x = f x }
  | Mul { a; b } -> Mul { a = f a; b = f b }
  | Permute { perm; x } -> Permute { perm; x = f x }
  | Relu { x } -> Relu { x = f x }
  | Rms_norm { params; x; weight } ->
      Rms_norm { params; x = f x; weight = Option.map f weight }
  | Subgraph { graph; args } -> Subgraph { graph; args = List.map f args }

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

let pp_optional_tensor_ref g =
  Fmt.option ~none:(Fmt.any "none") (pp_tensor_ref g)

let pp_axis_pair fmt (out_axis, in_axis) =
  Fmt.pf fmt "@[<h>%a<-%a@]" Axis.pp out_axis Axis.pp in_axis

let pp_perm fmt perm =
  pp_list pp_axis_pair fmt
    (List.filter
       (fun (out_axis, in_axis) -> not (Axis.equal out_axis in_axis))
       perm)

let pp_axis = Axis.pp
let pp_axes = pp_list pp_axis

let pp_hw pp fmt h =
  Fmt.pf fmt "@[<hv>{h=%a;@ w=%a}@]" pp h.Op_config.Hw.h pp h.Op_config.Hw.w

let pp_pos fmt p = Fmt.int fmt (Op_config.Pos.to_int p)
let pp_nonneg fmt p = Fmt.int fmt (Op_config.Nonneg.to_int p)

let pp_conv2d_params fmt (p : Conv.Conv2d.params) =
  Fmt.pf fmt "@[<hv>{kernel=%a;@ in_channels=%a;@ stride=%a;@ pad=%a}@]"
    (pp_hw Dim.pp) p.kernel Dim.pp p.in_channels (pp_hw pp_pos) p.stride
    (pp_hw pp_nonneg) p.pad

let pp_linear_params fmt (p : Linear.Linear.params) =
  Fmt.pf fmt "@[<hv>{in_features=%a}@]" Dim.pp p.in_features

let pp_max_pool2d_params fmt (p : Pool.MaxPool2d.params) =
  Fmt.pf fmt "@[<hv>{kernel=%a;@ stride=%a;@ pad=%a}@]" (pp_hw Dim.pp) p.kernel
    (pp_hw pp_pos) p.stride (pp_hw pp_nonneg) p.pad

let pp_avg_pool2d_params fmt (p : Pool.AvgPool2d.params) =
  Fmt.pf fmt "@[<hv>{kernel=%a;@ stride=%a;@ pad=%a}@]" (pp_hw Dim.pp) p.kernel
    (pp_hw pp_pos) p.stride (pp_hw pp_nonneg) p.pad

let pp_mean_params fmt (p : Reduce.Mean.params) =
  Fmt.pf fmt "@[<hv>{dims=%a;@ keepdim=%a}@]" pp_axes p.dims Fmt.bool p.keepdim

let pp_rms_norm_params fmt (p : Norm.RmsNorm.params) =
  Fmt.pf fmt "@[<hv>{dims=%a;@ eps=%a}@]" pp_axes p.dims Fmt.float p.eps

let pp_op_header g fmt (node : node) =
  Fmt.pf fmt "@[<h>%a: %a =@]" Node_id.pp node.Node.id (pp_tensor_defs g)
    node.outputs

let pp_op (g : graph) fmt : op -> unit = function
  | Add { a; b } ->
      Fmt.pf fmt "@[<hv 2>add@ a=%a@ b=%a@]" (pp_tensor_ref g) a
        (pp_tensor_ref g) b
  | Avg_pool2d { params; x } ->
      Fmt.pf fmt "@[<hv 2>avg_pool2d@ x=%a@ params=%a@]" (pp_tensor_ref g) x
        pp_avg_pool2d_params params
  | Bmm { input; mat2 } ->
      Fmt.pf fmt "@[<hv 2>bmm@ input=%a@ mat2=%a@]" (pp_tensor_ref g) input
        (pp_tensor_ref g) mat2
  | Conv2d { params; x; weight; bias } ->
      Fmt.pf fmt "@[<hv 2>conv2d@ x=%a@ weight=%a@ bias=%a@ params=%a@]"
        (pp_tensor_ref g) x (pp_tensor_ref g) weight (pp_optional_tensor_ref g)
        bias pp_conv2d_params params
  | Linear { params; x; weight; bias } ->
      Fmt.pf fmt "@[<hv 2>linear@ x=%a@ weight=%a@ bias=%a@ params=%a@]"
        (pp_tensor_ref g) x (pp_tensor_ref g) weight (pp_optional_tensor_ref g)
        bias pp_linear_params params
  | Max_pool2d { params; x } ->
      Fmt.pf fmt "@[<hv 2>max_pool2d@ x=%a@ params=%a@]" (pp_tensor_ref g) x
        pp_max_pool2d_params params
  | Mean { params; x } ->
      Fmt.pf fmt "@[<hv 2>mean@ x=%a@ params=%a@]" (pp_tensor_ref g) x
        pp_mean_params params
  | Mul { a; b } ->
      Fmt.pf fmt "@[<hv 2>mul@ a=%a@ b=%a@]" (pp_tensor_ref g) a
        (pp_tensor_ref g) b
  | Permute { perm; x } ->
      Fmt.pf fmt "@[<hv 2>permute@ x=%a@ perm=%a@]" (pp_tensor_ref g) x pp_perm
        perm
  | Relu { x } -> Fmt.pf fmt "@[<hv 2>relu@ x=%a@]" (pp_tensor_ref g) x
  | Rms_norm { params; x; weight } ->
      Fmt.pf fmt "@[<hv 2>rms_norm@ x=%a@ weight=%a@ params=%a@]"
        (pp_tensor_ref g) x (pp_optional_tensor_ref g) weight pp_rms_norm_params
        params
  | Subgraph { graph; args } ->
      Fmt.pf fmt "@[<hv 2>subgraph@ %s@ args=%a@]" graph.Graph.name
        (pp_tensor_refs g) args

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

let tensor_ref_jsont : tensor_ref Jsont.t =
  Jsont.map ~kind:"tensor_ref" ~dec:Tensor_id.of_int ~enc:Tensor_id.to_int
    Jsont.int

let opt_ref ms key = Json_util.opt_field ms key tensor_ref_jsont

let rec dec_op (json : Jsont.json) : op =
  Json_util.union ~kind:"op"
    [
      ( "Add",
        fun v ->
          let ms = Json_util.req_obj v "Add" in
          let get k c = Json_util.req_field ms k c "Add" in
          Add { a = get "a" tensor_ref_jsont; b = get "b" tensor_ref_jsont } );
      ( "Avg_pool2d",
        fun v ->
          let ms = Json_util.req_obj v "Avg_pool2d" in
          let get k c = Json_util.req_field ms k c "Avg_pool2d" in
          Avg_pool2d
            {
              params = get "params" Pool.AvgPool2d.params_jsont;
              x = get "x" tensor_ref_jsont;
            } );
      ( "Bmm",
        fun v ->
          let ms = Json_util.req_obj v "Bmm" in
          let get k c = Json_util.req_field ms k c "Bmm" in
          Bmm
            {
              input = get "input" tensor_ref_jsont;
              mat2 = get "mat2" tensor_ref_jsont;
            } );
      ( "Conv2d",
        fun v ->
          let ms = Json_util.req_obj v "Conv2d" in
          let get k c = Json_util.req_field ms k c "Conv2d" in
          Conv2d
            {
              params = get "params" Conv.Conv2d.params_jsont;
              x = get "x" tensor_ref_jsont;
              weight = get "weight" tensor_ref_jsont;
              bias = opt_ref ms "bias";
            } );
      ( "Linear",
        fun v ->
          let ms = Json_util.req_obj v "Linear" in
          let get k c = Json_util.req_field ms k c "Linear" in
          Linear
            {
              params = get "params" Linear.Linear.params_jsont;
              x = get "x" tensor_ref_jsont;
              weight = get "weight" tensor_ref_jsont;
              bias = opt_ref ms "bias";
            } );
      ( "Max_pool2d",
        fun v ->
          let ms = Json_util.req_obj v "Max_pool2d" in
          let get k c = Json_util.req_field ms k c "Max_pool2d" in
          Max_pool2d
            {
              params = get "params" Pool.MaxPool2d.params_jsont;
              x = get "x" tensor_ref_jsont;
            } );
      ( "Mean",
        fun v ->
          let ms = Json_util.req_obj v "Mean" in
          let get k c = Json_util.req_field ms k c "Mean" in
          Mean
            {
              params = get "params" Reduce.Mean.params_jsont;
              x = get "x" tensor_ref_jsont;
            } );
      ( "Mul",
        fun v ->
          let ms = Json_util.req_obj v "Mul" in
          let get k c = Json_util.req_field ms k c "Mul" in
          Mul { a = get "a" tensor_ref_jsont; b = get "b" tensor_ref_jsont } );
      ( "Permute",
        fun v ->
          let ms = Json_util.req_obj v "Permute" in
          let get k c = Json_util.req_field ms k c "Permute" in
          Permute
            {
              perm = get "perm" Permute.Permute.perm_jsont;
              x = get "x" tensor_ref_jsont;
            } );
      ( "Relu",
        fun v ->
          let ms = Json_util.req_obj v "Relu" in
          let get k c = Json_util.req_field ms k c "Relu" in
          Relu { x = get "x" tensor_ref_jsont } );
      ( "Rms_norm",
        fun v ->
          let ms = Json_util.req_obj v "Rms_norm" in
          let get k c = Json_util.req_field ms k c "Rms_norm" in
          Rms_norm
            {
              params = get "params" Norm.RmsNorm.params_jsont;
              x = get "x" tensor_ref_jsont;
              weight = opt_ref ms "weight";
            } );
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
            } );
    ]
    json

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
      (fun m (sg : Tensor_sig.t) ->
        Tensor_id.Map.add (Tensor_id.of_int sg.id) sg m)
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
  let ref_ = Json_util.enc tensor_ref_jsont in
  let opt_ref_ = function None -> [] | Some r -> [ ("bias", ref_ r) ] in
  let opt_weight_ = function None -> [] | Some r -> [ ("weight", ref_ r) ] in
  match op with
  | Add { a; b } ->
      Json_util.single ~case:"Add"
        (Json_util.jobj [ ("a", ref_ a); ("b", ref_ b) ])
  | Avg_pool2d { params; x } ->
      Json_util.single ~case:"Avg_pool2d"
        (Json_util.jobj
           [
             ("params", Json_util.enc Pool.AvgPool2d.params_jsont params);
             ("x", ref_ x);
           ])
  | Bmm { input; mat2 } ->
      Json_util.single ~case:"Bmm"
        (Json_util.jobj [ ("input", ref_ input); ("mat2", ref_ mat2) ])
  | Conv2d { params; x; weight; bias } ->
      Json_util.single ~case:"Conv2d"
        (Json_util.jobj
           (opt_ref_ bias
           @ [
               ("params", Json_util.enc Conv.Conv2d.params_jsont params);
               ("weight", ref_ weight);
               ("x", ref_ x);
             ]))
  | Linear { params; x; weight; bias } ->
      Json_util.single ~case:"Linear"
        (Json_util.jobj
           (opt_ref_ bias
           @ [
               ("params", Json_util.enc Linear.Linear.params_jsont params);
               ("weight", ref_ weight);
               ("x", ref_ x);
             ]))
  | Max_pool2d { params; x } ->
      Json_util.single ~case:"Max_pool2d"
        (Json_util.jobj
           [
             ("params", Json_util.enc Pool.MaxPool2d.params_jsont params);
             ("x", ref_ x);
           ])
  | Mean { params; x } ->
      Json_util.single ~case:"Mean"
        (Json_util.jobj
           [
             ("params", Json_util.enc Reduce.Mean.params_jsont params);
             ("x", ref_ x);
           ])
  | Mul { a; b } ->
      Json_util.single ~case:"Mul"
        (Json_util.jobj [ ("a", ref_ a); ("b", ref_ b) ])
  | Permute { perm; x } ->
      Json_util.single ~case:"Permute"
        (Json_util.jobj
           [
             ("perm", Json_util.enc Permute.Permute.perm_jsont perm);
             ("x", ref_ x);
           ])
  | Relu { x } ->
      Json_util.single ~case:"Relu" (Json_util.jobj [ ("x", ref_ x) ])
  | Rms_norm { params; x; weight } ->
      Json_util.single ~case:"Rms_norm"
        (Json_util.jobj
           ([ ("params", Json_util.enc Norm.RmsNorm.params_jsont params) ]
           @ opt_weight_ weight
           @ [ ("x", ref_ x) ]))
  | Subgraph { graph; args } ->
      Json_util.single ~case:"Subgraph"
        (Json_util.jobj
           [
             ("args", Json_util.jarr (List.map ref_ args));
             ("graph", enc_graph graph);
           ])

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
    |> List.map (fun (tid, sg) ->
        (* sg.id is from a global allocation counter; use the graph-local
            Tensor_id as the JSON key so refs and tensor metadata agree. *)
        let sg = { sg with Tensor_sig.id = Tensor_id.to_int tid } in
        Json_util.enc Tensor_sig.jsont sg)
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
