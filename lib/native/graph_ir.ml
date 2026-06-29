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
