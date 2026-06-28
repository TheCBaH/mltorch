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
  | Relu of { x : tensor_ref }
  | Add of { a : tensor_ref; b : tensor_ref }
  | Bmm of { input : tensor_ref; mat2 : tensor_ref }
  | Conv2d of {
      params : Conv.Conv2d.params;
      x : tensor_ref;
      weight : tensor_ref;
      bias : tensor_ref option;
    }
  | Permute of { perm : Permute.Permute.perm; x : tensor_ref }
  | Mean of { params : Reduce.Mean.params; x : tensor_ref }
  | Rms_norm of {
      params : Norm.RmsNorm.params;
      x : tensor_ref;
      weight : tensor_ref option;
    }
  | Linear of {
      params : Linear.Linear.params;
      x : tensor_ref;
      weight : tensor_ref;
      bias : tensor_ref option;
    }
  | Max_pool2d of { params : Pool.MaxPool2d.params; x : tensor_ref }
  | Avg_pool2d of { params : Pool.AvgPool2d.params; x : tensor_ref }
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
  | Relu { x } -> [ x ]
  | Add { a; b } -> [ a; b ]
  | Bmm { input; mat2 } -> [ input; mat2 ]
  | Conv2d { x; weight; bias; _ } -> [ x; weight ] @ Option.to_list bias
  | Permute { x; _ } -> [ x ]
  | Mean { x; _ } -> [ x ]
  | Rms_norm { x; weight; _ } -> x :: Option.to_list weight
  | Linear { x; weight; bias; _ } -> [ x; weight ] @ Option.to_list bias
  | Max_pool2d { x; _ } -> [ x ]
  | Avg_pool2d { x; _ } -> [ x ]
  | Subgraph { args; _ } -> args

(* Inline records (the [of { ... }] payloads) can't be captured as a value nor
   updated with [{ r with ... }], so each arm reconstructs explicitly. *)
let map_operands (f : tensor_ref -> tensor_ref) : op -> op = function
  | Relu { x } -> Relu { x = f x }
  | Add { a; b } -> Add { a = f a; b = f b }
  | Bmm { input; mat2 } -> Bmm { input = f input; mat2 = f mat2 }
  | Conv2d { params; x; weight; bias } ->
      Conv2d { params; x = f x; weight = f weight; bias = Option.map f bias }
  | Permute { perm; x } -> Permute { perm; x = f x }
  | Mean { params; x } -> Mean { params; x = f x }
  | Rms_norm { params; x; weight } ->
      Rms_norm { params; x = f x; weight = Option.map f weight }
  | Linear { params; x; weight; bias } ->
      Linear { params; x = f x; weight = f weight; bias = Option.map f bias }
  | Max_pool2d { params; x } -> Max_pool2d { params; x = f x }
  | Avg_pool2d { params; x } -> Avg_pool2d { params; x = f x }
  | Subgraph { graph; args } -> Subgraph { graph; args = List.map f args }
