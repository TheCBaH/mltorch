(* The graph vocabulary that is not dialect-specific: ids, grouping, input
   classification, and the two records parametrised by the operation type.

   Extracted from [Graph_ir] so a second dialect can reuse it rather than
   restate it. The records have to move for that, and the ID MODULES have to
   move WITH them — [Graph_common.Node.t] needs [Node_id.t], so leaving
   [Node_id] inside [Graph_ir] and referring to it from here would close a
   compilation-unit cycle and neither unit would build. [Tensor_id] already sets
   the precedent: its own top-level unit, re-exported from [Graph_ir] as an
   alias.

   [Graph_ir] re-exports everything here, monomorphically for the records, so no
   caller changes. See .ai/native4d_plan.md stage 4. *)

module Node_id = struct
  type t = int

  let of_int x = x
  let to_int x = x
  let equal = Int.equal
  let compare = Int.compare
  let pp fmt x = Format.fprintf fmt "n%d" x

  module Ord = struct
    type nonrec t = t

    let compare = compare
  end

  module Map = Map.Make (Ord)
  module Set = Set.Make (Ord)
end

module Group_id = struct
  type t = int

  let of_int x = x
  let to_int x = x
  let equal = Int.equal
  let compare = Int.compare
  let pp fmt x = Format.fprintf fmt "g%d" x

  module Ord = struct
    type nonrec t = t

    let compare = compare
  end

  module Map = Map.Make (Ord)
  module Set = Set.Make (Ord)
end

(* Inference source classification. [Input] is supplied by the caller per run;
   [Constant] is model-bound state. *)
module Input = struct
  type kind = Constant | Input

  let pp_kind fmt = function
    | Constant -> Format.pp_print_string fmt "constant"
    | Input -> Format.pp_print_string fmt "input"
end

(* An authoritative structural hierarchy over nodes, never a call boundary. No
   type parameter: grouping names node ids, not operations. *)
module Group = struct
  type item = Group of t | Node of Node_id.t
  and t = { id : Group_id.t; label : string option; items : item list }
end

module Node = struct
  type 'op t = { id : Node_id.t; op : 'op; outputs : Tensor_id.t list }
end

(* One parameter, not two: the tensor map stays [Tensor_sig.t], whose shape is a
   [Vec6.shape] in every dialect. Parametrising the signature by shape type
   would drag [Stage_program], [Expr], [Symbolic] and [Ground_expr] through a
   functor for no gain — the physical frame is identical across the dialect
   boundary. See .ai/native4d_plan.md, correction C3. *)
module Graph = struct
  type 'op t = {
    nodes : 'op Node.t list; (* globally topo-ordered by construction *)
    root : Group.t;
    tensors : Tensor_sig.t Tensor_id.Map.t;
    inputs : Tensor_id.t list; (* ordered = the graph's signature *)
    input_kinds : Input.kind Tensor_id.Map.t;
    outputs : Tensor_id.t list; (* ordered = the graph's signature *)
  }
end

let nodes (g : 'op Graph.t) = g.Graph.nodes

(* The EFFECTIVE classification. [input_kinds] is sparse by design — validation
   requires only that its keys are inputs, not that it covers them — so reading
   the map directly at a call site would silently change the answer for an
   unlisted input. That is why this is a function rather than record access, and
   why it moves down here with the record. *)
let input_kind (g : 'op Graph.t) id =
  Option.value
    (Tensor_id.Map.find_opt id g.Graph.input_kinds)
    ~default:Input.Input
