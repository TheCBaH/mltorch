(* The Native4D graph. Only the op variant and these two records are new: the
   structural vocabulary is Native's, unchanged.

   [Node_id], [Group_id], [Group], [Input], [Tensor_id] and — critically —
   [Tensor_sig.t] are REUSED verbatim rather than restated. Ids and grouping are
   not dialect-specific, and the signature keeps its [Vec6.shape] so the whole
   symbolic stack ([Stage_program], [Expr], [Symbolic], [Ground_expr]) works on
   a Native4D graph with no functor at all. [Shape4.t] guards op payloads and
   shape inference — the places that must not admit an invalid graph — not
   storage. See .ai/native4d_plan.md, correction C3.

   Stage 4 re-points these at the shared [Graph_common] records; they are
   written out here so stage 2 does not block on that extraction. *)

module Node = struct
  type t = {
    id : Graph_ir.Node_id.t;
    op : Op.t;
    outputs : Tensor_id.t list;
        (* Singleton for every current op — the dialect has no multi-output
           operation, which is why it needs no [Discard] either. A list, still,
           because the shared framework indexes outputs positionally. *)
  }
end

module Graph = struct
  type t = {
    nodes : Node.t list; (* globally topo-ordered by construction *)
    root : Graph_ir.Group.t; (* authoritative structural hierarchy *)
    tensors : Tensor_sig.t Tensor_id.Map.t;
    inputs : Tensor_id.t list; (* ordered = the graph's signature *)
    input_kinds : Graph_ir.Input.kind Tensor_id.Map.t;
    outputs : Tensor_id.t list; (* ordered = the graph's signature *)
  }
end

type node = Node.t
type graph = Graph.t

let nodes (g : graph) = g.Graph.nodes

(* Sparse by design, exactly as [Graph_ir.input_kind] is: the effective
   classification comes from here, not from map membership. *)
let input_kind (g : graph) id =
  Option.value
    (Tensor_id.Map.find_opt id g.Graph.input_kinds)
    ~default:Graph_ir.Input.Input

let pp_node ?(pp_ref = Tensor_id.pp) fmt (n : node) =
  Fmt.pf fmt "@[<hv 2>%a: [%a] =@ %a@]" Graph_ir.Node_id.pp n.Node.id
    (Fmt.list ~sep:Fmt.comma Tensor_id.pp)
    n.Node.outputs (Op.pp_with ~pp_ref) n.Node.op

let pp fmt (g : graph) =
  let sig_of id =
    match Tensor_id.Map.find_opt id g.Graph.tensors with
    | None -> Fmt.str "%a ?" Tensor_id.pp id
    | Some sg ->
        Fmt.str "%a %a" Tensor_id.pp id Vec6.pp_shape sg.Tensor_sig.shape
  in
  Fmt.pf fmt "@[<v>graph4@,inputs: [%a]@,@[<v 2>nodes:@,%a@]@,outputs: [%a]@]"
    (Fmt.list ~sep:Fmt.comma Fmt.string)
    (List.map sig_of g.Graph.inputs)
    (Fmt.list ~sep:Fmt.cut (pp_node ?pp_ref:None))
    g.Graph.nodes
    (Fmt.list ~sep:Fmt.comma Fmt.string)
    (List.map sig_of g.Graph.outputs)
