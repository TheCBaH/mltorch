(* The walk accumulator [acc] and its low-level operations, split out of
   lower_engine.ml under the tracked file-size ceiling so both
   lower_engine.ml and lower_engine_batch_norm.ml (the [Batch_norm] arm,
   split out separately) can depend on this without depending on each
   other. lower.ml also [open]s this, for [lower_node]/[resolve]/the [acc]
   record fields, exactly as it did when they lived in lower_engine.ml. *)

(* Bound BEFORE [open Graph_ir], which shadows [Graph] with the Native one. *)
module G4 = Graph
open Graph_ir

(* ---- what the walk accumulates -------------------------------------------- *)

type acc = {
  nodes : G4.node list; (* reversed *)
  tensors : Tensor_sig.t Tensor_id.Map.t;
  subst : Tensor_id.t Tensor_id.Map.t; (* clone removal, source-side rewiring *)
  next_tid : int;
  next_nid : int;
  created : Tensor_id.t list; (* fresh destination edges *)
  deleted : Tensor_id.t list; (* source edges with no destination *)
  claims : (Tensor_id.t * Correspondence.relation) list;
      (* weaker than Identical *)
  node_pairs : (Node_id.t * Node_id.t list) list;
  provenance : (Tensor_id.t list * Tensor_id.t) list;
  constants : Tensor.packed Tensor_id.Map.t;
  fresh_constants : Tensor_id.t list; (* new captured state, in creation order *)
}

let resolve acc id =
  Option.value (Tensor_id.Map.find_opt id acc.subst) ~default:id

let fresh_tensor acc shape =
  let id = Tensor_id.of_int acc.next_tid in
  let sg =
    Tensor_sig.create ~id ~name:"" ~shape:(Shape4.to_vec6 shape)
      ~fmt:(Payload.Fmt Payload.F32) ()
  in
  ( id,
    {
      acc with
      next_tid = acc.next_tid + 1;
      tensors = Tensor_id.Map.add id sg acc.tensors;
      created = id :: acc.created;
    } )

(* A fresh CONSTANT is not just a signature: it is captured model state, so it
   has to join [Graph.inputs] with kind [Constant] as well. Omitting that leaves
   it defined by no node and declared no input, which validation rejects — the
   symptom being an operand with no definition. *)
let fresh_constant acc shape payload =
  let id, acc = fresh_tensor acc shape in
  ( id,
    {
      acc with
      fresh_constants = id :: acc.fresh_constants;
      constants = Tensor_id.Map.add id payload acc.constants;
    } )

(* A destination node taking over [outputs]; [from] is the source node it came
   from, which is what the node map records.

   NODE ids follow the same policy as tensor ids, and for the same reason. The
   first destination node of a source node KEEPS that node's id; a second one
   (Mean keepdim=false) takes a fresh id above the source watermark.
   Allocating densely from zero instead would make destination node 0 a
   different node from source node 0 whenever anything was removed — the raw-id
   collision the design forbids for edges, reappearing for nodes. *)
let emit acc ~from op outputs =
  let already =
    List.exists (fun (s, _) -> Node_id.equal s from) acc.node_pairs
  in
  let nid = if already then Node_id.of_int acc.next_nid else from in
  let acc =
    {
      acc with
      next_nid = (if already then acc.next_nid + 1 else acc.next_nid);
      nodes = { G4.Node.id = nid; op; outputs } :: acc.nodes;
    }
  in
  let node_pairs =
    List.map
      (fun (s, ds) -> if Node_id.equal s from then (s, nid :: ds) else (s, ds))
      acc.node_pairs
  in
  let node_pairs =
    if List.exists (fun (s, _) -> Node_id.equal s from) node_pairs then
      node_pairs
    else (from, [ nid ]) :: node_pairs
  in
  { acc with node_pairs }
