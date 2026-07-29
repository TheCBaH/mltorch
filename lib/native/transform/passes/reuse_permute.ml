(* Reuse an existing alternate-layout edge to make a mixed elementwise operand
   set uniform, instead of speculatively inserting a new relayout. See
   .ai/native_layout_reuse_plan.md and .ai/native_transform_design.md §12e,
   which complements §12d's [Sink_permute]: that pass handles [op(P(a), P(b),
   ...) = P(op(a, b, ...))], uniform on every operand. This pass handles the
   mixed case

     op(P(a), b) = P(op(a, Q(b)))

   where [Q = P⁻¹] and [Q(b)] is an edge the graph ALREADY computes — some other
   consumer of [b] is already a [Permute(Q)]. Reusing it costs nothing; inserting
   a fresh [Permute(Q)] for [b] would only move the boundary, never remove one, so
   this pass never does that.

   Unary ops cannot have a mixed operand layout, so the scope is the broadcasting
   binary ops: [Add, Div, Mul, Sub]. *)

open Graph_ir

let binary_elementwise = function
  | (Add _ | Div _ | Mul _ | Sub _) as op -> Some op
  | _ -> None

let as_permute = function
  | Permute (p : Permute.Permute.t) -> Some p
  | _ -> None

(* Order-independent, as in [Sink_permute]: two [perm]s built by different call
   sites need not list their axis pairs in the same order to be the same
   permutation. *)
let perm_equal (a : Permute.Permute.perm) (b : Permute.Permute.perm) =
  List.for_all
    (fun axis ->
      Axis.equal (Permute.Permute.lookup a axis) (Permute.Permute.lookup b axis))
    Axis.all

(* [packed_fmt] is an existential over a GADT and has no usable structural
   order, so compare by format name, same as [Correspondence.Precision]. *)
let precision_equal (a : Tensor_sig.t) (b : Tensor_sig.t) =
  let fmt_key (Payload.Fmt f) = Payload.fmt_name f in
  String.equal (fmt_key a.Tensor_sig.fmt) (fmt_key b.Tensor_sig.fmt)
  && Stdlib.( = ) a.Tensor_sig.quant b.Tensor_sig.quant

(* How one operand was resolved into the target ([P]-input) layout: [Unwrap]
   claims and removes [node] (the [Permute(P)] that produced it); [Reuse]
   keeps [node] (an existing [Permute(Q)] consumer) alive and reads its
   output. [edge] is the resolved tensor id to feed the rebuilt op; [original]
   is the operand edge it replaces, for [map_operands]. Every resolution
   carries [node] — even [Reuse], which never removes it — so a later operand
   can be checked against it: the SAME node must never be both removed by one
   operand's [Unwrap] and read-while-kept by another's [Reuse]. *)
module Resolution = struct
  type kind = Unwrap | Reuse

  type t = {
    edge : Tensor_id.t;
    original : Tensor_id.t;
    kind : kind;
    node : Node_id.t;
  }
end

(* Which nodes earlier operands in THIS match have already committed to each
   role, so a later operand's own candidate search can rule out a node
   already spoken for the other way. *)
module Committed = struct
  type t = { removed : Node_id.Set.t; kept : Node_id.Set.t }
end

let no_commitments =
  { Committed.removed = Node_id.Set.empty; kept = Node_id.Set.empty }

let unwrap_candidate ~target_perm edge =
  let open Pattern in
  let* () = interior edge in
  let* (p : Permute.Permute.t), (n : node) = def edge as_permute in
  let+ () = guard (perm_equal p.perm target_perm) in
  {
    Resolution.edge = p.x;
    original = edge;
    kind = Resolution.Unwrap;
    node = n.Node.id;
  }

(* The nodes worth trying as a [Reuse] for [edge] — one per existing inverse
   consumer — computed without claiming or checking compatibility, so the
   caller can filter by [committed] before spending a claim on one that is
   already ruled out. *)
let reuse_node_candidates ~target_perm edge =
  let open Pattern in
  let+ consumers = uses edge in
  List.filter_map
    (fun (n : node) ->
      match (as_permute n.Node.op, n.Node.outputs) with
      | Some (q : Permute.Permute.t), [ out ]
        when Permute.Permute.are_inverse ~before:target_perm ~after:q.perm ->
          Some (n, out)
      | _ -> None)
    consumers

let reuse_candidate ~edge (n : node) out =
  let open Pattern in
  let* x_sig = sig_of edge in
  let* out_sig = sig_of out in
  let* () = guard (precision_equal x_sig out_sig) in
  let+ () = claim_shared n in
  {
    Resolution.edge = out;
    original = edge;
    kind = Resolution.Reuse;
    node = n.Node.id;
  }

(* Every resolution this operand could try, [Unwrap] first (deterministic:
   "first complete match wins"), then each reuse candidate in graph order
   ([Graph_view.uses] already returns consumers execution-ordered). Built as
   a plain list of independently-runnable [Pattern.t]s — none of them run
   yet — so [resolve_operands] can bundle each one with resolving the REST of
   the operands and let [choice] backtrack across the whole bundle, not just
   this one operand's own pick. *)
let operand_candidates ~target_perm edge =
  let open Pattern in
  let+ reuse_infos = reuse_node_candidates ~target_perm edge in
  unwrap_candidate ~target_perm edge
  :: List.map (fun (n, out) -> reuse_candidate ~edge n out) reuse_infos

(* [Found in review: a self-inverse permutation can be selected as both
   Unwrap and Reuse within ONE match — `add(swap_hw(b), b)` with `swap_hw`
   self-inverse resolves operand 1 by unwrapping (removing) [N], and operand
   2 by reusing [N]'s own output, since [N] is also its OWN inverse
   consumer.] The removed and the kept node sets must stay disjoint, checked
   incrementally as each operand commits: an [Unwrap] is rejected if its node
   was already [Reuse]d by an earlier operand, and vice versa. Rejecting
   inside the SAME [choice] that offers the operand's other candidates (here,
   propagated as this whole bundle's failure) is what lets a conflicting
   pick fall through to a later alternative instead of just failing the
   candidate outright — for THIS operand if it has another candidate, or for
   an EARLIER operand's own choice point if not, since each level's [choice]
   wraps its candidate together with resolving everything after it. *)
let rec resolve_operands ~target_perm ~(committed : Committed.t) = function
  | [] -> Pattern.return []
  | edge :: rest ->
      let open Pattern in
      let* candidates = operand_candidates ~target_perm edge in
      choice
        (List.map
           (fun candidate ->
             let* (r : Resolution.t) = candidate in
             let* () =
               guard
                 (match r.kind with
                 | Resolution.Unwrap ->
                     not (Node_id.Set.mem r.node committed.Committed.kept)
                 | Resolution.Reuse ->
                     not (Node_id.Set.mem r.node committed.Committed.removed))
             in
             let committed' =
               match r.kind with
               | Resolution.Unwrap ->
                   {
                     committed with
                     Committed.removed =
                       Node_id.Set.add r.node committed.Committed.removed;
                   }
               | Resolution.Reuse ->
                   {
                     committed with
                     Committed.kept =
                       Node_id.Set.add r.node committed.Committed.kept;
                   }
             in
             let+ rs =
               resolve_operands ~target_perm ~committed:committed' rest
             in
             r :: rs)
           candidates)

(* The candidate [P]s: one per operand actually produced by a [Permute], in
   operand order, WITHOUT claiming — claiming happens only for the candidate
   that is eventually chosen, inside [resolve_operand]. *)
let candidate_perms operands =
  let open Pattern in
  let rec go = function
    | [] -> return []
    | edge :: rest -> (
        let* maybe = optional (peek edge as_permute) in
        let+ tail = go rest in
        match maybe with
        | Some ((p : Permute.Permute.t), _) -> p.perm :: tail
        | None -> tail)
  in
  go operands

module Match = struct
  type t = {
    op : op;
    op_node : Node_id.t;
    resolutions : Resolution.t list;
    perm : Permute.Permute.perm;
    out : Tensor_id.t;
    y : Tensor_sig.t;
  }
end

(* At least one unwrapped operand AND at least one reused operand: all-unwrap is
   [Sink_permute]'s case, and keeping the passes disjoint keeps their tests and
   mappings easy to interpret; all-reuse would rebuild the op without removing or
   moving a boundary at all. *)
let try_candidate op op_node operands anchor target_perm =
  let open Pattern in
  let* resolutions =
    resolve_operands ~target_perm ~committed:no_commitments operands
  in
  let is_unwrap (r : Resolution.t) =
    match r.kind with Resolution.Unwrap -> true | Resolution.Reuse -> false
  in
  let* () =
    guard
      (List.exists is_unwrap resolutions
      && List.exists (fun r -> not (is_unwrap r)) resolutions)
  in
  let+ (y : Tensor_sig.t) = sig_of anchor in
  { Match.op; op_node; resolutions; perm = target_perm; out = anchor; y }

let pattern anchor =
  let open Pattern in
  let* (op : op), (op_node : node) = def anchor binary_elementwise in
  let operands = Graph_ir.operands op in
  let* perms = candidate_perms operands in
  match perms with
  | [] -> fail Rejected
  | _ ->
      choice (List.map (try_candidate op op_node.Node.id operands anchor) perms)

(* ---- building --------------------------------------------------------------- *)

(* Mirrors [Sink_permute.unpermuted_shape]: invert [perm] against the anchor's
   own shape via [Vec6.copy] — [perm] is already known valid, coming from a real
   [Permute] node in a well-formed graph. *)
let unpermuted_shape ~(perm : Permute.Permute.perm) (y_shape : Vec6.shape) =
  List.fold_left
    (fun s (out_ax, in_ax) -> Vec6.copy y_shape ~src:out_ax ~dst:in_ax s)
    (Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1)
    perm

let build (m : Match.t) _region =
  let open Recipe in
  let shape = unpermuted_shape ~perm:m.perm m.y.Tensor_sig.shape in
  let* pre = fresh ~fmt:m.y.Tensor_sig.fmt ?quant:m.y.Tensor_sig.quant shape in
  let* out = existing m.Match.out in
  (* Raw by construction: [Graph_ir.op] takes [tensor_ref]s, so the rewiring is
     over raw ids and the fresh edge is projected into the payload. *)
  let assoc =
    List.map
      (fun (r : Resolution.t) -> (r.original, r.edge))
      m.Match.resolutions
  in
  let op = Graph_ir.map_operands (fun e -> List.assoc e assoc) m.Match.op in
  let removed_nodes =
    List.filter_map
      (fun (r : Resolution.t) ->
        match r.kind with
        | Resolution.Unwrap -> Some r.node
        | Resolution.Reuse -> None)
      m.Match.resolutions
  in
  replace
    ~remove:(m.Match.op_node :: removed_nodes)
    ~insert:
      [
        { Recipe.op; outputs = [ Fresh pre ]; from = [ m.Match.op_node ] };
        {
          Recipe.op = Permute { perm = m.perm; x = raw_fresh pre };
          outputs = [ Preserved out ];
          from = removed_nodes;
        };
      ]
    ~claims:[ (out, Preserved out, Correspondence.Identical) ]
    ()

let pass = Pass.of_pattern ~name:"reuse_permute" ~pattern ~build:{ Pass.build }
