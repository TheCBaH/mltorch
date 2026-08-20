(* Sink a permute through the elementwise op reading it, so a permute trapped
   on the far side of a `Relu`/`Add`/... from its inverse partner can still
   reach it. See .ai/native_transform_design.md §12d.

   An elementwise op reads and writes each output slot independently of every
   other slot, with no axis-specific semantics, so it commutes exactly (bit
   for bit, not just numerically) with a permutation applied uniformly to
   every operand:

     op(permute_p(a), permute_p(b), ...) = permute_p(op(a, b, ...))

   For a unary op this is immediate (same shape in and out). For the
   broadcasting binary ops it still holds: broadcast comparison is per axis,
   and [p] maps corresponding axes of every operand identically, so the
   broadcast pattern permutes along with the data. It does NOT hold for
   [Mean] (reduces over named axes) or anything else with axis-specific
   semantics (conv, pool, bmm, linear, norm, reshape, permute itself, sdpa —
   which NAMES D as the batch axis and H as heads, W as the query/key
   sequence and C as the head dimension, so it is a barrier for the same
   reason [Layer_norm] is) — hence the explicit allowlist below rather than
   reaching for [Graph_ir.operands] on an arbitrary op. *)

open Graph_ir

(* The scalar-operand and clamped ops qualify for the same reason the rest do:
   their parameters ([scalar], the clamp bounds) are per-op constants, not
   per-axis data, so permuting the operand permutes the result identically. *)
let elementwise = function
  | ( Add _ | Add_scalar _ | Clamp _ | Clone _ | Div _ | Div_scalar _
    | Hardsigmoid _ | Hardswish _ | Hardtanh _ | Mul _ | Relu _ | Silu _
    | Sqrt _ | Sub _ ) as op ->
      Some op
  | _ -> None

let as_permute = function
  | Permute (p : Permute.Permute.t) -> Some p
  | _ -> None

(* A consumed operand: the permute node to remove, its perm, the operand edge
   it used to feed (so [map_operands] can be told what to replace), and its
   own input (what the rebuilt op will read instead). Its own module, per the
   repo convention: a top-level record risks a field-label clash with
   [Match.t] below ([perm]) that [type t = { ... }] rules out by construction. *)
module Operand_link = struct
  type t = {
    node : Node_id.t;
    perm : Permute.Permute.perm;
    out : Tensor_id.t;
    x : Tensor_id.t;
  }
end

let match_operand edge =
  let open Pattern in
  let* () = interior edge in
  let+ (p : Permute.Permute.t), (n : node) = def edge as_permute in
  { Operand_link.node = n.Node.id; perm = p.perm; out = edge; x = p.x }

(* Operands are independent (unlike a permute run), so this just visits each
   edge in turn rather than walking a chain. *)
let rec match_operands = function
  | [] -> Pattern.return []
  | edge :: rest ->
      let open Pattern in
      let* link = match_operand edge in
      let+ links = match_operands rest in
      link :: links

(* Order-independent: two [perm]s built by different call sites need not list
   their axis pairs in the same order to be the same permutation. *)
let perm_equal (a : Permute.Permute.perm) (b : Permute.Permute.perm) =
  List.for_all
    (fun axis ->
      Axis.equal (Permute.Permute.lookup a axis) (Permute.Permute.lookup b axis))
    Axis.all

module Match = struct
  type t = {
    op : op;
    op_node : Node_id.t;
    links : Operand_link.t list;
    perm : Permute.Permute.perm;
    out : Tensor_id.t;
    y : Tensor_sig.t;
  }
end

let pattern anchor =
  let open Pattern in
  let* (op : op), (op_node : node) = def anchor elementwise in
  let* links = match_operands (Graph_ir.operands op) in
  match links with
  | [] -> fail Rejected
  | (first : Operand_link.t) :: rest ->
      let* () =
        guard
          (List.for_all
             (fun (l : Operand_link.t) ->
               perm_equal l.Operand_link.perm first.Operand_link.perm)
             rest)
      in
      let+ (y : Tensor_sig.t) = sig_of anchor in
      {
        Match.op;
        op_node = op_node.Node.id;
        links;
        perm = first.Operand_link.perm;
        out = anchor;
        y;
      }

(* ---- building ------------------------------------------------------------- *)

(* The un-permuted op's output shape: invert [perm] against the anchor's own
   shape directly with [Vec6.copy], mirroring [Permute.Permute.output_shape]'s
   own fold minus validation — [perm] is already known valid, having come from
   real [Permute] nodes in a well-formed graph. *)
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
  (* [op] is raw by construction: [Graph_ir.op] takes [tensor_ref]s, so the
     rewiring happens on raw ids and the fresh edge is projected into it. *)
  let assoc =
    List.map (fun (l : Operand_link.t) -> (l.out, l.x)) m.Match.links
  in
  let op = Graph_ir.map_operands (fun e -> List.assoc e assoc) m.Match.op in
  let permute_nodes =
    List.map (fun (l : Operand_link.t) -> l.node) m.Match.links
  in
  replace
    ~remove:(m.Match.op_node :: permute_nodes)
    ~insert:
      [
        { Recipe.op; outputs = [ Fresh pre ]; from = [ m.Match.op_node ] };
        {
          Recipe.op = Permute { perm = m.perm; x = raw_fresh pre };
          outputs = [ Preserved out ];
          from = permute_nodes;
        };
      ]
    ~claims:[ (out, Preserved out, Correspondence.Identical) ]
    ()

let pass = Pass.of_pattern ~name:"sink_permute" ~pattern ~build:{ Pass.build }
