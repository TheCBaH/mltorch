(* Drop a run of permutes whose composition is the identity — the "no-op permute"
   and "no-op sequence of permutes" cases. See .ai/native_transform_design.md §12.

   The run is walked upstream from the anchor with [Pattern.chain]. Only the
   anchor may have consumers outside the run: an intermediate value is a genuine
   rearrangement of the run's input, not equal to it, so a second consumer would
   be left without a producer. [Pattern.interior] enforces that, and failing it
   merely ends the chain — a shorter run may still cancel. *)

open Graph_ir

let as_permute = function
  | Permute (p : Permute.Permute.t) -> Some p
  | _ -> None

(* A consumed link: the node to remove, the edge it defines, its perm, and the
   edge it reads. *)
type link = {
  node : Node_id.t;
  out : Tensor_id.t;
  perm : Permute.Permute.perm;
  x : Tensor_id.t;
}

(* One step: consume the permute defining [edge] and move to its input. *)
let step ~anchor edge =
  let open Pattern in
  let* () = if Tensor_id.equal edge anchor then return () else interior edge in
  let+ (p : Permute.Permute.t), (n : node) = def edge as_permute in
  ({ node = n.Node.id; out = edge; perm = p.perm; x = p.x }, p.x)

(* Composing outward from the run's input says not only whether the whole run
   cancels, but which of its edges already equal that input — those are tied to
   it and land in one value cluster, while the rest are honestly deleted. *)
let cancellations ~base inner_first =
  List.fold_left
    (fun (composite, tied) link ->
      let composite =
        Permute.Permute.compose ~before:composite ~after:link.perm
      in
      ( composite,
        if Permute.Permute.is_identity composite then (link.out, base) :: tied
        else tied ))
    (Permute.Permute.identity, [])
    inner_first

(* A tie claims the two edges are IDENTICAL, and that is false across a change of
   format or quantization: a permute's output is materialized as f32, so trimming
   one off a non-f32 edge drops a materialization the source really performs.
   Declining the match leaves the graph alone, which is the right answer for
   "this rewrite does not apply here" — applying it builds a map
   [Graph_map.create] rejects, and a rejected map takes down [Pass.run_all] and
   every later pass with it. *)
let fmt_name (Payload.Fmt f) = Payload.fmt_name f

let same_precision (a : Tensor_sig.t) (b : Tensor_sig.t) =
  String.equal (fmt_name a.fmt) (fmt_name b.fmt) && a.quant = b.quant

let rec ties_keep_precision ~base_sig = function
  | [] -> Pattern.return true
  | (out, _) :: rest ->
      let open Pattern in
      let* sg = sig_of out in
      if same_precision sg base_sig then ties_keep_precision ~base_sig rest
      else return false

let pattern anchor =
  let open Pattern in
  let* links = chain (step ~anchor) anchor in
  (* [chain] yields the run outermost-first; composition starts at the far end. *)
  match List.rev links with
  | [] -> fail Rejected
  | innermost :: _ as inner_first ->
      let base = innermost.x in
      let composite, tied = cancellations ~base inner_first in
      let* () = guard (Permute.Permute.is_identity composite) in
      let* base_sig = sig_of base in
      let* ok = ties_keep_precision ~base_sig tied in
      let+ () = guard ok in
      (List.map (fun link -> link.node) links, tied)

let build (remove, tie) _region = Recipe.trim ~remove ~tie
let pass = Pass.of_pattern ~name:"trim_permute" ~pattern ~build
