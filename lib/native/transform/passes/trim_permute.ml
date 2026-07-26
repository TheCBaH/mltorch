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

let pattern anchor =
  let open Pattern in
  let* links = chain (step ~anchor) anchor in
  (* [chain] yields the run outermost-first; composition starts at the far end. *)
  match List.rev links with
  | [] -> fail Rejected
  | innermost :: _ as inner_first ->
      let base = innermost.x in
      let composite, tied = cancellations ~base inner_first in
      let+ () = guard (Permute.Permute.is_identity composite) in
      (List.map (fun link -> link.node) links, tied)

let build (remove, tie) _region = Recipe.trim ~remove ~tie
let pass = Pass.of_pattern ~name:"trim_permute" ~pattern ~build
