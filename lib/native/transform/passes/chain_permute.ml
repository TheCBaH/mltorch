(* Fuse two adjacent permutes into one. See .ai/native_transform_design.md §12.

   The intermediate edge must be interior: it is a real rearrangement, so a
   second consumer still needs the node producing it, and "fusing" would then
   duplicate work rather than remove it. Under [Pass.fixpoint] this collapses a
   whole run to a single node. *)

open Graph_ir

type match_ = {
  out : Tensor_id.t; (* the fused node takes this id over *)
  perm : Permute.Permute.perm;
  remove : Node_id.t list;
  x : Tensor_id.t;
}

let as_permute = function
  | Permute (p : Permute.Permute.t) -> Some p
  | _ -> None

let pattern anchor =
  let open Pattern in
  let* (outer : Permute.Permute.t), (outer_node : node) =
    def anchor as_permute
  in
  let* () = interior outer.x in
  let+ (inner : Permute.Permute.t), (inner_node : node) =
    def outer.x as_permute
  in
  {
    out = anchor;
    perm = Permute.Permute.compose ~before:inner.perm ~after:outer.perm;
    remove = [ inner_node.Node.id; outer_node.Node.id ];
    x = inner.x;
  }

(* [out] keeps its id: the fused permute computes the very same tensor, so the
   id-identity rule is satisfied. Its DEFINITION changes, though, and [apply]
   never infers a claim — hence the explicit self-claim. *)
let build m _region =
  Recipe.replace ~remove:m.remove
    ~insert:
      [
        {
          Recipe.op = Permute { perm = m.perm; x = m.x };
          outputs = [ m.out ];
          from = m.remove;
        };
      ]
    ~claims:[ (m.out, m.out, Correspondence.Identical) ]
    ()

let pass = Pass.of_pattern ~name:"chain_permute" ~pattern ~build
