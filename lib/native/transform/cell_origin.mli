(* What each of two graphs' edges is entitled to mean to the verifier: one
   [Ground_expr.Origin.t] per edge per side. See
   .ai/native_transform_verify.md §7.

   Three answers, and which one an edge gets is the whole of this module:

   - [Input v] for CORRESPONDING GRAPH INPUTS. This is the σ hypothesis — "the
     two graphs are fed the same data" — and it is an assumption, not an
     obligation. It is also the only place a renaming is sound: renaming both
     sides of an internal cluster to one representative would assume the very
     claim under verification.

   - [Shared id] when the two graphs DEFINE the edge identically. Structural,
     computed from the graphs, never read off a label. An explicit [Identical]
     claim is the obligation under verification, not evidence for it —
     [recipe.ml] emits a self-claim precisely because an output "keeps its id
     while changing definition" — so granting [Shared] from a label would assume
     what is being proved.

   - [Src id] / [Dst id] otherwise, including any edge one of whose operands is
     already side-tagged: its definition is then incomparable.

   The comparison is on operand ORIGINS, pairwise, never on their categories.
   That distinction is the point. In `verify_test.ml`'s crossed-inputs case both
   graphs are literally `sub a b` and the map swaps the inputs, so every operand
   on both sides is *some* [Input] — a category test grants [Shared] on the
   output and proves sub(a,b) identical to sub(b,a), which is the false proof
   this whole line of work is named after. Under origin equality the source reads
   [v0; v1], the destination [v1; v0], and they do not match. *)

type t = {
  dst : Graph_ir.Tensor_id.t -> Ground_expr.Origin.t;
  src : Graph_ir.Tensor_id.t -> Ground_expr.Origin.t;
}

(* [clusters] is what [Graph_map.clusters_over] yields — explicit clusters plus
   the implicit identities — because every graph input has to fall in exactly one
   of them for the σ numbering to be total. *)
val classify :
  src:'src Snapshot.t ->
  dst:'dst Snapshot.t ->
  ('src, 'dst) Correspondence.Cluster.t list ->
  t
