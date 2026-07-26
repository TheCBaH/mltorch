(* Pure lowering of the static tensor subset of an ExportedProgram. *)

type error =
  [ `Unsupported_input of string
  | `Unsupported_operator of string
  | `Malformed_graph of string
  | `Tensor_bridge of string
  | `Eval of Eval_direct.error
  | `Build of Graph_builder.error
  | `Provenance of Pt2_native_graph.error
  | `Transform of Pass.error
  | `Lens of Pt2_native_graph.lens_error ]

type hooks =
  | Hooks : {
      on_start : Pt2_native_graph.t -> Graph_ir.node -> 'a;
      on_end : Pt2_native_graph.t -> Graph_ir.node -> 'a -> unit;
    }
      -> hooks

val pp_error : Format.formatter -> [< error ] -> unit

(* Lowers a root exported graph into one native graph.  PT2 SSA names remain
   solely in the provenance wrapper; native execution addresses every edge by
   [Tensor_id].  This first static slice covers the ResNet-18 export set. *)
val lower :
  Pytorch_types.ExportedProgram.t -> (Pt2_native_graph.t, error) Core.result

val lower_archive : Pt2_archive.t -> (Pt2_native_graph.t, error) Core.result

(* Execute a one-user-input static graph.  Captured tensor payloads are loaded
   through the sidecar's [Tensor_id -> target] map, never through native IR. *)
val run :
  ?hooks:hooks ->
  Pt2_archive.t ->
  input:Pt2_tensor.t ->
  (Tensor.packed list, error) Core.result

(* What the transformed run did, so a caller can see that the passes fired and
   that both payload sources were reached. *)
type report = {
  nodes_before : int;
  nodes_after : int;
  from_state : int; (* constants a pass computed *)
  from_archive : int; (* constants resolved back to a captured source *)
  derived : (Graph_ir.Tensor_id.t * string list) list;
      (* Constants with NO archive path, and the PT2 names they were computed
         from. A folded weight lands here: its bytes exist only in the transform
         state, while the tensors it derives from are still nameable. That
         separation is the point of keeping provenance out of the value
         lattice. *)
}

(* Execute the graph a pipeline produces, rather than the imported one.

   Payloads come from the two sources §10 of .ai/native_transform_design.md
   separates, in that order: the transform state first, because a pass-computed
   constant exists nowhere else, then the archive, reached by resolving the
   destination id back to a captured source through the lens. The archive is
   never reached through PROVENANCE — for a folded weight the captured bytes are
   the pre-fold layout and handing them back would be corruption, so the lens
   follows only an [Identical] correspondence and such an edge simply has no
   archive path.

   The state is seeded with no payloads, so this is the lazy path: only what the
   destination graph reads is loaded, and structural passes ([Fold_batch_norm],
   the permute simplifications) run without materialising any weight at all.
   [Fold_const] correspondingly folds nothing on that path — it declines a
   constant whose payload is not bound. [~preload:true] binds every captured
   payload up front instead, which is what lets folding hoist a permuted weight
   to load time; it costs reading the whole archive, so it is not the default. *)
val run_transformed :
  ?preload:bool ->
  Pt2_archive.t ->
  passes:Pass.t list ->
  input:Pt2_tensor.t ->
  (Tensor.packed list * report, error) Core.result
