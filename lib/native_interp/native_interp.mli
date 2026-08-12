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
  | `Verify of Map_verify.error
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
val lower : Pytorch_types.ExportedProgram.t -> (Pt2_native_graph.t, error) Err.t
val lower_archive : Pt2_archive.t -> (Pt2_native_graph.t, error) Err.t

(* Execute a one-user-input static graph.  Captured tensor payloads are loaded
   through the sidecar's [Tensor_id -> target] map, never through native IR. *)
val run :
  ?hooks:hooks ->
  Pt2_archive.t ->
  input:Pt2_tensor.t ->
  (Tensor.packed list, error) Err.t

(* Transforming, printing and executing are separate: a caller that only wants to
   SEE what a pipeline produced should not have to run inference to find out. *)

(* The result of a pipeline, existential in the destination version because that
   tag has no use beyond building the lens. *)
type transformed =
  | Transformed : {
      constants : Tensor.packed Graph_ir.Tensor_id.Map.t;
          (* payloads the passes computed; a folded weight lives only here *)
      derived : (Graph_ir.Tensor_id.t * string list) list;
          (* constants with NO archive path, and the PT2 names they were computed
             from. A folded weight lands here: its bytes exist only in the
             transform state, while the tensors it derives from are still
             nameable. That separation is the point of keeping provenance out of
             the value lattice. *)
      graph : Graph_ir.graph;
      lens : 'b Pt2_native_graph.lens;
      nodes_before : int;
      audits : Pass.Audit_log.t;
          (* one per pass that rewrote something, in execution order, when
             [~verify] was given — empty otherwise. A pass whose sweep matched
             nothing produces an identity step, which has nothing to check.

             BOUNDED: at most [~max_audit_reports] retained reports plus one
             aggregate summary, so a long pipeline over a real model cannot
             retain a report per changed leaf. *)
      composed : Map_verify.Report.t option;
          (* what survived the WHOLE pipeline, over the composed origin-to-final
             map, so its clusters are in this graph's ids and each destination
             edge carries one verdict. That is what a printed node can be
             annotated with; the per-pass audits cannot, since each speaks in
             its own intermediate id space.

             [~verify]'s policy is applied to this report as well as to each
             pass's: composition and terminal packing are the two steps no
             per-pass check covers. *)
    }
      -> transformed

(* Import, rewrite, pack, and build the lens onto the result.

   No payload is bound by default, so a structural pipeline never materialises a
   weight — but [Fold_const] then declines every node, since it refuses a
   constant whose payload is not bound. [~preload:true] binds every captured
   payload a node reads, which is what lets folding hoist a permuted weight; it
   reads the whole archive, so it is not the default.

   [~verify] symbolically checks each pass's mapping as it is applied, against
   the state it came from and WITHOUT payloads for the graph inputs — so it says
   something about every input rather than the one this run happens to use. It
   is a different check from the numeric one the CLI's [--verify] performs, and
   complementary: that one runs the whole model twice and compares outputs,
   which needs real weights and covers only the graph output; this one covers
   every corresponding edge but is budget-capped and so leaves a real model's
   activation-shaped clusters unexamined. See .ai/native_transform_verify.md. *)
val preload :
  Pt2_archive.t ->
  Pt2_native_graph.t ->
  (Tensor.packed Graph_ir.Tensor_id.Map.t, error) Err.t
(** The payloads a NODE READS, loaded from the archive. Only those: an archive
    holds buffers nothing evaluates — resnet18's int64 [num_batches_tracked]
    among them — and loading one would fail on a dtype the engine has no reason
    to support.

    Exposed because [transform]'s [~preload] is not the only caller any more:
    the exporter reaches [transform_lowered] directly, and without this it was
    silently running a fold that declines every node. One implementation, so the
    two cannot come to disagree about which payloads a fold gets. *)

val transform :
  ?preload:bool ->
  ?verify:Map_verify.Policy.t ->
  ?verify_budget:Map_verify.Budget.t ->
  ?verify_probe:int ->
  Pt2_archive.t ->
  passes:Pass.t list ->
  (transformed, error) Err.t

val transform_lowered :
  ?constants:Tensor.packed Graph_ir.Tensor_id.Map.t ->
  ?verify:Map_verify.Policy.t ->
  ?verify_budget:Map_verify.Budget.t ->
  ?verify_probe:int ->
  ?trace:bool ->
  ?max_trace_entries:int ->
  ?max_audit_reports:int ->
  Pt2_native_graph.t ->
  passes:Pass.t list ->
  (transformed, error) Err.t
(** [transform] without the archive: rewrite, verify, pack and build the lens
    over a graph the caller already lowered.

    The archive is a prerequisite of exactly one thing — binding captured
    payloads — so a caller that has none, or wants none, should not have to
    present one. That is the payload-free [model.json] path: it lowers to a
    [Pt2_native_graph.t] and has no weights to read, and under [transform] the
    only way to reach the pipeline was through a type it cannot produce.

    [~constants] seeds the transform state, as a map rather than
    [Rewrite.origin]'s association list: a duplicated id is not a case this
    boundary should have to answer for. [transform] passes the payloads a node
    reads; a structural caller passes none, and [Fold_const] then declines every
    node, exactly as it does under [transform] without [~preload]. *)

type loaded = {
  from_state : int; (* constants a pass computed *)
  from_archive : int; (* constants resolved back to a captured source *)
}

(* Execute a transformed graph. Payloads come from the two sources §10 of
   .ai/native_transform_design.md separates, in that order: the transform state
   first, because a pass-computed constant exists nowhere else, then the archive,
   reached by resolving the destination id back to a captured source through the
   lens. The archive is never reached through PROVENANCE — for a folded weight
   the captured bytes are the pre-fold values and handing them back would be
   corruption, so the lens follows only an [Identical] correspondence and such an
   edge simply has no archive path. *)
val evaluate :
  Pt2_archive.t ->
  transformed ->
  input:Pt2_tensor.t ->
  (Tensor.packed list * loaded, error) Err.t
