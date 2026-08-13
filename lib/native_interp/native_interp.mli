(* Pure lowering of the static tensor subset of an ExportedProgram. *)

(** {1 Error payloads}

    What a malformed graph actually was. These used to be one
    [`Malformed_graph of string] carrying a [Printf.ksprintf] sentence from 32
    sites, so a caller could read the failure but never branch on it. *)

type arg_kind =
  [ `Tensor
  | `Optional_tensor
  | `Int_list
  | `Int
  | `Bool
  | `Float
  | `Scalar
  | `Optional_scalar
  | `Tensor_or_scalar ]
(** What an argument had to be — exactly the set the decode helpers accept. *)

type dim_fault =
  [ `Negative of int
  | `Zero
    (** Not a sub-case of [`Negative]: the engine forbids an empty extent by
        construction ([Dim.extent] is >= 1), so a declared 0 is a shape this
        dialect has no form for rather than a nonsensical number — and it
        arrives from real models (ATen's unbind of a zero-length dim), where a
        negative size never does. *)
  | `Symbolic
  | `Rank_over_six
  | `Expected_rank_four of int  (** the rank actually offered *) ]

type metadata_role =
  [ `Tensor
  | `Convolution_weight
  | `Mean_input
  | `Permute_input
  | `Unbind_input
  | `Addmm_weight ]
(** Why the missing [tensor_values] entry was wanted. *)

type hw_param = [ `Stride | `Padding | `Dilation | `Kernel_size ]
(** The four parameters read as an [h, w] pair. *)

type unsupported_option = [ `Alpha of float | `Memory_format ]
(** Options this lowering rejects rather than silently drops: a non-unit [alpha]
    would compute the wrong thing, and a [memory_format] asks for a layout
    change the native IR cannot express. *)

type unsupported_input = [ `Non_tensor | `Not_exactly_one_user_input of int ]
(** Two different rejections, not one with a message. Both recoverable — see
    [Me_classify.lowering] — and only the second has a figure to report. *)

(** Own modules, per the record-namespace convention: three of these carry an
    [op] field and two an [arg]. *)

module Missing_arg : sig
  type t = { op : string; arg : string }
end

module Wrong_arg_kind : sig
  type t = { op : string; arg : string; expected : arg_kind }
end

module Bad_dimension : sig
  type t = { tensor : string; fault : dim_fault }
end

module Missing_metadata : sig
  type t = { ssa : string; role : metadata_role }
end

module Axis_out_of_range : sig
  type t = { axis : int; rank : int }
end

module Bad_arity : sig
  type t = { param : hw_param; got : int }
end

module Unsupported_option : sig
  type t = { op : string; option : unsupported_option }
end

module Output_arity : sig
  type t = { op : string; serialized : int; derived : int }
end

type malformed =
  [ `Missing_arg of Missing_arg.t
  | `Wrong_arg_kind of Wrong_arg_kind.t
  | `Missing_metadata of Missing_metadata.t
  | `Bad_dimension of Bad_dimension.t
  | `Axis_out_of_range of Axis_out_of_range.t
  | `Bad_arity of Bad_arity.t
  | `Unsupported_option of Unsupported_option.t
  | `Output_arity of Output_arity.t
    (** A `Tensor[]`-returning node's arity is model data on one side (the names
        in its single [Argument.Tensors] output) and derived from the operand's
        extent on the other. Disagreement is a malformed graph, checked before
        [add_env], whose [Invalid_argument] stays an invariant about this module
        rather than a report about the graph. *)
  | `Non_tensor_node_output of string  (** the node's target *)
  | `Non_tensor_graph_output
  | `Undefined_ssa of string
  | `Output_not_evaluated of Graph_ir.Tensor_id.t ]
(** A graph the decoder accepted and this lowering cannot read. FLAT-INCLUDED in
    {!error}: it is this module's own failure domain, not a crossed seam. *)

module Rank_mismatch : sig
  type t = { sizes : int; strides : int }
end

module Storage_range : sig
  type t = { lo : int64; hi : int64; data_bytes : int }
end

type tensor_bridge =
  [ malformed
    (* [shape_of_sizes] is written for graph metadata and the bridge reuses it,
       so its rows arrive here re-labelled rather than flattened. *)
  | `Rank_mismatch of Rank_mismatch.t
  | `Storage_index_overflow
  | `Storage_out_of_range of Storage_range.t
  | `Materialize_failed of string
    (** [Invalid_argument]'s own message — a third-party payload, named for its
        source rather than left to read as a case declined to classify *)
  | `Unsupported_dtype of Pt2_dtype.t
  | `Archive of Pt2_archive.error
    (** a real seam, so the whole row crosses it; this was
        [Format.asprintf "%a"] of the same value *) ]
(** Loading a captured tensor — a different job from reading graph metadata. *)

type error =
  [ `Unsupported_input of unsupported_input
  | `Unsupported_operator of string  (** the target *)
  | `Output_count_over_limit of Shape_error.Output_count.t
    (** A RESOURCE rejection, deliberately outside {!malformed}: a perfectly
        well-formed graph can still ask for more outputs than the engine will
        build. The distinction is load-bearing at the Model Explorer boundary,
        where every {!malformed} row is [Fatal] ("our bug") and this one is
        [Unavailable Over_limit] ("your model is too big"). [Shape_error]'s row
        is reused rather than restated, so this spelling and the nested
        [`Build (`Output_count_over_limit _)] carry the same payload. *)
  | malformed
  | `Tensor_bridge of tensor_bridge
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
val pp_malformed : Format.formatter -> [< malformed ] -> unit
val pp_tensor_bridge : Format.formatter -> [< tensor_bridge ] -> unit

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
