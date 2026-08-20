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
  | `Int_opt
  | `Bool
  | `Float
  | `Scalar
  | `Optional_scalar
  | `String
  | `Tensor_or_scalar ]
(** What an argument had to be — exactly the set the decode helpers accept. *)

(** A rank an arm requires against the rank the model declared. One row rather
    than one per arity: [conv2d] wants four, [linear] two, and a second variant
    per number would be the same fault spelled twice. Own module for the
    record-namespace convention. *)
module Expected_rank : sig
  type t = { expected : int; got : int }
end

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
  | `Expected_rank of Expected_rank.t
  | `Over_max_extent of int64
    (** A derived extent past [Kernel.Limits.Hard.extent]. Carried as [int64]
        because that is the only width it is guaranteed to fit: the value is a
        product of model-supplied factors, and js_of_ocaml's [int] is 32 bits,
        so reporting it as an [int] would print the wrapped number that made it
        a defect. *) ]

type metadata_role =
  [ `Tensor
  | `Convolution_weight
  | `Conv2d_weight
    (** Its own role, not shared with [`Convolution_weight]: the two overloads
        have separate arms and a shared label would leave the row unable to say
        which one failed. *)
  | `Conv2d_padding_weight
    (** Same reasoning again, and not an over-refinement: [conv2d.padding] reads
        the weight for its RANK only, so a report naming the [conv2d] role would
        send a reader looking for a channel check that arm does not perform. *)
  | `Linear_weight
  | `Conv2d_bias
  | `Conv2d_padding_bias
  | `Convolution_bias
  | `Linear_bias
  | `Rms_norm_weight
    (** The OPTIONAL operands, whose metadata is read for its declared RANK.
        [shape_of_sizes] right-aligns a size list into the six-axis frame, so
        [C] and [1,C] land on identical extents and the shared [Graph_shape]
        check cannot separate them — while ATen refuses a bias that is not 1-D.
        The rank exists only before that conversion, so each importer checks its
        own. *)
  | `Rms_norm_input
    (** rms_norm reads the INPUT's metadata, not a weight's: [normalized_shape]
        is checked against the input's trailing extents, which is the only place
        those extents can come from. *)
  | `Mean_input
  | `Permute_input
  | `Transpose_input
    (** Its own role, not shared with [`Permute_input]: [transpose.int] and
        [permute.default] have separate arms, and a shared label would leave the
        row unable to say which one failed -- the same reasoning as
        [`Conv2d_weight] vs [`Convolution_weight] above. *)
  | `Pad_input
  | `Slice_input
  | `Unbind_input
  | `Addmm_weight ]
(** Why the missing [tensor_values] entry was wanted. *)

type hw_param =
  [ `Stride | `Padding | `Output_padding | `Dilation | `Kernel_size ]
(** The parameters read as an [h, w] pair. Not the same set as
    {!Op_config.Bad.param}, which adds [`Groups]: a group count is a lone int
    and so can be a bad VALUE but never a bad arity. *)

type config_param = Op_config.Bad.param

type config_fault = Op_config.Bad.fault
(** Shared with {!Op_bridge} through [Op_config.Bad], not restated here. The two
    importers have to reject the same values, and two vocabularies for "this
    stride is zero" is one drift away from two contracts. *)

type unsupported_option =
  [ `Alpha of float | `Memory_format | `Dilation of int list | `Ceil_mode ]
(** Options this lowering rejects rather than silently drops: a non-unit [alpha]
    would compute the wrong thing, and a [memory_format] asks for a layout
    change the native IR cannot express.

    The two pooling options are the same kind of refusal for a different reason:
    [Pool.MaxPool2d.params] has no field for either, so carrying them was never
    an option and dropping them silently computed a different op under the right
    name. [`Dilation] keeps the whole offered list rather than a normalized
    pair, because the rejection happens before the arity check that would give
    it one. *)

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

(** A [SymInt] argument that arrived as a NAME rather than a value. Its own row
    rather than a {!Wrong_arg_kind}: the spelling and the kind are both right,
    and what is missing is a binding for the symbol — the same distinction
    {!Bad_dimension}'s [`Symbolic] fault draws for tensor metadata, applied to
    an argument. *)
module Unresolved_sym_arg : sig
  type t = { op : string; arg : string; symbol : string }
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

module Bad_config = Op_config.Bad

(** How many entries [normalized_shape] had against the rank it has to fit
    inside. Covers both ends -- an empty list and one longer than the rank -- as
    one fault, because both are the same question answered with the same two
    numbers. *)
module Normalized_rank : sig
  type t = { rank : int; got : int }
end

(** The input's trailing extents against the ones [normalized_shape] declared.
    Both lists, not a first differing index: a reader needs to see which axes
    were meant. *)
module Normalized_shape : sig
  type t = { expected : int list; got : int list }
end

module Unsupported_option : sig
  type t = { op : string; option : unsupported_option }
end

module Output_arity : sig
  type t = { op : string; serialized : int; derived : int }
end

(** A [view.default]/[_unsafe_view.default] target this module cannot satisfy:
    either [Aten_shape]'s [-1]-inference convention (multiple inferred dims,
    non-divisible, or a wrong total count) or the source's element count itself
    running over [Kernel.Limits.Hard.numel]. One row rather than two, because
    both are "this view request is not satisfiable" and the serialized [size] is
    the operand a reader needs either way. Not [`Bad_dimension]'s
    [`Over_max_extent]: a whole-tensor count and a per-axis extent are different
    quantities. *)
module Bad_view : sig
  type t = {
    size : int list;
    fault :
      [ `Aten_shape of Aten_shape.error
      | `Numel_over_limit of Vec6.Numel_bound.t ];
  }
end

(** A [slice.Tensor] request {!Aten_shape.resolve_slice} refuses — today only a
    non-positive step, which ATen refuses too. Carries the SERIALIZED spelling,
    optionals and all, because that is what a reader has to change; the
    canonical bounds do not exist yet at the point this is raised, and an empty
    RESULT is a different fault entirely, reported by {!Shape_error.Slice} once
    the extent is known. *)
module Bad_slice : sig
  type t = {
    start : int option;
    stop : int option;
    step : int;
    fault : [ `Aten_shape of Aten_shape.error ];
  }
end

type malformed =
  [ `Missing_arg of Missing_arg.t
  | `Wrong_arg_kind of Wrong_arg_kind.t
  | `Unresolved_sym_arg of Unresolved_sym_arg.t
  | `Missing_metadata of Missing_metadata.t
  | `Bad_dimension of Bad_dimension.t
  | `Axis_out_of_range of Axis_out_of_range.t
  | `Bad_arity of Bad_arity.t
  | `Bad_config of Bad_config.t
    (** An op-configuration value the engine's guarded types have no form for.
        Its own row rather than a [`Bad_dimension] variant: a stride is not a
        dimension, and the two are validated by different constructors against
        different rules. Before it existed these reached
        [Op_config.Pos.of_int]/[Nonneg.of_int] directly and left [lower] as an
        uncaught [Invalid_argument] — past the boundary that is supposed to
        classify them. *)
  | `Normalized_rank of Normalized_rank.t
  | `Normalized_shape of Normalized_shape.t
    (** The check [Op_bridge] does not do. It reads [normalized_shape]'s LENGTH
        and nothing else, so a shape naming the wrong extents normalizes over
        the wrong axes and returns a plausible wrong answer rather than an
        error. *)
  | `Unsupported_padding_mode of string
  | `Bad_pad_list of Pad.Pad.Bad_pad_list.t
    (** The mode [conv2d.padding] offered. A string because it is a third-party
        value out of the export rather than a case this module declined to
        classify — [Conv.Conv2d_padding.padding] has exactly two constructors
        and the tag names which argument produced the outlier. Its own row
        rather than an [`Unsupported_option]: that record means "recognised and
        refused", while an unknown mode is not recognised at all. *)
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
  | `Output_not_evaluated of Graph_ir.Tensor_id.t
  | `Bad_view of Bad_view.t
  | `Bad_slice of Bad_slice.t ]
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
