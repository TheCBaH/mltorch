(* The native graph IR: a holdable, evaluable, transformable representation of a
   native-op computation. Every edge (tensor) and node carries a stable id unique
   within one global SSA graph; each op names its operands as TYPED FIELDS (no
   positional list, no string keys), with optional operands as [tensor_ref option].
   [Group] is an authoritative structural hierarchy over nodes, never a call.

   Record types live in their own modules (type named [t]), per the project rule
   (CLAUDE.md). See .ai/native_graph_design.md. *)

module Tensor_id = Tensor_id
module Node_id = Graph_common.Node_id

(* Inference source classification. [Input] is supplied by the caller for each
   run; [Constant] is model-bound state. Payload lookup belongs to an importer
   sidecar, not to this generic native IR. *)
module Input = Graph_common.Input

(* A reference to a produced edge. Edges are single-assignment, so a reference is
   just the producing edge's id. *)
type tensor_ref = Tensor_id.t

(* Native operations are ordinary global SSA nodes.  Grouping is represented by
   [Group], not by an operation: it never introduces a call boundary. *)
type op =
  (* Constructors are kept in global alphabetical order so any op is easy to
     locate. Each op carries its own payload record (params +
     operand refs), owned by that op's module; the shared serialise / dataflow /
     pp logic iterates a registry of those modules rather than matching every
     constructor (so adding an op no longer means editing parallel matches here).
     [Eval_op]/[Graph_shape] still match per op, since they need shape/semantics
     context the payload can't carry. *)
  | Add of Pointwise.Add.t
  | Add_scalar of Pointwise.Add_scalar.t
  | Adaptive_avg_pool2d of Pool.AdaptiveAvgPool2d.t
  | Adaptive_max_pool2d of Pool.AdaptiveMaxPool2d.t
  | Adaptive_max_pool2d_with_indices of Pool.AdaptiveMaxPool2dWithIndices.t
  | Amax of Reduce.Amax.t
  | Avg_pool2d of Pool.AvgPool2d.t
  | Batch_norm of Norm.BatchNorm.t
  | Batch_norm_no_stats of Norm.BatchNormNoStats.t
  | Batched_matmul of Matmul.Batched_matmul.t
  | Bmm of Matmul.Bmm.t
  | Clamp of Pointwise.Clamp.t
  | Clone of Pointwise.Clone.t
  (* Variadic: joins its whole operand list along one axis. The only op
     besides [Unbind] whose arity is not fixed by the op itself, but on the
     INPUT side rather than the output — [operands]/[map_operands] already
     generalise to a list, so nothing about them changed to admit it. *)
  | Concat of Concat.Concat.t
  | Conv1d of Conv.Conv1d.t
  | Conv2d of Conv.Conv2d.t
  | Conv2d_padding of Conv.Conv2d_padding.t
  | Convolution of Conv.Convolution.t
  | Div of Pointwise.Div.t
  | Div_scalar of Pointwise.Div_scalar.t
  | Discard of { x : tensor_ref }
    (* A sink: consumes one edge and produces NO output (its [Node.outputs] is
       empty). Used to route a dead op output — e.g. the argmax indices of
       [Max_pool2d_with_indices] — so the op keeps its full ATen arity while the
       edge is explicitly marked unused for a future pruning pass. Like
       [Discard], it is handled inline wherever the [op_registry] is folded. *)
  | Expand of Pointwise.Expand.t
  | Eye of Factory.Eye.t
  | Gelu of Pointwise.Gelu.t
  (* Reshapes [channel] into [groups] equal chunks and normalises each
     (N, group) slice over that chunk plus every axis but N and [channel] --
     not a caller [dims] list the way [Layer_norm]/[Rms_norm] take one, since
     ATen's group_norm has no axis-selection parameter. Its own shape rule
     (channel count must divide by [groups]) and its own reduction (a
     windowed [channel] sum, not a full-extent one), so it is not a
     legalization onto either of them. *)
  | Group_norm of Norm.GroupNorm.t
  | Hardsigmoid of Pointwise.Hardsigmoid.t
  | Hardswish of Pointwise.Hardswish.t
  | Hardtanh of Pointwise.Hardtanh.t
  (* `index.Tensor`'s runtime gather, scoped to a rank-1 live index (see
     .ai/index_tensor_design.md): the gathered axis's position is resolved
     from the VALUE stored in [index], not known at graph-construction time --
     the one op in this engine whose index arithmetic is genuinely
     data-dependent, via [Semantics.load_index]/[Index.Data]. *)
  | Index_tensor of Index_tensor.Index_tensor.t
  | Layer_norm of Norm.LayerNorm.t
  | Leaky_relu of Pointwise.Leaky_relu.t
  | Linear of Linear.Linear.t
  | Max_pool2d of Pool.MaxPool2d.t
  | Max_pool2d_with_indices of Pool.MaxPool2dWithIndices.t
  | Mean of Reduce.Mean.t
  | Mul of Pointwise.Mul.t
  | Mul_scalar of Pointwise.Mul_scalar.t
  | Pad of Pad.Pad.t
  | Permute of Permute.Permute.t
  | Pow of Pointwise.Pow.t
  | Relu of Pointwise.Relu.t
  | Repeat of Repeat.Repeat.t
  | RepeatInterleave of Repeat.RepeatInterleave.t
  | Reshape of Reshape.Reshape.t
  | Rms_norm of Norm.RmsNorm.t
  (* `rsub.Scalar`'s own op: [other - alpha * self], the reverse of
     [sub.Tensor]'s scalar form, which legalizes to [Add_scalar] instead --
     this one needs its own node because it composes a multiply and a
     subtract, two Native ops, on one input. *)
  | Rsub_scalar of Pointwise.Rsub_scalar.t
  | Sdpa of Attention.Sdpa.t
  (* Picks one index along one axis and DROPS it, unlike [Slice] which keeps a
     strided range at unchanged rank. Reuses [Slice]'s [output_shape]/[Compute]
     over a one-wide window rather than naming its own shape/pixel rule, so it
     never decomposes into a [Slice]+[Reshape] pair. *)
  | Select of Split.Select.t
  (* The write-back counterpart of [Select]: [self] with position [index] of
     [axis] replaced by [src] (which has [Select]'s own output shape) and
     every other position carried through from [self] unchanged. A pure
     coordinate branch, no arithmetic -- reuses [Select]'s own axis-drop
     shape rule to check [src] rather than restating it. *)
  | Select_scatter of Split.Select_scatter.t
  | Sigmoid of Pointwise.Sigmoid.t
  | Silu of Pointwise.Silu.t
  (* Softmax over a single axis, keeping the input's full shape -- unlike
     [Amax]/[Mean]/[Vector_norm], which drop or collapse the reduced axes.
     General, not attention-specific: see .ai/matmul_softmax_design.md. *)
  | Softmax of Reduce.Softmax.t
  (* Selects a strided range along one axis and KEEPS it, so unlike [Unbind]
     the rank is unchanged and there is one output. Its bounds are canonical —
     defaulted, normalized and clamped by [Aten_shape.resolve_slice] before the
     payload exists. *)
  | Slice of Split.Slice.t
  (* Divides one axis into contiguous windows of the given SIZES, KEEPING the
     axis in every output — unlike [Unbind], which drops it. Each output is
     what [Slice] would give for that window; [Split_with_sizes] owns its own
     shape rule (sizes sum exactly to the axis's extent) rather than being N
     [Slice] nodes, the same design-goal reasoning [Concat]/[Stack] follow. *)
  | Split_with_sizes of Split.Split_with_sizes.t
  | Sqrt of Pointwise.Sqrt.t
  (* Variadic like [Concat], but inserts a new size-1 axis per operand before
     joining rather than joining along an existing one. Reuses [Concat]'s
     [output_shape] over each operand's unsqueezed shape rather than naming
     its own shape rule, so it never decomposes into N [Reshape]s + [Concat]. *)
  | Stack of Concat.Stack.t
  | Sub of Pointwise.Sub.t
  | Sum of Reduce.Sum.t
  (* [_to_copy.default]'s dtype cast, restricted to the three-way value-domain
     target this repo has corpus evidence for (bool/float/long) -- see
     [Pointwise.To_copy]'s own comment. Shape-preserving, like [Clone], but
     unlike [Clone] the per-pixel VALUE can change. *)
  | To_copy of Pointwise.To_copy.t
  (* The only op whose output COUNT is not fixed by the op: it is the extent at
     the selected axis, so the arity comes from the operand signature.
     [Graph_shape] returns one shape per slice and [Graph_builder.unbind]
     allocates one edge each. See .ai/native_multi_output_design.md §1a. *)
  | Unbind of Split.Unbind.t
  (* Bilinear resize to an explicit output size, either `align_corners`
     value. The per-axis coordinate transform is its own shape/compute
     concern, not a legalization onto any existing op. *)
  | Upsample_bilinear2d of Resize.Bilinear2d.t
  | Upsample_nearest2d of Resize.Nearest2d.t
  | Vector_norm of Reduce.Vector_norm.t
  | Arange of Factory.Arange.t
  | Zeros of Factory.Zeros.t

(* Aliases to [Graph_common], which owns the dialect-agnostic vocabulary so a
   second dialect can reuse it. The ID modules move with the records because
   [Graph_common.Node.t] needs [Node_id.t] — leaving [Node_id] here would be a
   compilation-unit cycle. [Tensor_id] already sets that precedent.

   [Node.t] and [Graph.t] are therefore PARAMETERISED; [node] and [graph] below
   are the monomorphic names, and are what every caller outside this file
   already uses. Field access ([n.Node.outputs]) resolves through the alias
   unchanged. *)
module Node = Graph_common.Node
module Group_id = Graph_common.Group_id
module Group = Graph_common.Group
module Graph = Graph_common.Graph

type node = op Node.t
type graph = op Graph.t

(* Optional, importer-owned annotations for a human-facing graph dump.  The
   generic IR deliberately has no names or foreign metadata; callers such as a
   PT2 importer can render those alongside the stable native ids without
   changing graph identity or serialisation. *)
module Printer : sig
  type t = {
    tensor : Format.formatter -> Tensor_id.t -> unit;
    node : Format.formatter -> Node_id.t -> unit;
  }
end

(* Generic dataflow view — the only generic accessors needed (evaluation reads the
   typed fields directly). [operands] lists an op's input edges; [map_operands]
   rewrites them (for transform/remap). Both are one exhaustiveness-checked match. *)
val operands : op -> tensor_ref list
val map_operands : (tensor_ref -> tensor_ref) -> op -> op
val nodes : graph -> node list

(* Precomputed producer/consumer lookup for the pretty-printer, so a caller
   dumping many nodes from the same graph (e.g. one [pp_node] per evaluated
   node) can build it once instead of paying an O(nodes) rescan per printed
   tensor id on every call. *)
module Index : sig
  type t

  val make : graph -> t
end

(* Deterministic graph dump — used by tests and by bin/native_graph. It
   prints graph inputs, its structural group hierarchy, op operands/parameters,
   and outputs. [pp_with] adds importer-side metadata inline after tensor
   definitions and node ids. *)
val pp_with : printer:Printer.t -> Format.formatter -> graph -> unit
val pp : Format.formatter -> graph -> unit

(* [index], if omitted, is built fresh from [graph]. Pass a precomputed one
   (see [Index.make]) when calling this repeatedly against the same graph —
   e.g. once per evaluated node — to avoid rebuilding it every call. *)
val pp_node :
  ?printer:Printer.t ->
  ?index:Index.t ->
  graph ->
  Format.formatter ->
  node ->
  unit

val pp_op : ?printer:Printer.t -> graph -> Format.formatter -> op -> unit

(* Op printing given only a way to render an operand reference — for callers
   with no graph to resolve against, such as a recipe describing nodes it has not
   spliced yet. [pp_op] is this with the graph's own reference printer. *)
val pp_op_with : pp_ref:tensor_ref Fmt.t -> Format.formatter -> op -> unit
val input_kind : graph -> Tensor_id.t -> Input.kind

val op_name : op -> string
(** The op's constructor name — the SAME string [op_jsont] uses as its case tag,
    so a projection that labels a node and a serialisation that names it cannot
    disagree. Total, [Discard] included. *)

val op_jsont : op Jsont.t
val graph_jsont : graph Jsont.t
