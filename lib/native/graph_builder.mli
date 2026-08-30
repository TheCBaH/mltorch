(* A functional (state-monad) builder for the native graph IR. The state — fresh
   id counters, the accumulated nodes/edges/inputs, and the default element type —
   is threaded purely; nothing is mutated. Output shapes are COMPUTED (via
   [Graph_shape]), never supplied. See .ai/native_graph_design.md. *)

open Graph_ir

type error =
  [ Graph_shape.error | `Expected_single_output_shape of output_count ]

and output_count = { count : int }

val pp_error : Format.formatter -> [< error ] -> unit

type 'a t

val return : 'a -> 'a t
val ( let* ) : 'a t -> ('a -> 'b t) -> 'b t
val ( let+ ) : 'a t -> ('a -> 'b) -> 'b t

(* An external input edge. [?name] is retained as an ignored compatibility
   argument; native graph identity is always the allocated [Tensor_id]. [?fmt]
   defaults to the builder's default element type (F32 unless [build] overrides
   it). *)
val input :
  shape:Vec6.shape ->
  ?name:string ->
  ?fmt:Payload.packed_fmt ->
  ?quant:Quant.t ->
  unit ->
  Tensor_id.t t

(* A captured, read-only inference tensor. Its model/archive payload association
   belongs to the importer sidecar, not to the generic native graph. *)
val constant :
  shape:Vec6.shape ->
  ?name:string ->
  ?fmt:Payload.packed_fmt ->
  ?quant:Quant.t ->
  unit ->
  Tensor_id.t t

(* Typed-operand op constructors: each appends a node and returns its fresh output
   edge. [?name] is an ignored compatibility argument. Omitting an optional
   operand ([?bias], [?weight]) records [None] in the IR; the evaluator fills the
   identity (zeros bias / ones weight). *)
(* Op constructors in global alphabetical order (see graph_ir.mli). *)
val add : ?name:string -> tensor_ref -> tensor_ref -> Tensor_id.t t

(* The scalar-parameter twins of [add]/[div], for a compile-time scalar the
   exporter serialised into a Tensor slot. [scalar] is narrowed to its
   f32-canonical value here, so callers need not. *)
val add_scalar : ?name:string -> float -> tensor_ref -> Tensor_id.t t

val adaptive_avg_pool2d :
  ?name:string -> Pool.AdaptiveAvgPool2d.params -> tensor_ref -> Tensor_id.t t

val amax : ?name:string -> Reduce.Amax.params -> tensor_ref -> Tensor_id.t t

val avg_pool2d :
  ?name:string -> Pool.AvgPool2d.params -> tensor_ref -> Tensor_id.t t

val batch_norm :
  ?name:string ->
  Norm.BatchNorm.params ->
  x:tensor_ref ->
  ?weight:tensor_ref ->
  ?bias:tensor_ref ->
  running_mean:tensor_ref ->
  running_var:tensor_ref ->
  unit ->
  Tensor_id.t t

val batched_matmul : ?name:string -> tensor_ref -> tensor_ref -> Tensor_id.t t
val bmm : ?name:string -> tensor_ref -> tensor_ref -> Tensor_id.t t

(* Errors with [`Clamp No_bounds] if neither bound is given, as ATen does. *)
val clamp :
  ?name:string -> Pointwise.Clamp.params -> tensor_ref -> Tensor_id.t t

val clone : ?name:string -> tensor_ref -> Tensor_id.t t

val concat :
  ?name:string -> Concat.Concat.params -> tensor_ref list -> Tensor_id.t t

val conv2d :
  ?name:string ->
  Conv.Conv2d.params ->
  x:tensor_ref ->
  weight:tensor_ref ->
  ?bias:tensor_ref ->
  unit ->
  Tensor_id.t t

val conv2d_padding :
  ?name:string ->
  Conv.Conv2d_padding.params ->
  x:tensor_ref ->
  weight:tensor_ref ->
  ?bias:tensor_ref ->
  unit ->
  Tensor_id.t t

val convolution :
  ?name:string ->
  Conv.Convolution.params ->
  x:tensor_ref ->
  weight:tensor_ref ->
  ?bias:tensor_ref ->
  unit ->
  Tensor_id.t t

val div : ?name:string -> tensor_ref -> tensor_ref -> Tensor_id.t t
val div_scalar : ?name:string -> float -> tensor_ref -> Tensor_id.t t

(* Route a dead edge into a [Discard] sink node (no output). Used to keep a
   multi-output op's full arity while marking an unused result for later pruning. *)
val discard : tensor_ref -> unit t

(* Broadcasts [x] to [params.size]. [Graph_shape] rejects a target that is not
   broadcast-compatible with [x]'s own shape (see [Pointwise.Expand.output_shape]). *)
val expand :
  ?name:string -> Pointwise.Expand.params -> tensor_ref -> Tensor_id.t t

val gelu :
  ?name:string -> Pointwise.Gelu.approximate -> tensor_ref -> Tensor_id.t t

(* Both affine operands are independently optional, the same convention
   [layer_norm]/[rms_norm] follow. *)
val group_norm :
  ?name:string ->
  Norm.GroupNorm.params ->
  x:tensor_ref ->
  ?weight:tensor_ref ->
  ?bias:tensor_ref ->
  unit ->
  Tensor_id.t t

val hardsigmoid : ?name:string -> tensor_ref -> Tensor_id.t t
val hardswish : ?name:string -> tensor_ref -> Tensor_id.t t

val hardtanh :
  ?name:string -> Pointwise.Hardtanh.params -> tensor_ref -> Tensor_id.t t

val index_tensor :
  ?name:string ->
  Index_tensor.Index_tensor.params ->
  self:tensor_ref ->
  index:tensor_ref ->
  Tensor_id.t t

val layer_norm :
  ?name:string ->
  Norm.LayerNorm.params ->
  x:tensor_ref ->
  ?weight:tensor_ref ->
  ?bias:tensor_ref ->
  unit ->
  Tensor_id.t t
(** Both affine operands are independently optional, exactly as ATen's
    [Tensor? weight, Tensor? bias] are. Absent means the identity (scale 1,
    shift 0) and is materialised by [Eval_op], not by the caller: a graph built
    with an explicit ones tensor is a DIFFERENT graph, and the two importers
    have to agree on which one they build. *)

val linear :
  ?name:string ->
  Linear.Linear.params ->
  x:tensor_ref ->
  weight:tensor_ref ->
  ?bias:tensor_ref ->
  unit ->
  Tensor_id.t t

val max_pool2d :
  ?name:string -> Pool.MaxPool2d.params -> tensor_ref -> Tensor_id.t t

(* max_pool2d_with_indices returns two edges: (values, indices). *)
val max_pool2d_with_indices :
  ?name:string ->
  Pool.MaxPool2dWithIndices.params ->
  tensor_ref ->
  (Tensor_id.t * Tensor_id.t) t

val mean : ?name:string -> Reduce.Mean.params -> tensor_ref -> Tensor_id.t t
val mul : ?name:string -> tensor_ref -> tensor_ref -> Tensor_id.t t
val mul_scalar : ?name:string -> float -> tensor_ref -> Tensor_id.t t

val pad : ?name:string -> Pad.Pad.params -> tensor_ref -> Tensor_id.t t
(** Narrows a [Constant] fill to its f32-canonical value, as [add_scalar] does
    for its scalar. Negative pad entries are accepted (they crop); the shape
    rule refuses one that empties an axis. *)

val permute :
  ?name:string -> Permute.Permute.perm -> tensor_ref -> Tensor_id.t t

val pow : ?name:string -> float -> tensor_ref -> Tensor_id.t t
val relu : ?name:string -> tensor_ref -> Tensor_id.t t

val reshape :
  ?name:string -> Reshape.Reshape.params -> tensor_ref -> Tensor_id.t t

val rms_norm :
  ?name:string ->
  Norm.RmsNorm.params ->
  x:tensor_ref ->
  ?weight:tensor_ref ->
  unit ->
  Tensor_id.t t

val sdpa :
  ?name:string ->
  Attention.Sdpa.params ->
  query:tensor_ref ->
  key:tensor_ref ->
  value:tensor_ref ->
  ?mask:tensor_ref ->
  unit ->
  Tensor_id.t t

val select : ?name:string -> Split.Select.params -> tensor_ref -> Tensor_id.t t
(** Picks one index along one axis and drops it. Takes a CANONICAL,
    already-in-range index — a caller holding ATen's spelling gets one from
    {!Aten_shape.resolve_index}. *)

val sigmoid : ?name:string -> tensor_ref -> Tensor_id.t t
val silu : ?name:string -> tensor_ref -> Tensor_id.t t

val softmax :
  ?name:string -> Reduce.Softmax.params -> tensor_ref -> Tensor_id.t t
(** Softmax over [params.axis], keeping the input's full shape. *)

val slice : ?name:string -> Split.Slice.params -> tensor_ref -> Tensor_id.t t
(** Takes CANONICAL bounds — non-negative, ordered, within the axis. A caller
    holding ATen's spelling gets them from {!Aten_shape.resolve_slice}; a caller
    that does not is refused by the shape rule rather than silently reading out
    of bounds. *)

val sqrt : ?name:string -> tensor_ref -> Tensor_id.t t

(* Divides one axis into contiguous windows of [params.sizes], keeping the
   axis in every output — the same "result length is not fixed by the op"
   shape [unbind] has, and for the same reason its slices retain the input
   format and quantization metadata. The sizes must be positive and sum
   exactly to the axis's extent, checked by the shape rule rather than
   assumed. *)
val split_with_sizes :
  ?name:string ->
  Split.Split_with_sizes.params ->
  tensor_ref ->
  Tensor_id.t list t

val stack :
  ?name:string -> Concat.Stack.params -> tensor_ref list -> Tensor_id.t t
(** Inserts a new size-1 axis per operand at [params.axis], then joins them —
    the single-node counterpart of an ATen [stack], which reshapes every operand
    before concatenating them. *)

val sub : ?name:string -> tensor_ref -> tensor_ref -> Tensor_id.t t
val sum : ?name:string -> Reduce.Sum.params -> tensor_ref -> Tensor_id.t t

(* Unbind returns EVERY slice, in ordinal order — the only builder whose result
   length is not fixed by the op. The count is derived from the input signature
   ([extent] at the selected axis), never passed in, so a caller holding a list
   of serialized output names can check it rather than zip against it. The
   list is never empty: Native extents are >= 1. Unlike arithmetic outputs, its
   slices retain the input format and quantization metadata. *)
val unbind :
  ?name:string -> Split.Unbind.params -> tensor_ref -> Tensor_id.t list t

val upsample_bilinear2d :
  ?name:string -> Resize.Bilinear2d.params -> tensor_ref -> Tensor_id.t t
(** Bilinear resize to an explicit output size, either [align_corners] value. *)

val upsample_nearest2d :
  ?name:string -> Resize.Nearest2d.params -> tensor_ref -> Tensor_id.t t
(** Nearest-neighbor resize to an explicit output size. *)

val vector_norm :
  ?name:string -> Reduce.Vector_norm.params -> tensor_ref -> Tensor_id.t t

(* Structurally group nodes emitted by [body].  The group shares the enclosing
   graph's global SSA namespace and has no inputs, outputs, or call semantics. *)
val group : ?label:string -> 'a t -> 'a t

(* Run a builder computation from the empty state and finalise into a graph;
   [outputs] selects the graph outputs from the computation's result. [?dtype] is
   the default element type for [input] (default F32). [name] is ignored. *)
val build :
  ?dtype:Payload.packed_fmt ->
  name:string ->
  outputs:('a -> Tensor_id.t list) ->
  'a t ->
  (graph, error) Err.t
