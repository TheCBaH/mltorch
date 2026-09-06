type perm_side = [ `Input | `Output ]

module Broadcast : sig
  type t = { axis : Axis.t; lhs : Dim.extent Dim.t; rhs : Dim.extent Dim.t }

  val pp : Format.formatter -> t -> unit
end

module Window : sig
  type t = {
    out : int64;
        (** [int64], not [int]: a SUM of model-supplied extents minus a PRODUCT
            of two more, and js_of_ocaml's [int] is 32 bits — reporting it as an
            [int] would print the wrapped number that made it a defect. *)
    in_extent : Dim.extent Dim.t;
    kernel : Dim.extent Dim.t;
    stride : Op_config.Pos.t;
    pad_before : Op_config.Nonneg.t;
    pad_after : Op_config.Nonneg.t;
    dilation : Op_config.Pos.t;
  }

  val pp : Format.formatter -> t -> unit
end

(** An output extent, or the effective kernel it is computed from, past the
    engine's per-axis ceiling. Distinct from {!Window}, which is the same
    arithmetic coming out too SMALL: a window that shrinks an axis to nothing is
    an ordinary configuration error, while one that exceeds the ceiling is a
    value the engine has no representation for at all. *)
module Window_over_limit : sig
  type quantity =
    [ `Dilation
    | `Effective_kernel
    | `In_channels
    | `Input_extent
    | `Kernel
    | `Output_extent
    | `Padding
    | `Stride ]
  (** Which quantity ran past the ceiling. A closed set rather than a string:
      each is a specific field or a specific derived product, and a reader has
      to know which. [`In_channels] is the conv weight's per-group extent times
      its group count — named for what it IS, not for the extent it is on its
      way to becoming. *)

  type t = { what : quantity; value : int64; limit : int64 }

  val pp : Format.formatter -> t -> unit
end

(** Adaptive-pooling's [input_extent * output_size] aggregate exceeded the
    JS-safe index ceiling. The product is checked before index arithmetic. *)
module Adaptive_pool : sig
  type t = {
    axis : Axis.t;
    input_extent : Dim.extent Dim.t;
    output_size : Op_config.Pos.t;
    aggregate : int64;
    limit : int64;
  }

  val pp : Format.formatter -> t -> unit
end

(** An OPTIONAL operand whose shape disagrees with the one the op requires. One
    row rather than one per op: the fault is identical in every case and the
    [operand] tag says which slot. Every [output_shape] takes the REQUIRED
    operands only, so before this an optional one reached evaluation unchecked
    and raised from [Tensor.read]'s bounds check partway through the result. *)
module Operand_shape : sig
  type t = {
    operand :
      [ `Bias
      | `Batch_norm_no_stats_affine
      | `Group_norm_bias
      | `Group_norm_weight
      | `Layer_norm_bias
      | `Layer_norm_weight
      | `Lstm_bias
      | `Lstm_input
      | `Lstm_state
      | `Lstm_weight_hh
      | `Lstm_weight_ih
      | `Rms_norm_weight ];
    expected : Vec6.shape;
    actual : Vec6.shape;
  }

  val pp : Format.formatter -> t -> unit
end

module Clamp : sig
  (* [clamp] with neither bound is rejected, as ATen's meta function rejects it
     ("At least one of 'min' or 'max' must not be None"). Without this the op
     would silently degrade to the identity. *)
  type error = No_bounds

  val pp_error : Format.formatter -> error -> unit
end

module Linear : sig
  type channels_mismatch = {
    actual : Dim.extent Dim.t;
    expected : Dim.extent Dim.t;
  }

  type error =
    | Input_channels_mismatch of channels_mismatch
    | Weight_channels_mismatch of channels_mismatch

  val pp_error : Format.formatter -> error -> unit
end

module Bmm : sig
  type dims_mismatch = { lhs : Dim.extent Dim.t; rhs : Dim.extent Dim.t }

  type error =
    | Batch_mismatch of dims_mismatch
    | Contract_mismatch of dims_mismatch

  val pp_error : Format.formatter -> error -> unit
end

module Batched_matmul : sig
  type dims_mismatch = {
    axis : Axis.t;
    lhs : Dim.extent Dim.t;
    rhs : Dim.extent Dim.t;
  }

  type error =
    | Batch_mismatch of dims_mismatch
    | Contract_mismatch of { lhs : Dim.extent Dim.t; rhs : Dim.extent Dim.t }

  val pp_error : Format.formatter -> error -> unit
end

module Permute : sig
  type t = { side : perm_side; axis : Axis.t; count : int }

  val pp : Format.formatter -> t -> unit
end

(* `aten.cat.default`'s two possible faults: no tensors at all, or a
   non-concatenated axis that disagrees between two operands. The
   concatenated axis's own overflow reuses [Window_over_limit]'s
   [`Output_extent] row instead of a third fault here -- it IS the output
   extent on that axis. *)
module Concat : sig
  type mismatch = {
    axis : Axis.t;
    first : Dim.extent Dim.t;
    other : Dim.extent Dim.t;
  }

  type t = Axis_mismatch of mismatch | Empty

  val pp : Format.formatter -> t -> unit
end

(* `aten.split_with_sizes.default`'s own fault: a size that is not positive
   (Native tensors have no empty extent, the same rule [Slice]'s `Empty`
   enforces), or a sizes list that does not sum EXACTLY to the split axis's
   extent -- ATen's own contract, checked explicitly here because the sizes
   are untrusted model data rather than assumed to already satisfy it. The
   output COUNT's own overflow reuses [Output_count]'s row, the same one
   [Unbind]'s derived count uses, rather than a third fault here. *)
module Split_with_sizes : sig
  type fault =
    | Non_positive_size of { index : int; size : int }
    | Size_mismatch of { total : int64 }

  type t = { axis : Axis.t; in_extent : Dim.extent Dim.t; fault : fault }

  val pp : Format.formatter -> t -> unit
end

(* [Reshape.output_shape]'s numel-preservation precondition, violated: a target
   whose element count disagrees with the source's. *)
module Reshape : sig
  type t = { source : Vec6.shape; target : Vec6.shape }

  val pp : Format.formatter -> t -> unit
end

(** [select_scatter]'s [src] does not have exactly the shape [select] itself
    would produce at [axis]/[index]; [expected] is that derived shape. *)
module Select_scatter : sig
  type t = {
    axis : Axis.t;
    index : int;
    expected : Vec6.shape;
    actual : Vec6.shape;
  }

  val pp : Format.formatter -> t -> unit
end

(** A slice with no Native result. Two faults, and they are different in kind:

    - [`Empty] — the canonical bounds select nothing. ATen returns a size-0
      tensor; the engine has no empty extent ([Dim.extent] is >= 1), so this is
      the typed "legal upstream, outside the Native domain" boundary rather than
      a configuration error. Same boundary as {!Pad}'s [`Empty].
    - [`Out_of_range] — the stored bounds are not
      [0 <= start <= stop <= extent]. Not reachable from either importer, which
      build their bounds with {!Aten_shape.resolve_slice} and so satisfy it by
      construction; this guards {!Graph_builder} and JSON decoding, where
      nothing else would, and it is the check that keeps [Compute]'s read in
      bounds — [clamp_low] bounds the source coordinate below and nothing bounds
      it above.

    NOT folded into {!Window}: that payload names a kernel and a dilation, which
    a slice does not have, and one row for both faults would make them
    indistinguishable to a reader.

    [out] is [int64] for {!Window}'s reason — a difference of model-supplied
    bounds, divided, and js_of_ocaml's [int] is 32 bits. The bounds are the
    CANONICAL ones (defaults applied, negatives normalized, clamped), because
    those are what produced [out]; the serialized spelling belongs to the
    importer's own row. *)
module Slice : sig
  type fault = [ `Empty | `Out_of_range ]

  type t = {
    axis : Axis.t;
    in_extent : Dim.extent Dim.t;
    start : int;
    stop : int;
    step : Op_config.Pos.t;
    out : int64;
    fault : fault;
  }

  val pp : Format.formatter -> t -> unit
end

(** A pad configuration with no Native result. Two faults, because they are
    different rules and a reader has to know which:

    - [`Empty] — the padded extent is below 1. Reachable only through a NEGATIVE
      pad, which crops: [aten.pad] permits it and the engine cannot represent
      the result. Same boundary as {!Slice}'s, on the other structural op.
    - [`Reflect_width] — reflect mode with a pad at least as wide as the axis it
      mirrors, which PyTorch also rejects. Only the positive side can violate
      it: a negative pad shrinks the read range, so the mirror never fires.
    - [`Duplicate_axis] — two entries for one axis. Not reachable from either
      importer (both build from a rank-indexed list), so this guards
      {!Graph_builder} and JSON decoding. Which entry was meant is unknowable,
      which is why it is a refusal and not a last-one-wins.

    [before]/[after] are signed and [int64]: signed because cropping is
    supported, [int64] because their sum with the extent is a model-supplied
    aggregate. *)
module Pad : sig
  type fault = [ `Duplicate_axis | `Empty | `Reflect_width ]

  type t = {
    axis : Axis.t;
    in_extent : Dim.extent Dim.t;
    before : int64;
    after : int64;
    out : int64;
    fault : fault;
  }

  val pp : Format.formatter -> t -> unit
end

(** `aten.unfold.default`'s own two faults, both about which axis a window can
    land on: [Reserved_axis] -- the caller named [C], but [C] is reserved for
    the window OFFSET (Unfold always writes it there, per its own doc comment)
    -- and [Rank_exceeded] -- the input's own [N] axis is not unit, so the
    six-axis frame has no room left for the window-count axis this op
    introduces. *)
module Unfold : sig
  type fault = Reserved_axis | Rank_exceeded
  type t = { axis : Axis.t; fault : fault }

  val pp : Format.formatter -> t -> unit
end

module Convolution : sig
  type channels_divisibility = { channels : int; groups : int }

  type weight_channels_mismatch = {
    weight_in_per_group : int;
    expected_in_per_group : int;
  }

  type weight_kernel_mismatch = {
    axis : Axis.t;
    weight_extent : Dim.extent Dim.t;
    kernel_extent : Dim.extent Dim.t;
  }

  type input_channels_mismatch = {
    input_channels : Dim.extent Dim.t;
    expected_in_channels : Dim.extent Dim.t;
  }

  type output_padding = { h : Op_config.Nonneg.t; w : Op_config.Nonneg.t }

  type transposed_weight_input_mismatch = {
    weight_input_channels : Dim.extent Dim.t;
    input_channels : Dim.extent Dim.t;
  }

  type transposed_window_output = {
    out : int;
    in_extent : Dim.extent Dim.t;
    kernel : Dim.extent Dim.t;
    stride : Op_config.Pos.t;
    pad : Op_config.Nonneg.t;
    dilation : Op_config.Pos.t;
    output_padding : Op_config.Nonneg.t;
  }

  type error =
    | In_channels_not_divisible_by_groups of channels_divisibility
    | Input_channels_mismatch of input_channels_mismatch
    | Out_channels_not_divisible_by_groups of channels_divisibility
    | Output_padding_nonzero of output_padding
    | Same_padding_requires_stride_one of { stride : Op_config.Pos.t }
    | Transposed_input_channels_not_divisible_by_groups of channels_divisibility
    | Transposed_not_supported
    | Transposed_output_non_positive of transposed_window_output
    | Transposed_weight_input_mismatch of transposed_weight_input_mismatch
    | Weight_channels_mismatch of weight_channels_mismatch
    | Weight_kernel_mismatch of weight_kernel_mismatch

  val pp_error : Format.formatter -> error -> unit
end

(* A per-node output cardinality that model data can choose: [Unbind]'s output
   count is the extent at the selected axis, so a declared size decides how many
   edges shape inference is asked to produce. The ceiling is
   [Kernel.Limits.Hard.outputs] and the rule is EXCLUSIVE, matching
   [Kernel.Limits.create]'s own [v >= hard] test — reusing that constant with a
   different boundary would be worse than picking a second number.

   [observed] exists because the two producers know different things. Shape
   inference holds the extent, so it reports [Exact]. A bounded list traversal
   stops AT the exclusive limit and never learns the real length, so it reports
   [At_least] — recording that as exact would claim evidence nobody gathered. *)
module Output_count : sig
  type observed = At_least of int | Exact of int
  type t = { limit : int; observed : observed }

  val pp : Format.formatter -> t -> unit
end

(** [Sdpa]'s own row: the extent-agreement checks common to every operand pair,
    the mask's broadcast-or-equal rule against {!Attention.Sdpa.score_shape},
    and the sixth-factor total-work bound (op8-impl.md F12) that neither the
    score-count nor the output-numel [`Numel_over_limit] bound implies. *)
module Sdpa : sig
  type check = [ `Key_value | `Query_key | `Query_value ]

  type dims_mismatch = {
    axis : Axis.t;
    check : check;
    lhs : Dim.extent Dim.t;
    rhs : Dim.extent Dim.t;
  }

  type mask_mismatch = {
    axis : Axis.t;
    mask : Dim.extent Dim.t;
    score : Dim.extent Dim.t;
  }

  type work_factor =
    | Batch
    | Head_dim
    | Heads
    | Key_len
    | Outer_n
    | Outer_t
    | Query_len

  type work_over_limit = {
    factor : work_factor;
    prefix : int64;
    extent : int64;
    limit : int64;
  }

  type error =
    | Batch_mismatch of dims_mismatch
    | Extent_mismatch of dims_mismatch
    | Mask_shape of mask_mismatch
    | Total_work_over_limit of work_over_limit

  val pp_error : Format.formatter -> error -> unit
end

(* `group_norm.default`'s own precondition: [num_groups] must divide the
   channel count evenly. *)
module Group_norm : sig
  type t = { channels : Dim.extent Dim.t; groups : Op_config.Pos.t }

  val pp : Format.formatter -> t -> unit
end

(* `index.Tensor`'s round-12 [Graph_ir]-level enforcement of the rank-1
   restriction: an index frame axis other than [C] with a non-unit extent,
   reachable from a direct [Graph_builder] call or a JSON-decoded graph even
   though neither importer can ever produce one. *)
module Index_tensor : sig
  type t = { axis : Axis.t; extent : Dim.extent Dim.t }

  val pp : Format.formatter -> t -> unit
end

(** `upsample_bilinear2d.vec`'s own aggregate: the coordinate transform scales
    the output index by [in_extent - 1] before dividing by [out_extent - 1] (see
    [Resize.Bilinear_axis]), and that product is a model-supplied aggregate the
    same way {!Adaptive_pool}'s [input_extent * output_size] is -- checked
    before [Resize]'s [Compute] ever multiplies the two. *)
module Resize : sig
  type t = {
    axis : Axis.t;
    in_extent : Dim.extent Dim.t;
    out_extent : Op_config.Pos.t;
    aggregate : int64;
    limit : int64;
  }

  val pp : Format.formatter -> t -> unit
end

(** `upsample_nearest2d.vec`'s own aggregate: the coordinate transform scales
    the output index by [in_extent] directly (see {!Resize.Nearest_axis}), no
    [- 1] anywhere -- a different formula from {!Resize}'s, so a distinct
    message rather than a shared field-compatible type that would describe the
    wrong arithmetic. *)
module Resize_nearest : sig
  type t = {
    axis : Axis.t;
    in_extent : Dim.extent Dim.t;
    out_extent : Op_config.Pos.t;
    aggregate : int64;
    limit : int64;
  }

  val pp : Format.formatter -> t -> unit
end

module Arange : sig
  type fault = [ `Empty | `Non_finite | `Non_positive_step | `Over_limit ]
  type t = { start : float; stop : float; step : float; fault : fault }

  val pp : Format.formatter -> t -> unit
end

type t =
  [ `Adaptive_pool of Adaptive_pool.t
  | `Batched_matmul of Batched_matmul.error
  | `Bmm of Bmm.error
  | `Broadcast of Broadcast.t
  | `Clamp of Clamp.error
  | `Concat of Concat.t
  | `Convolution of Convolution.error
  | `Group_norm of Group_norm.t
  | `Index_tensor of Index_tensor.t
  | `Linear of Linear.error
  | `Numel_over_limit of Vec6.Numel_bound.t
  | `Operand_shape of Operand_shape.t
  | `Output_count_over_limit of Output_count.t
  | `Pad of Pad.t
  | `Permute of Permute.t
  | `Reshape of Reshape.t
  | `Arange of Arange.t
  | `Resize of Resize.t
  | `Resize_nearest of Resize_nearest.t
  | `Sdpa of Sdpa.error
  | `Select_scatter of Select_scatter.t
  | `Slice of Slice.t
  | `Split_with_sizes of Split_with_sizes.t
  | `Unfold of Unfold.t
  | `Window of Window.t
  | `Window_over_limit of Window_over_limit.t ]

val pp : Format.formatter -> [< t ] -> unit
