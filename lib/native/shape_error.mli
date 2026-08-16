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
    [ `Kernel
    | `Dilation
    | `Stride
    | `Padding
    | `Input_extent
    | `Effective_kernel
    | `Output_extent
    | `In_channels ]
  (** Which quantity ran past the ceiling. A closed set rather than a string:
      each is a specific field or a specific derived product, and a reader has
      to know which. [`In_channels] is the conv weight's per-group extent times
      its group count — named for what it IS, not for the extent it is on its
      way to becoming. *)

  type t = { what : quantity; value : int64; limit : int64 }

  val pp : Format.formatter -> t -> unit
end

(** An OPTIONAL operand whose shape disagrees with the one the op requires. One
    row rather than one per op: the fault is identical in every case and the
    [operand] tag says which slot. Every [output_shape] takes the REQUIRED
    operands only, so before this an optional one reached evaluation unchecked
    and raised from [Tensor.read]'s bounds check partway through the result. *)
module Operand_shape : sig
  type t = {
    operand : [ `Bias | `Rms_norm_weight ];
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

module Permute : sig
  type t = { side : perm_side; axis : Axis.t; count : int }

  val pp : Format.formatter -> t -> unit
end

(* [Reshape.output_shape]'s numel-preservation precondition, violated: a target
   whose element count disagrees with the source's. *)
module Reshape : sig
  type t = { source : Vec6.shape; target : Vec6.shape }

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
    | Out_channels_not_divisible_by_groups of channels_divisibility
    | Weight_channels_mismatch of weight_channels_mismatch
    | Weight_kernel_mismatch of weight_kernel_mismatch
    | Input_channels_mismatch of input_channels_mismatch
    | Same_padding_requires_stride_one of { stride : Op_config.Pos.t }
    | Transposed_not_supported
    | Output_padding_nonzero of output_padding
    | Transposed_weight_input_mismatch of transposed_weight_input_mismatch
    | Transposed_input_channels_not_divisible_by_groups of channels_divisibility
    | Transposed_output_non_positive of transposed_window_output

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
  type observed = Exact of int | At_least of int
  type t = { limit : int; observed : observed }

  val pp : Format.formatter -> t -> unit
end

type t =
  [ `Broadcast of Broadcast.t
  | `Window of Window.t
  | `Window_over_limit of Window_over_limit.t
  | `Operand_shape of Operand_shape.t
  | `Clamp of Clamp.error
  | `Linear of Linear.error
  | `Bmm of Bmm.error
  | `Output_count_over_limit of Output_count.t
  | `Permute of Permute.t
  | `Reshape of Reshape.t
  | `Numel_over_limit of Vec6.Numel_bound.t
  | `Convolution of Convolution.error ]

val pp : Format.formatter -> [< t ] -> unit
