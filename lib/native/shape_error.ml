type perm_side = [ `Input | `Output ]

let pp_perm_side ppf = function
  | `Input -> Fmt.string ppf "input"
  | `Output -> Fmt.string ppf "output"

module Broadcast = struct
  type t = { axis : Axis.t; lhs : Dim.extent Dim.t; rhs : Dim.extent Dim.t }

  let pp ppf { axis; lhs; rhs } =
    Fmt.pf ppf "incompatible broadcast extents on axis %a: %a vs %a" Axis.pp
      axis Dim.pp lhs Dim.pp rhs
end

module Window = struct
  type t = {
    out : int64;
        (* [int64], not [int]: this figure is a SUM of model-supplied extents
           minus a PRODUCT of two more, and js_of_ocaml's [int] is 32 bits, so
           reporting it as an [int] would print the wrapped number that made it
           a defect. *)
    in_extent : Dim.extent Dim.t;
    kernel : Dim.extent Dim.t;
    stride : Op_config.Pos.t;
    pad_before : Op_config.Nonneg.t;
    pad_after : Op_config.Nonneg.t;
    dilation : Op_config.Pos.t;
  }

  let pp ppf { out; in_extent; kernel; stride; pad_before; pad_after; dilation }
      =
    Fmt.pf ppf
      "output extent must be >= 1, got %Ld (in=%a kernel=%a stride=%a \
       pad_before=%a pad_after=%a dilation=%a)"
      out Dim.pp in_extent Dim.pp kernel Op_config.Pos.pp stride
      Op_config.Nonneg.pp pad_before Op_config.Nonneg.pp pad_after
      Op_config.Pos.pp dilation
end

(* An output extent, or the effective kernel it is computed from, past the
   engine's per-axis ceiling. Distinct from [Window], which is the same
   arithmetic coming out too SMALL: a window that shrinks an axis to nothing is
   an ordinary configuration error, while one that exceeds the ceiling is a
   value the engine has no representation for at all. Both figures are [int64]
   for the reason [Window.out] is. *)
module Window_over_limit = struct
  (* Which quantity ran past the ceiling. A closed set rather than a string:
     every one of these is a specific field or a specific derived product, and a
     reader has to know which. [`In_channels] is the conv weight's per-group
     extent times its group count -- named for what it IS, not for the extent it
     is on its way to becoming. *)
  type quantity =
    [ `Kernel
    | `Dilation
    | `Stride
    | `Padding
    | `Input_extent
    | `Effective_kernel
    | `Output_extent
    | `In_channels ]

  type t = { what : quantity; value : int64; limit : int64 }

  let pp ppf { what; value; limit } =
    Fmt.pf ppf "%s is %Ld, over the engine maximum of %Ld"
      (match what with
      | `Kernel -> "the kernel extent"
      | `Dilation -> "the dilation"
      | `Stride -> "the stride"
      | `Padding -> "the padding"
      | `Input_extent -> "the input extent"
      | `Effective_kernel -> "the effective kernel dilation * (kernel - 1) + 1"
      | `Output_extent -> "the output extent"
      | `In_channels -> "the input channel count weight.C * groups")
      value limit
end

(* An OPTIONAL operand whose shape disagrees with the one the op requires. Its
   own row rather than a per-op variant because the fault is the same in every
   case -- a shape was offered and a different one was needed -- and the
   [operand] tag says which slot without multiplying the rows.

   These were checked nowhere. Every [output_shape] takes the required operands
   only, so an optional one passed straight through to evaluation and raised
   from [Tensor.read]'s strict bounds check, uncaught, partway through the
   result. *)
module Operand_shape = struct
  type t = {
    operand : [ `Bias | `Rms_norm_weight ];
    expected : Vec6.shape;
    actual : Vec6.shape;
  }

  let pp ppf { operand; expected; actual } =
    Fmt.pf ppf "%s shape must be %a, got %a"
      (match operand with
      | `Bias -> "bias"
      | `Rms_norm_weight -> "rms_norm weight")
      Vec6.pp_shape expected Vec6.pp_shape actual
end

module Clamp = struct
  type error = No_bounds

  let pp_error ppf = function
    | No_bounds ->
        Fmt.string ppf "clamp: at least one of 'min' or 'max' must be given"
end

module Linear = struct
  type channels_mismatch = {
    actual : Dim.extent Dim.t;
    expected : Dim.extent Dim.t;
  }

  type error =
    | Input_channels_mismatch of channels_mismatch
    | Weight_channels_mismatch of channels_mismatch

  let pp_error ppf (e : error) =
    match e with
    | Input_channels_mismatch { actual; expected } ->
        Fmt.pf ppf "input C extent must equal in_features: %a vs %a" Dim.pp
          actual Dim.pp expected
    | Weight_channels_mismatch { actual; expected } ->
        Fmt.pf ppf "weight C extent must equal in_features: %a vs %a" Dim.pp
          actual Dim.pp expected
end

module Bmm = struct
  type dims_mismatch = { lhs : Dim.extent Dim.t; rhs : Dim.extent Dim.t }

  type error =
    | Batch_mismatch of dims_mismatch
    | Contract_mismatch of dims_mismatch

  let pp_error ppf (e : error) =
    match e with
    | Batch_mismatch { lhs; rhs } ->
        Fmt.pf ppf "input H extent must equal mat2 H extent: %a vs %a" Dim.pp
          lhs Dim.pp rhs
    | Contract_mismatch { lhs; rhs } ->
        Fmt.pf ppf "input C extent must equal mat2 W extent: %a vs %a" Dim.pp
          lhs Dim.pp rhs
end

module Permute = struct
  type t = { side : perm_side; axis : Axis.t; count : int }

  let pp ppf { side; axis; count } =
    Fmt.pf ppf
      "expected axis %a to appear exactly once on the %a side, found %d" Axis.pp
      axis pp_perm_side side count
end

module Reshape = struct
  type t = { source : Vec6.shape; target : Vec6.shape }

  let pp ppf { source; target } =
    Fmt.pf ppf "reshape target %a does not preserve the element count of %a"
      Vec6.pp_shape target Vec6.pp_shape source
end

module Convolution = struct
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

  let pp_transposed_window_output ppf
      { out; in_extent; kernel; stride; pad; dilation; output_padding } =
    Fmt.pf ppf
      "transposed output extent must be >= 1, got %d (in=%a kernel=%a \
       stride=%a pad=%a dilation=%a output_padding=%a)"
      out Dim.pp in_extent Dim.pp kernel Op_config.Pos.pp stride
      Op_config.Nonneg.pp pad Op_config.Pos.pp dilation Op_config.Nonneg.pp
      output_padding

  let pp_error ppf (e : error) =
    match e with
    | In_channels_not_divisible_by_groups { channels; groups } ->
        Fmt.pf ppf "in_channels %d must be divisible by groups %d" channels
          groups
    | Out_channels_not_divisible_by_groups { channels; groups } ->
        Fmt.pf ppf "out_channels %d must be divisible by groups %d" channels
          groups
    | Weight_channels_mismatch { weight_in_per_group; expected_in_per_group } ->
        Fmt.pf ppf "weight C extent %d must equal in_channels/groups %d"
          weight_in_per_group expected_in_per_group
    | Weight_kernel_mismatch { axis; weight_extent; kernel_extent } ->
        Fmt.pf ppf "weight %a extent must equal kernel extent: %a vs %a" Axis.pp
          axis Dim.pp weight_extent Dim.pp kernel_extent
    | Input_channels_mismatch { input_channels; expected_in_channels } ->
        Fmt.pf ppf "input C extent must equal in_channels: %a vs %a" Dim.pp
          input_channels Dim.pp expected_in_channels
    | Same_padding_requires_stride_one { stride } ->
        Fmt.pf ppf
          "padding=\"same\" is not supported for strided convolutions \
           (stride=%a)"
          Op_config.Pos.pp stride
    | Transposed_not_supported ->
        Fmt.string ppf
          "transposed convolutions are not supported in forward Conv2d lowering"
    | Output_padding_nonzero { h; w } ->
        Fmt.pf ppf
          "output_padding must be zero for non-transposed convolution, got \
           {h=%a; w=%a}"
          Op_config.Nonneg.pp h Op_config.Nonneg.pp w
    | Transposed_weight_input_mismatch { weight_input_channels; input_channels }
      ->
        Fmt.pf ppf
          "transposed weight N extent must equal input C extent: %a vs %a"
          Dim.pp weight_input_channels Dim.pp input_channels
    | Transposed_input_channels_not_divisible_by_groups { channels; groups } ->
        Fmt.pf ppf "input C extent %d must be divisible by groups %d" channels
          groups
    | Transposed_output_non_positive d -> pp_transposed_window_output ppf d
end

module Output_count = struct
  type observed = Exact of int | At_least of int
  type t = { limit : int; observed : observed }

  (* [limit] is EXCLUSIVE, so the printed figure is the largest accepted count.
     Printing [limit] itself reads as a contradiction in the boundary case,
     where the observed count equals it. *)
  let pp ppf { limit; observed } =
    match observed with
    | Exact n -> Fmt.pf ppf "%d outputs, above the maximum of %d" n (limit - 1)
    | At_least n ->
        Fmt.pf ppf "at least %d outputs, above the maximum of %d" n (limit - 1)
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

let pp ppf = function
  | `Broadcast e -> Broadcast.pp ppf e
  | `Window e -> Window.pp ppf e
  | `Window_over_limit e -> Window_over_limit.pp ppf e
  | `Operand_shape e -> Operand_shape.pp ppf e
  | `Clamp e -> Clamp.pp_error ppf e
  | `Linear e -> Linear.pp_error ppf e
  | `Bmm e -> Bmm.pp_error ppf e
  | `Output_count_over_limit e -> Output_count.pp ppf e
  | `Permute e -> Permute.pp ppf e
  | `Reshape e -> Reshape.pp ppf e
  | `Numel_over_limit e -> Vec6.Numel_bound.pp ppf e
  | `Convolution e -> Convolution.pp_error ppf e
