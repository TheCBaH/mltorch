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
    out : int;
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
      "output extent must be >= 1, got %d (in=%a kernel=%a stride=%a \
       pad_before=%a pad_after=%a dilation=%a)"
      out Dim.pp in_extent Dim.pp kernel Op_config.Pos.pp stride
      Op_config.Nonneg.pp pad_before Op_config.Nonneg.pp pad_after
      Op_config.Pos.pp dilation
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

type t =
  [ `Broadcast of Broadcast.t
  | `Window of Window.t
  | `Clamp of Clamp.error
  | `Linear of Linear.error
  | `Bmm of Bmm.error
  | `Permute of Permute.t
  | `Convolution of Convolution.error ]

let pp ppf = function
  | `Broadcast e -> Broadcast.pp ppf e
  | `Window e -> Window.pp ppf e
  | `Clamp e -> Clamp.pp_error ppf e
  | `Linear e -> Linear.pp_error ppf e
  | `Bmm e -> Bmm.pp_error ppf e
  | `Permute e -> Permute.pp ppf e
  | `Convolution e -> Convolution.pp_error ppf e
