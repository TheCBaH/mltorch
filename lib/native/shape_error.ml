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
    [ `Dilation
    | `Effective_kernel
    | `In_channels
    | `Input_extent
    | `Kernel
    | `Output_extent
    | `Padding
    | `Stride ]

  type t = { what : quantity; value : int64; limit : int64 }

  let pp ppf { what; value; limit } =
    Fmt.pf ppf "%s is %Ld, over the engine maximum of %Ld"
      (match what with
      | `Dilation -> "the dilation"
      | `Effective_kernel -> "the effective kernel dilation * (kernel - 1) + 1"
      | `In_channels -> "the input channel count weight.C * groups"
      | `Input_extent -> "the input extent"
      | `Kernel -> "the kernel extent"
      | `Output_extent -> "the output extent"
      | `Padding -> "the padding"
      | `Stride -> "the stride")
      value limit
end

(* Adaptive pooling multiplies an input extent by the requested output extent
   before dividing it into ATen's variable-width bins.  The product must stay
   below the JS-safe index ceiling: [index_scale] takes an [int], and
   js_of_ocaml's [int] is 32 bits.  This is deliberately a distinct error from
   [Window_over_limit], whose fields describe fixed-window geometry. *)
module Adaptive_pool = struct
  type t = {
    axis : Axis.t;
    input_extent : Dim.extent Dim.t;
    output_size : Op_config.Pos.t;
    aggregate : int64;
    limit : int64;
  }

  let pp ppf { axis; input_extent; output_size; aggregate; limit } =
    Fmt.pf ppf
      "adaptive_avg_pool2d: input extent %a times output_size %a on axis %a is \
       %Ld, which must be below the engine maximum of %Ld"
      Dim.pp input_extent Op_config.Pos.pp output_size Axis.pp axis aggregate
      limit
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

  let pp ppf { operand; expected; actual } =
    Fmt.pf ppf "%s shape must be %a, got %a"
      (match operand with
      | `Bias -> "bias"
      | `Batch_norm_no_stats_affine -> "batch_norm_no_stats affine"
      | `Group_norm_bias -> "group_norm bias"
      | `Group_norm_weight -> "group_norm weight"
      | `Layer_norm_bias -> "layer_norm bias"
      | `Layer_norm_weight -> "layer_norm weight"
      | `Lstm_bias -> "lstm bias"
      | `Lstm_input -> "lstm input"
      | `Lstm_state -> "lstm initial state"
      | `Lstm_weight_hh -> "lstm weight_hh"
      | `Lstm_weight_ih -> "lstm weight_ih"
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

(* Configuration rejections for the WIP `Lstm` op (project step 13):
   payload/domain facts no per-operand shape comparison can express, distinct
   from [Operand_shape]'s `Lstm_*` rows (which compare one operand's shape
   against an expected one). Each is a real _ai_/project_todo.md M3b/M5 gap,
   not a placeholder -- rejected with a typed diagnostic rather than silently
   mishandled. *)
module Lstm = struct
  type error = Empty_layers | Nonuniform_direction

  let pp_error ppf = function
    | Empty_layers -> Fmt.string ppf "lstm: at least one layer is required"
    | Nonuniform_direction ->
        Fmt.string ppf
          "lstm: every layer must have a reverse direction, or none must"
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

(* [Bmm]'s batched/multi-head generalization (`Matmul.Batched_matmul`): the
   batch mismatch names WHICH of [N]/[T]/[D]/[H] disagreed, since any of the
   four can be the one an unequal-rank or unbroadcast import produces. *)
module Batched_matmul = struct
  type dims_mismatch = {
    axis : Axis.t;
    lhs : Dim.extent Dim.t;
    rhs : Dim.extent Dim.t;
  }

  type error =
    | Batch_mismatch of dims_mismatch
    | Contract_mismatch of { lhs : Dim.extent Dim.t; rhs : Dim.extent Dim.t }

  let pp_error ppf (e : error) =
    match e with
    | Batch_mismatch { axis; lhs; rhs } ->
        Fmt.pf ppf
          "input %a extent must equal mat2 %a extent, or one must be 1: %a vs \
           %a"
          Axis.pp axis Axis.pp axis Dim.pp lhs Dim.pp rhs
    | Contract_mismatch { lhs; rhs } ->
        Fmt.pf ppf "input C extent must equal mat2 W extent: %a vs %a" Dim.pp
          lhs Dim.pp rhs
end

(* `index.Tensor`'s round-12 [Graph_ir]-level enforcement of the rank-1
   restriction: an index frame axis other than [C] with a non-unit extent,
   reachable from a direct [Graph_builder] call or a JSON-decoded graph even
   though neither importer can ever produce one. *)
module Index_tensor = struct
  type t = { axis : Axis.t; extent : Dim.extent Dim.t }

  let pp ppf { axis; extent } =
    Fmt.pf ppf
      "index.Tensor: index axis %a must have extent 1 (only axis C carries \
       real data), got %a"
      Axis.pp axis Dim.pp extent
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

(* A factory range has no operand shape to constrain it, so its scalar bounds
   are checked by the factory itself.  Native does not represent empty tensors,
   and keeps every concrete extent below the same JS-safe limit as a graph
   input. *)
module Arange = struct
  type fault = [ `Empty | `Non_finite | `Non_positive_step | `Over_limit ]
  type t = { start : float; stop : float; step : float; fault : fault }

  let pp ppf { start; stop; step; fault } =
    Fmt.pf ppf "arange(%g, %g, %g): %s" start stop step
      (match fault with
      | `Empty -> "selects no elements; Native has no empty tensor"
      | `Non_finite -> "bounds and step must be finite"
      | `Non_positive_step -> "only a positive step is supported"
      | `Over_limit -> "result extent exceeds the engine limit")
end

(* [select_scatter]'s [src] does not have exactly the shape [select] itself
   would produce at [axis]/[index] -- [expected] is that derived shape, not
   restated here (see [Split.Select_scatter.output_shape]). *)
module Select_scatter = struct
  type t = {
    axis : Axis.t;
    index : int;
    expected : Vec6.shape;
    actual : Vec6.shape;
  }

  let pp ppf { axis; index; expected; actual } =
    Fmt.pf ppf "select_scatter src shape must be %a (axis=%a index=%d), got %a"
      Vec6.pp_shape expected Axis.pp axis index Vec6.pp_shape actual
end

module Slice = struct
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

  let pp ppf { axis; in_extent; start; stop; step; out; fault } =
    Fmt.pf ppf "slice of axis %a [%d, %d) step %d over extent %a %s" Axis.pp
      axis start stop
      (step :> int)
      Dim.pp in_extent
      (match fault with
      | `Empty ->
          Fmt.str "selects %Ld elements; the engine has no empty extent" out
      | `Out_of_range -> "is not within 0 <= start <= stop <= extent")
end

module Pad = struct
  type fault = [ `Duplicate_axis | `Empty | `Reflect_width ]

  type t = {
    axis : Axis.t;
    in_extent : Dim.extent Dim.t;
    before : int64;
    after : int64;
    out : int64;
    fault : fault;
  }

  let pp ppf { axis; in_extent; before; after; out; fault } =
    match fault with
    | `Duplicate_axis ->
        Fmt.pf ppf "axis %a has more than one pad entry" Axis.pp axis
    | `Empty ->
        Fmt.pf ppf
          "pad of axis %a by (%Ld, %Ld) over extent %a leaves %Ld elements; \
           the engine has no empty extent"
          Axis.pp axis before after Dim.pp in_extent out
    | `Reflect_width ->
        Fmt.pf ppf
          "reflect pad of axis %a by (%Ld, %Ld) needs each side below the \
           extent %a"
          Axis.pp axis before after Dim.pp in_extent
end

(* `aten.cat.default`'s two possible faults: no tensors at all, or a
   non-concatenated axis that disagrees between two operands. The
   concatenated axis's own overflow reuses [Window_over_limit]'s
   [`Output_extent] row instead of a third fault here -- it IS the output
   extent on that axis. *)
module Concat = struct
  type mismatch = {
    axis : Axis.t;
    first : Dim.extent Dim.t;
    other : Dim.extent Dim.t;
  }

  type t = Axis_mismatch of mismatch | Empty

  let pp ppf = function
    | Axis_mismatch { axis; first; other } ->
        Fmt.pf ppf
          "concat: axis %a extent must agree across every tensor (it is not \
           the concatenated axis): %a vs %a"
          Axis.pp axis Dim.pp first Dim.pp other
    | Empty -> Fmt.string ppf "concat: at least one tensor is required"
end

module Split_with_sizes = struct
  type fault =
    | Non_positive_size of { index : int; size : int }
    | Size_mismatch of { total : int64 }

  type t = { axis : Axis.t; in_extent : Dim.extent Dim.t; fault : fault }

  let pp ppf { axis; in_extent; fault } =
    match fault with
    | Non_positive_size { index; size } ->
        Fmt.pf ppf
          "split_with_sizes of axis %a over extent %a: size %d at index %d is \
           not positive; the engine has no empty extent"
          Axis.pp axis Dim.pp in_extent size index
    | Size_mismatch { total } ->
        Fmt.pf ppf
          "split_with_sizes of axis %a: sizes sum to %Ld, not the axis's \
           extent %a"
          Axis.pp axis total Dim.pp in_extent
end

module Unfold = struct
  type fault = Reserved_axis | Rank_exceeded
  type t = { axis : Axis.t; fault : fault }

  let pp ppf { axis; fault } =
    match fault with
    | Reserved_axis ->
        Fmt.pf ppf
          "unfold axis must not be C (got %a): C is reserved for the window \
           offset"
          Axis.pp axis
    | Rank_exceeded ->
        Fmt.pf ppf
          "unfold axis %a: input already uses all six axes (N has extent > 1), \
           no room for the new window axis"
          Axis.pp axis
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
    | Input_channels_mismatch { input_channels; expected_in_channels } ->
        Fmt.pf ppf "input C extent must equal in_channels: %a vs %a" Dim.pp
          input_channels Dim.pp expected_in_channels
    | Out_channels_not_divisible_by_groups { channels; groups } ->
        Fmt.pf ppf "out_channels %d must be divisible by groups %d" channels
          groups
    | Output_padding_nonzero { h; w } ->
        Fmt.pf ppf
          "output_padding must be zero for non-transposed convolution, got \
           {h=%a; w=%a}"
          Op_config.Nonneg.pp h Op_config.Nonneg.pp w
    | Same_padding_requires_stride_one { stride } ->
        Fmt.pf ppf
          "padding=\"same\" is not supported for strided convolutions \
           (stride=%a)"
          Op_config.Pos.pp stride
    | Transposed_input_channels_not_divisible_by_groups { channels; groups } ->
        Fmt.pf ppf "input C extent %d must be divisible by groups %d" channels
          groups
    | Transposed_not_supported ->
        Fmt.string ppf
          "transposed convolutions are not supported in forward Conv2d lowering"
    | Transposed_output_non_positive d -> pp_transposed_window_output ppf d
    | Transposed_weight_input_mismatch { weight_input_channels; input_channels }
      ->
        Fmt.pf ppf
          "transposed weight N extent must equal input C extent: %a vs %a"
          Dim.pp weight_input_channels Dim.pp input_channels
    | Weight_channels_mismatch { weight_in_per_group; expected_in_per_group } ->
        Fmt.pf ppf "weight C extent %d must equal in_channels/groups %d"
          weight_in_per_group expected_in_per_group
    | Weight_kernel_mismatch { axis; weight_extent; kernel_extent } ->
        Fmt.pf ppf "weight %a extent must equal kernel extent: %a vs %a" Axis.pp
          axis Dim.pp weight_extent Dim.pp kernel_extent
end

module Output_count = struct
  type observed = At_least of int | Exact of int
  type t = { limit : int; observed : observed }

  (* [limit] is EXCLUSIVE, so the printed figure is the largest accepted count.
     Printing [limit] itself reads as a contradiction in the boundary case,
     where the observed count equals it. *)
  let pp ppf { limit; observed } =
    match observed with
    | At_least n ->
        Fmt.pf ppf "at least %d outputs, above the maximum of %d" n (limit - 1)
    | Exact n -> Fmt.pf ppf "%d outputs, above the maximum of %d" n (limit - 1)
end

(* [Sdpa]'s own row. The two per-shape numel checks (score count, output
   numel) reuse the generic [`Numel_over_limit] above; this module is only
   what that one cannot express: the two extent-agreement checks common to
   every operand pair, the mask's broadcast-or-equal rule against
   [Attention.Sdpa.score_shape], and the SIXTH-factor total-work bound
   (op8-impl.md F12) — [D * H * Wq * Wk * E * E], recomputed per output
   feature, which no single [Vec6.shape] holds and neither the score-count nor
   the output-numel bound implies (the F12 counterexample passes both by six
   orders of magnitude). *)
module Sdpa = struct
  (* Which pair of operands an extent-agreement check compared, so the message
     says which two shapes disagreed rather than just "some pair". *)
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

  (* [Outer_n]/[Outer_t] are the frame's leading N/T axes -- unused by this
     op's own semantics (F7 names only D/H/W/C), but review op8-impl-review.md
     P1 found they were left unchecked entirely: a standalone or JSON-decoded
     graph could give query/key/value disagreeing N or T, which either reads
     an operand out of bounds or silently drops part of it, and the resource
     bound below did not count their contribution to total work either. They
     get exactly [Batch]/[Heads]'s treatment: cross-operand equality, and a
     factor in the work bound. *)
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

  let pp_check ppf = function
    | `Key_value -> Fmt.string ppf "key vs value"
    | `Query_key -> Fmt.string ppf "query vs key"
    | `Query_value -> Fmt.string ppf "query vs value"

  let pp_work_factor ppf = function
    | Batch -> Fmt.string ppf "the batch extent D"
    | Head_dim -> Fmt.string ppf "the head-dimension extent E"
    | Heads -> Fmt.string ppf "the head-count extent H"
    | Key_len -> Fmt.string ppf "the key sequence extent Wk"
    | Outer_n -> Fmt.string ppf "the outer extent N"
    | Outer_t -> Fmt.string ppf "the outer extent T"
    | Query_len -> Fmt.string ppf "the query sequence extent Wq"

  let pp_error ppf = function
    (* [N]/[T]/[D]/[H], the four batch-like axes real ATen broadcasts across
       the op's two chained matmuls (`query @ key^T`, then `attn @ value`) --
       equal, or one side 1, [Matmul.Batched_matmul]'s own rule. *)
    | Batch_mismatch { axis; check; lhs; rhs } ->
        Fmt.pf ppf "sdpa: %a extent must agree (%a), or one must be 1: %a vs %a"
          Axis.pp axis pp_check check Dim.pp lhs Dim.pp rhs
    (* [C] (the head dimension, `Ev = E`) and [W] (key/value's shared
       sequence extent) are never batch axes -- STRICT equality, no
       broadcast, the flash-oracle constraint stays exactly what it was. *)
    | Extent_mismatch { axis; check; lhs; rhs } ->
        Fmt.pf ppf "sdpa: %a extent must agree (%a): %a vs %a" Axis.pp axis
          pp_check check Dim.pp lhs Dim.pp rhs
    | Mask_shape { axis; mask; score } ->
        Fmt.pf ppf
          "sdpa: mask extent on %a must be %a (the score extent) or 1, got %a"
          Axis.pp axis Dim.pp score Dim.pp mask
    | Total_work_over_limit { factor; prefix; extent; limit } ->
        Fmt.pf ppf
          "sdpa: total work N*T*D*H*Wq*Wk*E*E (score, row max and denominator \
           are recomputed per output feature) exceeds %Ld after folding in %a \
           (running product %Ld, this factor %Ld)"
          limit pp_work_factor factor prefix extent
end

(* `group_norm.default`'s own precondition: [num_groups] must divide the
   channel count evenly, ATen's own contract. Nothing upstream proves it --
   [num_groups] arrives as an ordinary int from a PT2 graph -- so it is
   checked here rather than assumed. *)
module Group_norm = struct
  type t = { channels : Dim.extent Dim.t; groups : Op_config.Pos.t }

  let pp ppf { channels; groups } =
    Fmt.pf ppf "group_norm: channel count %a is not divisible by num_groups %a"
      Dim.pp channels Op_config.Pos.pp groups
end

(* `upsample_bilinear2d.vec`'s own aggregate: the coordinate transform scales
   the output index by [in_extent - 1] before dividing by [out_extent - 1]
   (see [Resize.Bilinear_axis]), and that product is a model-supplied
   aggregate the same way [Adaptive_pool]'s [input_extent * output_size] is --
   checked here, before [Resize]'s [Compute] ever multiplies the two. *)
module Resize = struct
  type t = {
    axis : Axis.t;
    in_extent : Dim.extent Dim.t;
    out_extent : Op_config.Pos.t;
    aggregate : int64;
    limit : int64;
  }

  let pp ppf { axis; in_extent; out_extent; aggregate; limit } =
    Fmt.pf ppf
      "upsample_bilinear2d: (input extent %a - 1) times (output_size %a - 1) \
       on axis %a is %Ld, which must be below the engine maximum of %Ld"
      Dim.pp in_extent Op_config.Pos.pp out_extent Axis.pp axis aggregate limit
end

(* `upsample_nearest2d.vec`'s own aggregate: the coordinate transform scales
   the output index by [in_extent] directly (see [Resize.Nearest_axis]), no
   [- 1] anywhere -- a different formula from [Resize.Bilinear_axis]'s, so a
   distinct message rather than a shared field-compatible type that would
   describe the wrong arithmetic. Same field shape as [Adaptive_pool]'s
   [input_extent * output_size], and the same phrasing. *)
module Resize_nearest = struct
  type t = {
    axis : Axis.t;
    in_extent : Dim.extent Dim.t;
    out_extent : Op_config.Pos.t;
    aggregate : int64;
    limit : int64;
  }

  let pp ppf { axis; in_extent; out_extent; aggregate; limit } =
    Fmt.pf ppf
      "upsample_nearest2d: input extent %a times output_size %a on axis %a is \
       %Ld, which must be below the engine maximum of %Ld"
      Dim.pp in_extent Op_config.Pos.pp out_extent Axis.pp axis aggregate limit
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
  | `Lstm of Lstm.error
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

let pp ppf = function
  | `Adaptive_pool e -> Adaptive_pool.pp ppf e
  | `Batched_matmul e -> Batched_matmul.pp_error ppf e
  | `Bmm e -> Bmm.pp_error ppf e
  | `Broadcast e -> Broadcast.pp ppf e
  | `Clamp e -> Clamp.pp_error ppf e
  | `Concat e -> Concat.pp ppf e
  | `Convolution e -> Convolution.pp_error ppf e
  | `Group_norm e -> Group_norm.pp ppf e
  | `Index_tensor e -> Index_tensor.pp ppf e
  | `Linear e -> Linear.pp_error ppf e
  | `Lstm e -> Lstm.pp_error ppf e
  | `Numel_over_limit e -> Vec6.Numel_bound.pp ppf e
  | `Operand_shape e -> Operand_shape.pp ppf e
  | `Output_count_over_limit e -> Output_count.pp ppf e
  | `Pad e -> Pad.pp ppf e
  | `Permute e -> Permute.pp ppf e
  | `Reshape e -> Reshape.pp ppf e
  | `Arange e -> Arange.pp ppf e
  | `Resize e -> Resize.pp ppf e
  | `Resize_nearest e -> Resize_nearest.pp ppf e
  | `Sdpa e -> Sdpa.pp_error ppf e
  | `Select_scatter e -> Select_scatter.pp ppf e
  | `Slice e -> Slice.pp ppf e
  | `Split_with_sizes e -> Split_with_sizes.pp ppf e
  | `Unfold e -> Unfold.pp ppf e
  | `Window e -> Window.pp ppf e
  | `Window_over_limit e -> Window_over_limit.pp ppf e
