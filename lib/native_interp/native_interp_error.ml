(* Error payload types for [Native_interp], split from native_interp.ml. Purely data plus deterministic printing -- no dependency on
   [Pytorch_types] or the decode/lowering machinery, so this compiles first.
   native_interp.mli documents the public contract (types, [pp_error]); this
   file is not published under a [.mli] of its own because native_interp.ml
   re-exports every item here by manifest type/value alias, so the outer
   [.mli] is the only place the contract needs to be legible. *)

type arg_kind =
  [ `Tensor
  | `Optional_tensor
  | `Int_list
  | `Float_list
  | `Int
  | `Int_opt
  | `Bool
  | `Float
  | `Scalar
  | `Optional_scalar
  | `String
  | `Tensor_or_scalar
  | `Tensor_list
  | `Memory_format_opt ]

(* [Zero] is not a sub-case of [Negative]: the engine forbids an empty extent by
   construction ([Dim.extent] is >= 1), so a declared 0 is a shape this dialect
   has no form for rather than a nonsensical number. It also arrives from real
   models -- ATen's unbind of a zero-length dim returns an empty list -- where a
   negative size never does. Without its own arm it reached [Dim.extent] and
   escaped as an uncaught [Invalid_argument]. *)
module Expected_rank = struct
  type t = { expected : int; got : int }
end

type dim_fault =
  [ `Negative of int
  | `Zero
  | `Symbolic
  | `Rank_over_six
  | `Expected_rank of Expected_rank.t
  | `Over_max_extent of int64 ]

module Normalized_rank = struct
  type t = { op : Norm.Target.t; rank : int; got : int }
end

module Normalized_shape = struct
  type t = { op : Norm.Target.t; expected : int list; got : int list }
end

(* [native_layer_norm] returns [(out, mean, rstd)]. Native's [Layer_norm] has
   ONE output, so the trailing two are dropped -- which is only sound while
   nothing reads them. Unlike [_native_batch_norm_legit_no_training]'s, they are
   not empty: vit_b_32 records them as real [1, 50, 1] f32 tensors, so "recorded
   size-0 and therefore meaningless" is not the reason they may go. The reason
   is that every occurrence in the corpus is dead, and this row is what says so
   when one is not, instead of the [`Undefined_ssa] the dropped name would
   otherwise produce three nodes later -- a diagnostic naming the consumer
   rather than the op that failed to provide. *)
module Live_layer_norm_stats = struct
  type t = { op : string; stat : [ `Mean | `Rstd ]; ssa : string }
end

type metadata_role =
  [ `Tensor
  | `Convolution_weight
  | `Conv2d_weight
  | `Conv2d_padding_weight
  | `Linear_weight
  | `Conv2d_bias
  | `Conv2d_padding_bias
  | `Convolution_bias
  | `Linear_bias
  | `Rms_norm_input
  | `Rms_norm_weight
  | `Layer_norm_input
  | `Layer_norm_weight
  | `Layer_norm_bias
  | `Group_norm_weight
  | `Group_norm_bias
  | `Amax_input
  | `Mean_input
  | `Permute_input
  | `Transpose_input
  | `Pad_input
  | `Slice_input
  | `Unbind_input
  | `Split_with_sizes_input
  | `Select_input
  | `Unsqueeze_input
  | `Concat_input
  | `Stack_input
  | `Addmm_weight
  | `Sdpa_query
  | `Sdpa_key
  | `Sdpa_value
  | `Sdpa_mask
  | `Adaptive_avg_pool2d_input
  | `Vector_norm_input
  | `Upsample_bilinear2d_input ]

type hw_param =
  [ `Stride
  | `Padding
  | `Output_padding
  | `Dilation
  | `Kernel_size
  | `Output_size ]

(* ONE definition of this vocabulary, shared with [Op_bridge] through
   [Op_config.Bad]: the two importers must reject the same values, and two
   spellings of "this stride is zero" is one drift away from two contracts. *)
type config_param = Op_config.Bad.param
type config_fault = Op_config.Bad.fault

(* Two genuinely different rejections, not one with a message: a graph input
   that is not a tensor, and a graph whose user-input arity the runner cannot
   satisfy. Both are recoverable ([Me_classify.lowering]); only the second has
   a figure to report. *)
type unsupported_input = [ `Non_tensor | `Not_exactly_one_user_input of int ]

type unsupported_option =
  [ `Alpha of float
  | `Memory_format of [ `Channels_last | `Channels_last_3d | `Unknown ]
  | `Dilation of int list
  | `Approximate of string
  | `Dtype
  | `Vector_norm_ord of float ]

(* Own modules, per the record-namespace convention: three of these carry an
   [op] field and two an [arg], and distinct namespaces are how this repo keeps
   labels unique rather than silencing warning 30. *)
module Missing_arg = struct
  type t = { op : string; arg : string }
end

module Wrong_arg_kind = struct
  type t = { op : string; arg : string; expected : arg_kind }
end

(* A [SymInt] argument that arrived as a NAME rather than a value. Its own row
   rather than a [`Wrong_arg_kind]: the spelling is right and the kind is right,
   and what is missing is a binding for the symbol -- the same distinction
   [`Bad_dimension { fault = `Symbolic }] draws for tensor METADATA, applied to
   an argument. Carries the symbol, because "which one" is the actionable half. *)
module Unresolved_sym_arg = struct
  type t = { op : string; arg : string; symbol : string }
end

module Bad_dimension = struct
  type t = { tensor : string; fault : dim_fault }
end

module Missing_metadata = struct
  type t = { ssa : string; role : metadata_role }
end

module Axis_out_of_range = struct
  type t = { axis : int; rank : int }
end

module Bad_arity = struct
  type t = { param : hw_param; got : int }
end

module Adaptive_pool_rank = struct
  type t = { tensor : string; got : int }
end

module Bad_config = Op_config.Bad

module Unsupported_option = struct
  type t = { op : string; option : unsupported_option }
end

(* A `Tensor[]`-returning node carries ONE output holding every result name, so
   its arity is model data on one side and derived from the operand's extent on
   the other. Disagreement is a malformed graph, not a defect: [add_env]'s
   [Invalid_argument] stays an invariant about this module, and this check runs
   before it so a bad export can never reach that boundary. *)
module Output_arity = struct
  type t = { op : string; serialized : int; derived : int }
end

module Bad_view = struct
  type t = {
    size : int list;
    fault :
      [ `Aten_shape of Aten_shape.error
      | `Numel_over_limit of Vec6.Numel_bound.t ];
  }
end

module Bad_slice = struct
  type t = {
    start : int option;
    stop : int option;
    step : int;
    fault : [ `Aten_shape of Aten_shape.error ];
  }
end

module Bad_select = struct
  type t = { index : int; fault : [ `Aten_shape of Aten_shape.error ] }
end

(* [cat.default]/[stack.default]: every tensor in the list must share one
   rank, the same check [Op_bridge]'s [Concat_rank_mismatch] makes and for
   the same reason -- two different ranks land their data on different frame
   axes, so a rank disagreement would otherwise surface as a confusing
   off-axis extent mismatch from [Concat]'s own shape rule rather than the
   rank fault it actually is. *)
module Concat_rank_mismatch = struct
  type t = { op : string; first : int; other : int }
end

(* `upsample_bilinear2d.vec`'s own contract, mirroring ATen's own
   `compute_output_size`: exactly one of [output_size]/[scale_factors] must be
   given, and a given [scale_factors] must name both spatial axes. *)
module Bad_upsample_size = struct
  type fault = Neither | Both | Bad_scale_arity of int
  type t = { op : string; fault : fault }
end

type malformed =
  [ `Missing_arg of Missing_arg.t
  | `Wrong_arg_kind of Wrong_arg_kind.t
  | `Unresolved_sym_arg of Unresolved_sym_arg.t
  | `Missing_metadata of Missing_metadata.t
  | `Bad_dimension of Bad_dimension.t
  | `Axis_out_of_range of Axis_out_of_range.t
  | `Bad_arity of Bad_arity.t
  | `Adaptive_pool_rank of Adaptive_pool_rank.t
  | `Bad_config of Bad_config.t
  | `Unsupported_padding_mode of string
  | `Bad_pad_list of Pad.Pad.Bad_pad_list.t
  | `Normalized_rank of Normalized_rank.t
  | `Normalized_shape of Normalized_shape.t
  | `Live_layer_norm_stats of Live_layer_norm_stats.t
  | `Unsupported_option of Unsupported_option.t
  | `Output_arity of Output_arity.t
  | `Non_tensor_node_output of string
  | `Non_tensor_graph_output
  | `Undefined_ssa of string
  | `Output_not_evaluated of Graph_ir.Tensor_id.t
  | `Bad_view of Bad_view.t
  | `Bad_slice of Bad_slice.t
  | `Bad_select of Bad_select.t
  | `Sdpa_reject of Attention.Sdpa.Reject.t
  | `Concat_no_tensors of string
  | `Concat_rank_mismatch of Concat_rank_mismatch.t
  | `Bad_upsample_size of Bad_upsample_size.t ]

module Rank_mismatch = struct
  type t = { sizes : int; strides : int }
end

module Storage_range = struct
  type t = { lo : int64; hi : int64; data_bytes : int }
end

(* Loading a captured tensor, which is a different job from reading graph
   metadata even though it reuses [shape_of_sizes] and so can throw its rows. *)
type tensor_bridge =
  [ malformed
  | `Rank_mismatch of Rank_mismatch.t
  | `Storage_index_overflow
  | `Storage_out_of_range of Storage_range.t
  | `Materialize_failed of string
    (* [Invalid_argument]'s own message — a third-party payload, named for its
       source rather than left to read as a case declined to classify. *)
  | `Unsupported_dtype of Pt2_dtype.t
  | `Archive of Pt2_archive.error ]

(* A RESOURCE rejection, deliberately not one of [malformed]'s rows: the graph
   can be perfectly well formed and still ask for more outputs than the engine
   will build. That distinction is load-bearing at the Model Explorer boundary,
   where every [malformed] row is [Fatal] ("our bug") and this one has to be
   [Unavailable Over_limit] ("your model is too big"). The row is
   [Shape_error]'s, reused rather than restated, so the two spellings a caller
   can meet -- this one and [`Build (`Output_count_over_limit _)] -- carry the
   same payload. *)
type error =
  [ `Unsupported_input of unsupported_input
  | `Unsupported_operator of string
  | `Output_count_over_limit of Shape_error.Output_count.t
  | malformed
  | `Tensor_bridge of tensor_bridge
  | `Materialize of Const_ssa_materialize.error
  | `Eval of Eval_direct.error
  | `Build of Graph_builder.error
  | `Provenance of Pt2_native_graph.error
  | `Transform of Pass.error
  | `Verify of Map_verify.error
  | `Lens of Pt2_native_graph.lens_error ]

let pp_arg_kind ppf : arg_kind -> unit = function
  | `Tensor -> Fmt.string ppf "a tensor"
  | `Optional_tensor -> Fmt.string ppf "an optional tensor"
  | `Int_list -> Fmt.string ppf "an int list"
  | `Float_list -> Fmt.string ppf "a float list"
  | `Int -> Fmt.string ppf "an int"
  | `Int_opt -> Fmt.string ppf "an optional int"
  | `Bool -> Fmt.string ppf "a bool"
  | `Float -> Fmt.string ppf "a float"
  | `Scalar -> Fmt.string ppf "a scalar"
  | `Optional_scalar -> Fmt.string ppf "an optional scalar"
  | `String -> Fmt.string ppf "a string"
  | `Tensor_or_scalar -> Fmt.string ppf "a tensor or scalar"
  | `Tensor_list -> Fmt.string ppf "a tensor list"
  | `Memory_format_opt -> Fmt.string ppf "an optional memory format"

let pp_metadata_role ppf : metadata_role -> unit = function
  | `Tensor -> Fmt.string ppf "tensor"
  | `Convolution_weight -> Fmt.string ppf "convolution weight"
  | `Conv2d_weight -> Fmt.string ppf "conv2d weight"
  | `Conv2d_padding_weight -> Fmt.string ppf "conv2d padding weight"
  | `Linear_weight -> Fmt.string ppf "linear weight"
  | `Conv2d_bias -> Fmt.string ppf "conv2d bias"
  | `Conv2d_padding_bias -> Fmt.string ppf "conv2d padding bias"
  | `Convolution_bias -> Fmt.string ppf "convolution bias"
  | `Linear_bias -> Fmt.string ppf "linear bias"
  | `Rms_norm_input -> Fmt.string ppf "rms_norm input"
  | `Rms_norm_weight -> Fmt.string ppf "rms_norm weight"
  | `Layer_norm_input -> Fmt.string ppf "layer_norm input"
  | `Layer_norm_weight -> Fmt.string ppf "layer_norm weight"
  | `Layer_norm_bias -> Fmt.string ppf "layer_norm bias"
  | `Group_norm_weight -> Fmt.string ppf "group_norm weight"
  | `Group_norm_bias -> Fmt.string ppf "group_norm bias"
  | `Amax_input -> Fmt.string ppf "amax input"
  | `Mean_input -> Fmt.string ppf "mean input"
  | `Permute_input -> Fmt.string ppf "permute input"
  | `Transpose_input -> Fmt.string ppf "transpose input"
  | `Pad_input -> Fmt.string ppf "pad input"
  | `Slice_input -> Fmt.string ppf "slice input"
  | `Unbind_input -> Fmt.string ppf "unbind input"
  | `Split_with_sizes_input -> Fmt.string ppf "split_with_sizes input"
  | `Select_input -> Fmt.string ppf "select input"
  | `Unsqueeze_input -> Fmt.string ppf "unsqueeze input"
  | `Concat_input -> Fmt.string ppf "concat input"
  | `Stack_input -> Fmt.string ppf "stack input"
  | `Addmm_weight -> Fmt.string ppf "addmm weight"
  | `Sdpa_query -> Fmt.string ppf "sdpa query"
  | `Sdpa_key -> Fmt.string ppf "sdpa key"
  | `Sdpa_value -> Fmt.string ppf "sdpa value"
  | `Sdpa_mask -> Fmt.string ppf "sdpa attn_mask"
  | `Adaptive_avg_pool2d_input -> Fmt.string ppf "adaptive_avg_pool2d input"
  | `Vector_norm_input -> Fmt.string ppf "vector_norm input"
  | `Upsample_bilinear2d_input -> Fmt.string ppf "upsample_bilinear2d input"

let pp_hw_param ppf : hw_param -> unit = function
  | `Stride -> Fmt.string ppf "stride"
  | `Padding -> Fmt.string ppf "padding"
  | `Output_padding -> Fmt.string ppf "output_padding"
  | `Dilation -> Fmt.string ppf "dilation"
  | `Kernel_size -> Fmt.string ppf "kernel_size"
  | `Output_size -> Fmt.string ppf "output_size"

let pp_malformed ppf : [< malformed ] -> unit = function
  | `Missing_arg { Missing_arg.op; arg } ->
      Fmt.pf ppf "%s: missing argument %S" op arg
  | `Wrong_arg_kind { Wrong_arg_kind.op; arg; expected } ->
      Fmt.pf ppf "%s.%s is not %a" op arg pp_arg_kind expected
  | `Unresolved_sym_arg { Unresolved_sym_arg.op; arg; symbol } ->
      Fmt.pf ppf "%s.%s is the unresolved symbol %S" op arg symbol
  | `Missing_metadata { Missing_metadata.ssa; role } ->
      Fmt.pf ppf "no %a metadata for %S" pp_metadata_role role ssa
  | `Bad_dimension { Bad_dimension.tensor; fault } -> (
      match fault with
      | `Negative i -> Fmt.pf ppf "%s has negative dimension %d" tensor i
      | `Zero -> Fmt.pf ppf "%s has a zero-length dimension" tensor
      | `Symbolic -> Fmt.pf ppf "%s has a symbolic dimension" tensor
      | `Rank_over_six -> Fmt.pf ppf "%s has rank greater than six" tensor
      | `Expected_rank { Expected_rank.expected; got } ->
          Fmt.pf ppf "%s is rank %d, expected %d" tensor got expected
      | `Over_max_extent n ->
          Fmt.pf ppf "%s has extent %Ld, over the engine maximum of %Ld" tensor
            n Kernel.Limits.Hard.extent)
  | `Axis_out_of_range { Axis_out_of_range.axis; rank } ->
      Fmt.pf ppf "invalid dimension %d for rank %d" axis rank
  | `Bad_arity { Bad_arity.param; got } ->
      Fmt.pf ppf "%a must have %s, got %d" pp_hw_param param
        (match param with
        | `Output_size -> "exactly two values"
        | _ -> "one or two values")
        got
  | `Adaptive_pool_rank { Adaptive_pool_rank.tensor; got } ->
      Fmt.pf ppf "%s must be rank-3 (CHW) or rank-4 (NCHW), got rank-%d" tensor
        got
  | `Bad_config e -> Op_config.Bad.pp ppf e
  | `Output_arity { Output_arity.op; serialized; derived } ->
      Fmt.pf ppf "%s declares %d outputs but produces %d" op serialized derived
  | `Unsupported_option { Unsupported_option.op; option } -> (
      match option with
      | `Alpha a -> Fmt.pf ppf "%s: alpha=%g is not supported (only 1)" op a
      | `Memory_format mf ->
          Fmt.pf ppf "%s: memory_format=%s is not supported" op
            (match mf with
            | `Channels_last -> "channels_last"
            | `Channels_last_3d -> "channels_last_3d"
            | `Unknown -> "unknown")
      | `Dilation d ->
          Fmt.pf ppf "%s: dilation=[%a] is not supported (only 1)" op
            Fmt.(list ~sep:(any ",") int)
            d
      | `Approximate a ->
          Fmt.pf ppf
            "%s: approximate=%S is not supported (only \"none\" or \"tanh\")" op
            a
      | `Dtype -> Fmt.pf ppf "%s: dtype is not supported" op
      | `Vector_norm_ord o ->
          Fmt.pf ppf "%s: ord=%g is not supported (only 2)" op o)
  | `Unsupported_padding_mode s ->
      Fmt.pf ppf "padding mode %S is neither \"valid\" nor \"same\"" s
  | `Bad_pad_list e -> Pad.Pad.Bad_pad_list.pp ppf e
  | `Normalized_rank { Normalized_rank.op; rank; got } ->
      Fmt.pf ppf
        "%a: normalized_shape has %d entries, outside [1, %d] for this rank"
        Norm.Target.pp op got rank
  | `Normalized_shape { Normalized_shape.op; expected; got } ->
      let ints = Fmt.(list ~sep:(any ",") int) in
      Fmt.pf ppf
        "%a: normalized_shape [%a] does not match the input's trailing extents \
         [%a]"
        Norm.Target.pp op ints got ints expected
  | `Live_layer_norm_stats { Live_layer_norm_stats.op; stat; ssa } ->
      Fmt.pf ppf "%s: %s output %S is read, and this graph does not have it" op
        (match stat with `Mean -> "mean" | `Rstd -> "rstd")
        ssa
  | `Non_tensor_node_output op -> Fmt.pf ppf "%s has a non-tensor output" op
  | `Non_tensor_graph_output -> Fmt.string ppf "non-tensor graph output"
  | `Undefined_ssa name -> Fmt.pf ppf "SSA tensor %S is not defined" name
  | `Output_not_evaluated id ->
      Fmt.pf ppf "native output %a was not evaluated" Tensor_id.pp id
  | `Bad_slice { Bad_slice.start; stop; step; fault } -> (
      let bound = Fmt.(option ~none:(any "none") int) in
      match fault with
      | `Aten_shape e ->
          Fmt.pf ppf "slice [%a, %a) step %d: %a" bound start bound stop step
            Aten_shape.pp_error e)
  | `Bad_view { Bad_view.size; fault } -> (
      let ints = Fmt.(list ~sep:(any ", ") int) in
      match fault with
      | `Aten_shape e ->
          Fmt.pf ppf "view size [%a]: %a" ints size Aten_shape.pp_error e
      | `Numel_over_limit e ->
          Fmt.pf ppf "view size [%a]: %a" ints size Vec6.Numel_bound.pp e)
  | `Bad_select { Bad_select.index; fault } -> (
      match fault with
      | `Aten_shape e ->
          Fmt.pf ppf "select index %d: %a" index Aten_shape.pp_error e)
  | `Sdpa_reject e -> Attention.Sdpa.Reject.pp ppf e
  | `Concat_no_tensors op -> Fmt.pf ppf "%s: at least one tensor is required" op
  | `Concat_rank_mismatch { Concat_rank_mismatch.op; first; other } ->
      Fmt.pf ppf "%s: every tensor must have the same rank: %d vs %d" op first
        other
  | `Bad_upsample_size { Bad_upsample_size.op; fault } -> (
      match fault with
      | Bad_upsample_size.Neither ->
          Fmt.pf ppf
            "%s: exactly one of output_size or scale_factors must be given" op
      | Bad_upsample_size.Both ->
          Fmt.pf ppf "%s: output_size and scale_factors are mutually exclusive"
            op
      | Bad_upsample_size.Bad_scale_arity got ->
          Fmt.pf ppf "%s: scale_factors must have exactly 2 elements, got %d" op
            got)

let pp_tensor_bridge ppf : [< tensor_bridge ] -> unit = function
  | #malformed as e -> pp_malformed ppf e
  | `Rank_mismatch { Rank_mismatch.sizes; strides } ->
      Fmt.pf ppf "%d sizes but %d strides" sizes strides
  | `Storage_index_overflow ->
      Fmt.string ppf "storage index range overflows a 64-bit integer"
  | `Storage_out_of_range { Storage_range.lo; hi; data_bytes } ->
      Fmt.pf ppf "storage index range [%Ld, %Ld] is outside %d bytes of data" lo
        hi data_bytes
  | `Materialize_failed m -> Fmt.pf ppf "materializing the tensor: %s" m
  | `Unsupported_dtype d ->
      Fmt.pf ppf "only float32 is supported, got %s" (Pt2_dtype.to_string d)
  | `Archive e -> Pt2_archive.pp_error ppf e

let pp_error ppf : [< error ] -> unit = function
  | `Unsupported_input `Non_tensor ->
      Fmt.string ppf "unsupported PT2 input: not a tensor"
  | `Unsupported_input (`Not_exactly_one_user_input n) ->
      Fmt.pf ppf "unsupported PT2 input: expected one user input, got %d" n
  | `Unsupported_operator s -> Fmt.pf ppf "unsupported PT2 operator: %s" s
  | `Output_count_over_limit e ->
      Fmt.pf ppf "PT2 graph over limit: %a" Shape_error.Output_count.pp e
  | #malformed as e -> Fmt.pf ppf "malformed PT2 graph: %a" pp_malformed e
  | `Tensor_bridge e -> Fmt.pf ppf "PT2 tensor bridge: %a" pp_tensor_bridge e
  | `Materialize e -> Const_ssa_materialize.pp_error ppf e
  | `Eval e -> Eval_direct.pp_error ppf e
  | `Build e -> Graph_builder.pp_error ppf e
  | `Provenance e -> Pt2_native_graph.pp_error ppf e
  | `Transform e -> Pass.pp_error ppf e
  | `Verify e -> Map_verify.pp_error ppf e
  | `Lens e -> Pt2_native_graph.pp_lens_error ppf e
