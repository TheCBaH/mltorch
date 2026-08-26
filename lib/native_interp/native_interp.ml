open Pytorch_types
open Schema_runtime
module Tensor_id = Graph_ir.Tensor_id
module Node_id = Graph_ir.Node_id

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
  | `Tensor_or_scalar
  | `Tensor_list ]

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
  | `Vector_norm_input ]

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
  | `Memory_format
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
  | `Concat_rank_mismatch of Concat_rank_mismatch.t ]

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
  | `Int -> Fmt.string ppf "an int"
  | `Int_opt -> Fmt.string ppf "an optional int"
  | `Bool -> Fmt.string ppf "a bool"
  | `Float -> Fmt.string ppf "a float"
  | `Scalar -> Fmt.string ppf "a scalar"
  | `Optional_scalar -> Fmt.string ppf "an optional scalar"
  | `String -> Fmt.string ppf "a string"
  | `Tensor_or_scalar -> Fmt.string ppf "a tensor or scalar"
  | `Tensor_list -> Fmt.string ppf "a tensor list"

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
      | `Memory_format -> Fmt.pf ppf "%s: memory_format is not supported" op
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

type hooks =
  | Hooks : {
      on_start : Pt2_native_graph.t -> Graph_ir.node -> 'a;
      on_end : Pt2_native_graph.t -> Graph_ir.node -> 'a -> unit;
    }
      -> hooks

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

(* Internal control flow only — the .mli exposes [error] and nothing else. The
   lowering walk is deeply recursive and threading a result through every arm
   would rewrite it, so it exits through [Err.Escape] instead: one token per
   [with_escape] call, threaded to every helper that can detect a fault.

   It replaces a private [exception Lower_error of error] carrying the BARE row.
   That lost the [Err.Error.t] across this module's own boundary — the catch
   rebuilt the wrapper with [Err.fail], so every malformed graph was reported as
   detected at the catch site rather than where the fault was found — and it let
   [tensor_of_pt2]'s re-labelled row escape uncaught past an [Err.t]
   signature. [throw] records [Detect] where the fault is, and [with_escape]
   catches by construction. *)
let malformed esc (e : malformed) = Err.Escape.throw esc (e :> error)

let shape_of_sizes esc name sizes =
  let dims =
    List.map
      (function
        | SymInt.Int i when i >= 1 -> i
        | SymInt.Int 0 ->
            malformed esc (`Bad_dimension { tensor = name; fault = `Zero })
        | SymInt.Int i ->
            malformed esc
              (`Bad_dimension { tensor = name; fault = `Negative i })
        | SymInt.Expr _ ->
            malformed esc (`Bad_dimension { tensor = name; fault = `Symbolic }))
      sizes
  in
  match List.rev dims with
  | [ c; w; h; d; t; n ] -> Vec6.shape ~n ~t ~d ~h ~w ~c
  | [ c; w; h; d; t ] -> Vec6.shape ~n:1 ~t ~d ~h ~w ~c
  | [ c; w; h; d ] -> Vec6.shape ~n:1 ~t:1 ~d ~h ~w ~c
  | [ c; w; h ] -> Vec6.shape ~n:1 ~t:1 ~d:1 ~h ~w ~c
  | [ c; w ] -> Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w ~c
  | c :: [] -> Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c
  | [] -> Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1
  | _ ->
      malformed esc (`Bad_dimension { tensor = name; fault = `Rank_over_six })

let tensor_shape esc (graph : Pytorch_types.Graph.t) name =
  match String_map.find_opt name graph.tensor_values with
  | Some meta -> shape_of_sizes esc name meta.TensorMeta.sizes
  | None -> malformed esc (`Missing_metadata { ssa = name; role = `Tensor })

let find_arg esc (node : Pytorch_types.Node.t) name =
  match
    List.find_opt (fun (a : NamedArgument.t) -> a.name = name) node.Node.inputs
  with
  | Some a -> a.arg
  | None -> malformed esc (`Missing_arg { op = node.target; arg = name })

let tensor_name esc (node : Pytorch_types.Node.t) name =
  match find_arg esc node name with
  | Argument.Tensor t -> t.TensorArgument.name
  | _ ->
      malformed esc
        (`Wrong_arg_kind { op = node.target; arg = name; expected = `Tensor })

(* [cat.default]/[stack.default]'s [tensors] argument, the first Tensor[]-typed
   ARGUMENT this module decodes (every earlier [Argument.Tensors] use is on
   the OUTPUT side, e.g. [output_names]). *)
let tensor_names_arg esc (node : Pytorch_types.Node.t) name =
  match find_arg esc node name with
  | Argument.Tensors ts -> List.map (fun (t : TensorArgument.t) -> t.name) ts
  | _ ->
      malformed esc
        (`Wrong_arg_kind
           { op = node.target; arg = name; expected = `Tensor_list })

(* [~absent_ok] distinguishes an argument that is PRESENT and None from one not
   in the node's input list at all. The schema default for every optional tensor
   here is None, and [Op_bridge] already reads omission that way
   ([optional_tensor_present]), so an exact target that refused it would
   disagree with the other importer about the same node — which is the property
   the two paths exist to cross-check.

   Defaulted to [false] so the arms that predate this keep the behaviour their
   goldens pin; they are fed only by real exports, which serialise every
   argument explicitly, and each can revisit it in its own row. *)
let optional_tensor_name ?(absent_ok = false) esc (node : Pytorch_types.Node.t)
    name =
  match
    List.find_opt (fun (a : NamedArgument.t) -> a.name = name) node.Node.inputs
  with
  | None when absent_ok -> None
  | _ -> (
      match find_arg esc node name with
      | Argument.Tensor t -> Some t.TensorArgument.name
      | Argument.None _ -> None
      | Argument.Optional_tensor (OptionalTensorArgument.Tensor t) ->
          Some t.TensorArgument.name
      | Argument.Optional_tensor (OptionalTensorArgument.None _) -> None
      | _ ->
          malformed esc
            (`Wrong_arg_kind
               { op = node.target; arg = name; expected = `Optional_tensor }))

let sym_int_value esc (node : Pytorch_types.Node.t) name = function
  | SymIntArgument.Int i -> i
  | SymIntArgument.Name symbol ->
      malformed esc
        (`Unresolved_sym_arg
           { Unresolved_sym_arg.op = node.target; arg = name; symbol })

let ints_arg esc ?(default = []) (node : Pytorch_types.Node.t) name =
  match
    List.find_opt (fun (a : NamedArgument.t) -> a.name = name) node.Node.inputs
  with
  | None -> default
  | Some { arg = Argument.Ints xs; _ } -> xs
  | Some { arg = Argument.Sym_ints xs; _ } ->
      List.map (sym_int_value esc node name) xs
  | Some { arg = Argument.None _; _ } -> default
  | Some _ ->
      malformed esc
        (`Wrong_arg_kind { op = node.target; arg = name; expected = `Int_list })

(* A resolved [SymInt] is accepted and a NAMED one is refused as an unresolved
   symbol -- the same rule [Interp_decode.sym_int_value] applies on the ATen
   path, and the same one [Bad_dimension]'s [`Symbolic] fault already applied to
   tensor METADATA here. Before [slice.Tensor] no bound op had a [SymInt]
   argument, so an [Argument.Sym_int] reached [`Wrong_arg_kind] whatever it
   carried: a resolved bound was refused for the wrong reason and an unresolved
   one for a reason that did not name the symbol. *)
let int_arg esc ?(default = 0) (node : Pytorch_types.Node.t) name =
  match
    List.find_opt (fun (a : NamedArgument.t) -> a.name = name) node.Node.inputs
  with
  | None -> default
  | Some { arg = Argument.Int i; _ } -> i
  | Some { arg = Argument.Sym_int sv; _ } -> sym_int_value esc node name sv
  | Some _ ->
      malformed esc
        (`Wrong_arg_kind { op = node.target; arg = name; expected = `Int })

(* An [int?] whose ABSENCE is a distinguishable answer, as [float_opt_arg_opt]
   is for [pad]'s fill: [slice]'s [start]/[end] default to the whole axis, and
   [Aten_shape.resolve_slice] is what knows that. An explicit [Argument.None] is
   the same as an absent argument, which is the schema's own default
   ([SymInt? start=None]) and not a guess. *)
let int_opt_arg_opt esc (node : Pytorch_types.Node.t) name =
  match
    List.find_opt (fun (a : NamedArgument.t) -> a.name = name) node.Node.inputs
  with
  | None -> None
  | Some { arg = Argument.Int i; _ } -> Some i
  | Some { arg = Argument.Sym_int sv; _ } ->
      Some (sym_int_value esc node name sv)
  | Some { arg = Argument.None _; _ } -> None
  | Some _ ->
      malformed esc
        (`Wrong_arg_kind { op = node.target; arg = name; expected = `Int_opt })

let string_arg esc ~default (node : Pytorch_types.Node.t) name =
  match
    List.find_opt (fun (a : NamedArgument.t) -> a.name = name) node.Node.inputs
  with
  | None -> default
  | Some { arg = Argument.String s; _ } -> s
  | Some _ ->
      malformed esc
        (`Wrong_arg_kind { op = node.target; arg = name; expected = `String })

let bool_arg esc ?(default = false) (node : Pytorch_types.Node.t) name =
  match
    List.find_opt (fun (a : NamedArgument.t) -> a.name = name) node.Node.inputs
  with
  | None -> default
  | Some { arg = Argument.Bool b; _ } -> b
  | Some _ ->
      malformed esc
        (`Wrong_arg_kind { op = node.target; arg = name; expected = `Bool })

(* A REQUIRED [float]: no default, so omission is [`Missing_arg] and an explicit
   none is [`Wrong_arg_kind]. Carrying a [?(default = 0.)] here was the same
   mistake in a quieter form than the explicit-none one -- it made "required"
   mean "defaults to zero", so a batch-norm node that simply omitted [eps]
   computed with an epsilon of zero, while [Op_bridge]'s decoder reported the
   argument missing. Anything with a real schema default passes it explicitly. *)
let float_arg esc ?default (node : Pytorch_types.Node.t) name =
  match
    List.find_opt (fun (a : NamedArgument.t) -> a.name = name) node.Node.inputs
  with
  | None -> (
      match default with
      | Some d -> d
      | None -> malformed esc (`Missing_arg { op = node.target; arg = name }))
  | Some { arg = Argument.Float f; _ } -> f
  | Some _ ->
      malformed esc
        (`Wrong_arg_kind { op = node.target; arg = name; expected = `Float })

(* A [float?]. Omission and an explicit none are the SAME REQUEST -- both are
   serialized, since the generated op-spec path writes [Float_opt None] out as
   [Argument.None] -- and anything else is still refused.

   Separate from [float_arg] rather than an arm added to it. Accepting an
   explicit none for every caller made
   [_native_batch_norm_legit_no_training.default], whose schema has a REQUIRED
   [float eps], silently read a null epsilon as 0. -- a different op, computed
   under the right name. The optionality belongs to the argument, so it belongs
   at the call site. *)
let float_opt_arg esc ~default (node : Pytorch_types.Node.t) name =
  match
    List.find_opt (fun (a : NamedArgument.t) -> a.name = name) node.Node.inputs
  with
  | None -> default
  | Some { arg = Argument.Float f; _ } -> f
  | Some { arg = Argument.None _; _ } -> default
  | Some _ ->
      malformed esc
        (`Wrong_arg_kind { op = node.target; arg = name; expected = `Float })

(* A [float?] whose ABSENCE is a distinguishable answer rather than a default:
   [aten.pad]'s [value] means 0.0 in constant mode and must be absent (or zero)
   in reflect, so collapsing the two here would erase the distinction the mode
   check needs. Contrast [float_opt_arg] above, which supplies a default because
   its callers have one. *)
let float_opt_arg_opt esc (node : Pytorch_types.Node.t) name =
  match
    List.find_opt (fun (a : NamedArgument.t) -> a.name = name) node.Node.inputs
  with
  | None -> None
  | Some { arg = Argument.Float f; _ } -> Some f
  | Some { arg = Argument.Int i; _ } -> Some (float_of_int i)
  | Some { arg = Argument.None _; _ } -> None
  | Some _ ->
      malformed esc
        (`Wrong_arg_kind { op = node.target; arg = name; expected = `Float })

(* A schema [Scalar] argument crosses as either an Int or a Float — clamp's
   bounds arrive as `as_int` in MobileNet-v3 and hardtanh's as `as_float` in v2,
   for the same kind of parameter. Mirrors [Interp_decode.scalar_arg] /
   [scalar_opt_arg], which the ATen path decodes with. *)
let scalar_arg esc ~default (node : Pytorch_types.Node.t) name =
  match
    List.find_opt (fun (a : NamedArgument.t) -> a.name = name) node.Node.inputs
  with
  | None -> default
  | Some { arg = Argument.Int i; _ } -> float_of_int i
  | Some { arg = Argument.Float f; _ } -> f
  | Some { arg = Argument.None _; _ } -> default
  | Some _ ->
      malformed esc
        (`Wrong_arg_kind { op = node.target; arg = name; expected = `Scalar })

let scalar_opt_arg esc (node : Pytorch_types.Node.t) name =
  match
    List.find_opt (fun (a : NamedArgument.t) -> a.name = name) node.Node.inputs
  with
  | None -> None
  | Some { arg = Argument.Int i; _ } -> Some (float_of_int i)
  | Some { arg = Argument.Float f; _ } -> Some f
  | Some { arg = Argument.None _; _ } -> None
  | Some _ ->
      malformed esc
        (`Wrong_arg_kind
           { op = node.target; arg = name; expected = `Optional_scalar })

(* A REQUIRED schema [Scalar]: no default, so omission is [`Missing_arg] and an
   explicit none (or any other kind) is [`Wrong_arg_kind]. [mul.Scalar]'s
   [other] has no schema default, so reusing [scalar_arg]'s [~default] -- which
   treats both omission and an explicit none as the default -- would silently
   read a missing multiplier as the caller's placeholder value; the same
   mistake [float_arg]'s comment documents for batch-norm's [eps]. *)
let required_scalar_arg esc (node : Pytorch_types.Node.t) name =
  match
    List.find_opt (fun (a : NamedArgument.t) -> a.name = name) node.Node.inputs
  with
  | None -> malformed esc (`Missing_arg { op = node.target; arg = name })
  | Some { arg = Argument.Int i; _ } -> float_of_int i
  | Some { arg = Argument.Float f; _ } -> f
  | Some _ ->
      malformed esc
        (`Wrong_arg_kind { op = node.target; arg = name; expected = `Scalar })

(* [add.Tensor]/[sub.Tensor] carry `*, Scalar alpha=1` and compute
   [self + alpha * other]. Nothing in this model zoo serialises a non-default
   alpha, so it is not implemented — but it must not be silently dropped either,
   since that would quietly compute the wrong thing. Reject instead. *)
let reject_alpha esc (node : Pytorch_types.Node.t) =
  match scalar_opt_arg esc node "alpha" with
  | None -> ()
  | Some a when Float.equal a 1. -> ()
  | Some a ->
      malformed esc
        (`Unsupported_option { op = node.target; option = `Alpha a })

(* [clone] with a [memory_format] asks for a layout change this op does not
   perform. The native IR has one layout per shape, so honouring the request is
   impossible and ignoring it would misreport what was computed. *)
let reject_memory_format esc (node : Pytorch_types.Node.t) =
  match
    List.find_opt
      (fun (a : NamedArgument.t) -> a.name = "memory_format")
      node.Node.inputs
  with
  | None | Some { arg = Argument.None _; _ } -> ()
  | Some _ ->
      malformed esc
        (`Unsupported_option { op = node.target; option = `Memory_format })

(* [linalg_vector_norm.default]'s [dtype] casts before reducing; the native IR
   has no dtype-conversion op, so honouring the request is impossible and
   ignoring it would misreport what was computed -- same reasoning as
   [reject_memory_format] above, for a different argument. *)
let reject_dtype esc (node : Pytorch_types.Node.t) =
  match
    List.find_opt
      (fun (a : NamedArgument.t) -> a.name = "dtype")
      node.Node.inputs
  with
  | None | Some { arg = Argument.None _; _ } -> ()
  | Some _ ->
      malformed esc (`Unsupported_option { op = node.target; option = `Dtype })

(* A `Tensor[]` return is ONE output of kind [Argument.Tensors] holding every
   result name in order — a different shape from a fixed tuple, whose elements
   are separate [Argument.Tensor] entries. Both flatten to a name list here.

   THE CEILING LIVES IN THIS FUNCTION, not in the operator arm that wants it.
   [lower_node] calls [materialized_output_names] before it calls [lower_op], so
   an arm-local preflight would run after the first [List.map] had already built
   a list sized by model data. [take_bounded] therefore counts as it walks and
   stops AT the limit, never learning the real length — which is exactly why
   [Shape_error.Output_count] distinguishes [At_least] from [Exact].

   The rule is [>= limit], matching [Kernel.Limits.create] and
   [Split.Unbind.output_shapes]: 4095 names are accepted, 4096 are not. *)
let output_limit = Kernel.Limits.Hard.outputs

(* The rule is [>= output_limit], so the allowance is one less than the limit:
   4095 names are accepted, the 4096th is refused. Carried as a remaining
   budget rather than a running total so the traversal can stop without ever
   holding a length. *)
let output_allowance = output_limit - 1

let over_limit esc =
  Err.Escape.throw esc
    (`Output_count_over_limit
       {
         Shape_error.Output_count.limit = output_limit;
         observed = Shape_error.Output_count.At_least output_limit;
       })

(* Prepend [xs]'s names to [acc] while [budget] lasts; throws on the element
   that would reach the limit. Counting rather than [List.length]-then-check is
   the point: the list may be arbitrarily long, and this never walks past the
   ceiling. *)
let rec take_bounded esc ~budget acc = function
  | [] -> (acc, budget)
  | (t : TensorArgument.t) :: rest ->
      if budget <= 0 then over_limit esc
      else
        take_bounded esc ~budget:(budget - 1)
          (t.TensorArgument.name :: acc)
          rest

(* Flatten one argument's tensor names, threading the remaining budget so that
   SEVERAL list-valued arguments are bounded in aggregate rather than each on
   its own — several individually legal lists can exceed the ceiling together. *)
let flatten_output esc ~on_bad_kind ~budget acc (a : Argument.t) =
  match a with
  | Argument.Tensor t ->
      if budget <= 0 then over_limit esc
      else (t.TensorArgument.name :: acc, budget - 1)
  | Argument.Tensors ts -> take_bounded esc ~budget acc ts
  | _ -> on_bad_kind ()

let flatten_outputs esc ~on_bad_kind args =
  let names, _ =
    List.fold_left
      (fun (acc, budget) a -> flatten_output esc ~on_bad_kind ~budget acc a)
      ([], output_allowance) args
  in
  List.rev names

let output_names esc (node : Pytorch_types.Node.t) =
  flatten_outputs esc
    ~on_bad_kind:(fun () -> malformed esc (`Non_tensor_node_output node.target))
    node.outputs

let is_nontrivial_node (node : Pytorch_types.Node.t) =
  match node.target with
  | "torch.ops.aten.conv2d.default" | "torch.ops.aten.conv2d.padding"
  | "torch.ops.aten.convolution.default" | "torch.ops.aten.linear.default"
  | "torch.ops.aten._native_batch_norm_legit_no_training.default"
  | "torch.ops.aten.max_pool2d.default"
  | "torch.ops.aten.adaptive_avg_pool2d.default"
  | "torch.ops.aten.max_pool2d_with_indices.default"
  | "torch.ops.aten.rms_norm.default" | "torch.ops.aten.layer_norm.default"
  | "torch.ops.aten.native_layer_norm.default" | "torch.ops.aten.addmm.default"
  | "torch.ops.aten.scaled_dot_product_attention.default" ->
      true
  | _ -> false

let materialized_output_names esc (node : Pytorch_types.Node.t) =
  match node.target with
  | "torch.ops.aten._native_batch_norm_legit_no_training.default"
  | "torch.ops.aten.max_pool2d_with_indices.default"
  (* Third entry, and the first whose dropped outputs are NOT empty: they are
     real f32 tensors that happen to be dead in every occurrence the corpus
     contains. Dropping them here is what makes the [`Live_layer_norm_stats]
     check below load-bearing rather than decorative. *)
  | "torch.ops.aten.native_layer_norm.default" ->
      [ List.hd (output_names esc node) ]
  | _ -> output_names esc node

let hw2 esc param = function
  | [ h; w ] -> (h, w)
  | [ x ] -> (x, x)
  | xs -> malformed esc (`Bad_arity { param; got = List.length xs })

(* [Op_config.Pos]/[Nonneg]/[Dim.extent] assert a TRUSTED precondition and
   raise [Invalid_argument] when it fails. Every value below is decoded from the
   model, so none of them may reach those constructors unguarded: the raise
   crosses the [Err.Escape] frame and leaves [lower] as an exception, which is
   what [malformed_test.ml]'s three config witnesses pinned.

   These are the ONLY approved route from a decoded argument to a guarded
   config type in this module. [Dim.extent_checked] already existed for exactly
   this ([dim.mli]: "the validated form for an untrusted size"); the other two
   have no checked form, so the test is written out here. *)
let pos esc ~op ~param n =
  match Op_config.Bad.pos ~op ~param n with
  | Ok v -> v
  | Error e -> malformed esc (`Bad_config e)

let nonneg esc ~op ~param n =
  match Op_config.Bad.nonneg ~op ~param n with
  | Ok v -> v
  | Error e -> malformed esc (`Bad_config e)

let extent esc ~op ~param n =
  match Dim.extent_checked n with
  | Ok e -> e
  | Error _ ->
      malformed esc
        (`Bad_config { Op_config.Bad.op; param; fault = `Not_positive n })

(* The same asserting constructor reached from tensor METADATA rather than from
   an op-configuration field, so it gets the row that already describes that:
   [Bad_dimension]'s [`Zero] and [`Negative] faults, which the module documents
   as distinct because only [`Zero] arrives from real models. *)
let dim_extent esc ~tensor n =
  match Dim.extent_checked n with
  | Ok e -> e
  | Error _ ->
      malformed esc
        (`Bad_dimension
           { tensor; fault = (if n = 0 then `Zero else `Negative n) })

let pos_hw esc ~op ~param (h, w) =
  { Op_config.Hw.h = pos esc ~op ~param h; w = pos esc ~op ~param w }

let nonneg_hw esc ~op ~param (h, w) =
  { Op_config.Hw.h = nonneg esc ~op ~param h; w = nonneg esc ~op ~param w }

let env_find esc env name =
  match String_map.find_opt name env with
  | Some x -> x
  | None -> malformed esc (`Undefined_ssa name)

let add_env env names ids =
  (* An INVARIANT of this module, not a fact about the model: [names] comes
     from [materialized_output_names] and [ids] from the op call just made, so a
     mismatch is a defect here. It must not reach the caller dressed as a
     malformed graph. Same treatment as [Graph_ir.Index.assert_matches]. *)
  if List.compare_lengths names ids <> 0 then
    invalid_arg
      "Native_interp.add_env: output arity does not match the ids produced";
  List.fold_left2 (fun e name id -> String_map.add name id e) env names ids

let perm_nchw_to_nhwc =
  let open Axis in
  [ (N, N); (T, T); (D, D); (H, W); (W, C); (C, H) ]

let perm_nhwc_to_nchw =
  let open Axis in
  [ (N, N); (T, T); (D, D); (H, C); (W, H); (C, W) ]

let perm_oihw_to_conv_weight =
  let open Axis in
  [ (N, D); (T, T); (D, N); (H, W); (W, C); (C, H) ]

(* Rank-2 addmm weight [In,Out] (W=In, C=Out) -> native [N=Out, C=In]. *)
let perm_addmm_weight =
  let open Axis in
  [ (N, C); (T, T); (D, D); (H, H); (W, N); (C, W) ]

(* Rank-2 linear weight [Out,In] (W=Out, C=In) -> native [N=Out, C=In]. NOT the
   permutation above, and not a rename of it: `addmm`'s [mat2] is the transpose
   of `linear`'s [weight], so an arm that reused one for the other would build a
   weight whose output and input axes are swapped. Both spellings exist in
   [Op_bridge] (op_bridge.ml:221,227) for the same reason. *)
let perm_linear_weight =
  let open Axis in
  [ (N, W); (T, T); (D, D); (H, H); (W, N); (C, C) ]

(* The [tensor_values] lookup, open-coded at five sites with the same three
   steps and a different role label each. Three functions rather than one
   because the sites want different depths: [mean.dim], [permute.default] and
   [unbind.int] need only the RANK, which a symbolic dimension does not
   prevent, while a conv weight needs the extents themselves.

   [role] stays a parameter so each caller keeps its own diagnostic. Sharing one
   role across two arms would make the row ambiguous about which one failed,
   which is the property that made these worth typing in the first place. *)
let tensor_meta esc (graph : Pytorch_types.Graph.t) ~ssa ~role =
  match String_map.find_opt ssa graph.tensor_values with
  | Some x -> x
  | None -> malformed esc (`Missing_metadata { ssa; role })

let meta_rank (meta : TensorMeta.t) = List.length meta.TensorMeta.sizes

(* [shape_of_sizes] RIGHT-ALIGNS a declared size list into the six-axis frame,
   so [C] and [1,C] land on exactly the same extents. [Graph_shape]'s operand
   check compares those frames and therefore cannot tell the two apart -- but
   ATen can, and refuses a bias that is not 1-D. The declared RANK exists only
   on this side of the conversion, so no shared native rule can cover it and
   each importer has to check its own. *)
let require_rank esc (graph : Pytorch_types.Graph.t) ~ssa ~role ~expected =
  let got = meta_rank (tensor_meta esc graph ~ssa ~role) in
  if got <> expected then
    malformed esc
      (`Bad_dimension { tensor = ssa; fault = `Expected_rank { expected; got } })

let static_sizes esc ~tensor (meta : TensorMeta.t) =
  List.map
    (function
      | SymInt.Int i -> i
      | SymInt.Expr _ ->
          malformed esc (`Bad_dimension { tensor; fault = `Symbolic }))
    meta.TensorMeta.sizes

let sizes_rank_4 esc ~tensor = function
  | [ a; b; c; d ] -> (a, b, c, d)
  | sizes ->
      malformed esc
        (`Bad_dimension
           {
             tensor;
             fault = `Expected_rank { expected = 4; got = List.length sizes };
           })

let sizes_rank_2 esc ~tensor = function
  | [ a; b ] -> (a, b)
  | sizes ->
      malformed esc
        (`Bad_dimension
           {
             tensor;
             fault = `Expected_rank { expected = 2; got = List.length sizes };
           })

(* [Conv2d.params.in_channels] is the ACTIVATION's channel count: the weight's
   per-group input extent times the group count. Same rule as
   [Op_bridge.make_conv2d_params] and [Conv.Conv2d_padding.to_conv2d_params],
   restated here only because the importer reads serialized metadata where those
   read a live tensor.

   COMPUTED IN int64 AND BOUNDED BEFORE NARROWING, which is the whole point.
   js_of_ocaml's [int] is 32 bits and [Kernel.Limits.Hard.extent] is
   0x8000_0000, so two individually plausible factors can multiply past the
   representable range and WRAP to a small positive number — a silently wrong
   graph rather than a rejected one. Bounding the factors would not catch it:
   the product is its own quantity. [test/native_interp] runs under node
   ([modes best js]), so this is reachable and not a theoretical concern. *)
let conv_in_channels esc ~tensor ~cin ~groups =
  (* Each FACTOR is bounded before the multiplication, not only the product.
     Widening to [int64] does not by itself make the multiplication safe: these
     are raw decoded ints, and on a 63-bit-[int] backend two factors near
     2^62 overflow [Int64.mul] silently and land back inside the range the
     check below accepts. The bound is unobservable under js_of_ocaml, where a
     factor that large is not representable in a 32-bit [int] at all -- which is
     exactly why it cannot be left to the product check. *)
  let over n = Int64.of_int n >= Kernel.Limits.Hard.extent in
  if over cin || over groups then
    malformed esc
      (`Bad_dimension
         {
           tensor;
           fault =
             `Over_max_extent
               (if over cin then Int64.of_int cin else Int64.of_int groups);
         });
  let product = Int64.mul (Int64.of_int cin) (Int64.of_int groups) in
  if product >= Kernel.Limits.Hard.extent then
    malformed esc (`Bad_dimension { tensor; fault = `Over_max_extent product })
  else if product < 1L then
    malformed esc
      (`Bad_dimension
         { tensor; fault = (if product = 0L then `Zero else `Negative cin) })
  else Dim.extent (Int64.to_int product)

(* The exact `conv2d.default` overload, whose [params] record is NOT
   [Convolution]'s: per-axis windows carrying their own kernel extent, an
   activation channel count, and no transposed/output_padding fields. Sharing
   the metadata and H/W decoding with [conv_params] while keeping the two
   records apart is what lets the exact IR node survive to Native4D, which reads
   [Conv2d] directly. *)
let conv2d_params esc (graph : Pytorch_types.Graph.t)
    (node : Pytorch_types.Node.t) =
  let op = node.Node.target in
  let weight_name = tensor_name esc node "weight" in
  let sizes =
    static_sizes esc ~tensor:weight_name
      (tensor_meta esc graph ~ssa:weight_name ~role:`Conv2d_weight)
  in
  let _cout, cin, kh, kw = sizes_rank_4 esc ~tensor:weight_name sizes in
  let sh, sw = hw2 esc `Stride (ints_arg esc ~default:[ 1; 1 ] node "stride") in
  let ph, pw =
    hw2 esc `Padding (ints_arg esc ~default:[ 0; 0 ] node "padding")
  in
  let dh, dw =
    hw2 esc `Dilation (ints_arg esc ~default:[ 1; 1 ] node "dilation")
  in
  let groups = int_arg esc ~default:1 node "groups" in
  (* Serialized integer padding is SYMMETRIC — ATen pads both sides of an axis
     equally — so both fields take the one value. The asymmetric form exists for
     [Conv2d_padding]'s "same", which splits an odd total unevenly. *)
  let axis ~kernel ~stride ~pad ~dilation : Conv.Conv2d.axis_window =
    {
      kernel = extent esc ~op ~param:`Kernel_size kernel;
      stride = pos esc ~op ~param:`Stride stride;
      pad_before = nonneg esc ~op ~param:`Padding pad;
      pad_after = nonneg esc ~op ~param:`Padding pad;
      dilation = pos esc ~op ~param:`Dilation dilation;
    }
  in
  {
    Conv.Conv2d.h = axis ~kernel:kh ~stride:sh ~pad:ph ~dilation:dh;
    w = axis ~kernel:kw ~stride:sw ~pad:pw ~dilation:dw;
    in_channels = conv_in_channels esc ~tensor:weight_name ~cin ~groups;
    groups = pos esc ~op ~param:`Groups groups;
  }

(* The overload whose padding is a MODE rather than a number. Its [params] carry
   neither a kernel extent nor a channel count: [Conv2d_padding.to_conv2d_params]
   derives both from the weight's shape, and that one definition is already
   shared by shape inference, [Compute] and Native4D. Resolving the mode here
   would make a fourth. *)
let conv2d_padding_params esc (graph : Pytorch_types.Graph.t)
    (node : Pytorch_types.Node.t) =
  let op = node.Node.target in
  let weight_name = tensor_name esc node "weight" in
  (* Read for its RANK alone -- the extents are shape inference's business here.
     Checked all the same, so the two conv2d arms accept the same weights: a
     rank-3 weight would otherwise be right-aligned into the six-axis frame and
     relayouted as though its leading axis were the output channel. *)
  let sizes =
    static_sizes esc ~tensor:weight_name
      (tensor_meta esc graph ~ssa:weight_name ~role:`Conv2d_padding_weight)
  in
  let _ = sizes_rank_4 esc ~tensor:weight_name sizes in
  let padding =
    (* NOT [Conv.Conv2d_padding.padding_of_string], which [invalid_arg]s on
       anything else (conv.ml:394). This string is model data, so it gets a
       typed row -- the same rule the guarded config constructors follow. *)
    match
      Conv.Conv2d_padding.of_string
        (string_arg esc ~default:"valid" node "padding")
    with
    | Ok p -> p
    | Error s -> malformed esc (`Unsupported_padding_mode s)
  in
  let stride = hw2 esc `Stride (ints_arg esc ~default:[ 1; 1 ] node "stride") in
  let dilation =
    hw2 esc `Dilation (ints_arg esc ~default:[ 1; 1 ] node "dilation")
  in
  {
    Conv.Conv2d_padding.stride = pos_hw esc ~op ~param:`Stride stride;
    padding;
    dilation = pos_hw esc ~op ~param:`Dilation dilation;
    groups = pos esc ~op ~param:`Groups (int_arg esc ~default:1 node "groups");
  }

let conv_params esc (graph : Pytorch_types.Graph.t)
    (node : Pytorch_types.Node.t) =
  let weight_name = tensor_name esc node "weight" in
  let sizes =
    static_sizes esc ~tensor:weight_name
      (tensor_meta esc graph ~ssa:weight_name ~role:`Convolution_weight)
  in
  let cout, cin, kh, kw = sizes_rank_4 esc ~tensor:weight_name sizes in
  let _ = cout in
  let op = node.Node.target in
  let stride = hw2 esc `Stride (ints_arg esc ~default:[ 1; 1 ] node "stride") in
  let padding =
    hw2 esc `Padding (ints_arg esc ~default:[ 0; 0 ] node "padding")
  in
  let dilation =
    hw2 esc `Dilation (ints_arg esc ~default:[ 1; 1 ] node "dilation")
  in
  let groups = int_arg esc ~default:1 node "groups" in
  (* [output_padding] IS an argument of this overload -- the schema is
     convolution(..., bool transposed, SymInt[] output_padding, int groups) --
     and forcing it to zero was wrong in both directions. A transposed
     convolution's output extent includes it, so a nonzero value built a smaller
     op than the model asked for; and a NON-transposed one with a nonzero value
     is invalid, which [Convolution.output_shape] rejects (conv.ml:833) and
     discarding the argument let through. [Op_bridge] has always decoded it, so
     the two importers built different ops from one node. *)
  let output_padding =
    hw2 esc `Output_padding
      (ints_arg esc ~default:[ 0; 0 ] node "output_padding")
  in
  ( {
      Conv.Convolution.stride = pos_hw esc ~op ~param:`Stride stride;
      padding = nonneg_hw esc ~op ~param:`Padding padding;
      dilation = pos_hw esc ~op ~param:`Dilation dilation;
      transposed = bool_arg esc node "transposed";
      output_padding = nonneg_hw esc ~op ~param:`Output_padding output_padding;
      groups = pos esc ~op ~param:`Groups groups;
    },
    cin,
    kh,
    kw )

(* Shared by both pooling arms, which is why the [dilation] rejection below is
   here rather than in either one: [max_pool2d_with_indices.default] dropped
   it exactly as silently as the functional overload would have, and one copy
   of the check cannot come to disagree with the other.

   [dilation] is REJECTED, not carried: [Pool.MaxPool2d.params] has no field
   for it, so a non-default value would compute a different op under the right
   name. Extending the native IR is warranted only on a measured need, and no
   model this repository can download serialises either pooling target with a
   non-default dilation -- so there is nothing to measure and a rejection is
   the honest answer. [ceil_mode] IS carried (see
   [Pool.MaxPool2d.params.ceil_mode]) -- the 100-model sweep found real models
   needing it. Revisit [dilation], and [count_include_pad], when
   avg_pool2d.default forces the same question. *)
let pool_params esc (node : Pytorch_types.Node.t) =
  let op = node.Node.target in
  let kh, kw = hw2 esc `Kernel_size (ints_arg esc node "kernel_size") in
  (* Validated BEFORE the stride is defaulted from it. The default makes the two
     the same value, so a kernel of 0 reached the stride's check first and was
     reported as a bad stride -- a diagnostic naming an argument the model never
     supplied. *)
  let kernel =
    {
      Op_config.Hw.h = extent esc ~op ~param:`Kernel_size kh;
      w = extent esc ~op ~param:`Kernel_size kw;
    }
  in
  (* An EMPTY stride list means "same as the kernel" and is a different
     spelling from the argument being absent, so both normalize here. Same rule
     as [Op_bridge.pool_stride] (op_bridge.ml:371). *)
  let stride =
    match ints_arg esc ~default:[ kh; kw ] node "stride" with
    | [] -> [ kh; kw ]
    | s -> s
  in
  let stride = hw2 esc `Stride stride in
  let padding =
    hw2 esc `Padding (ints_arg esc ~default:[ 0; 0 ] node "padding")
  in
  (* NORMALIZED FIRST, then compared against the only value the params can hold.
     Testing "does some element differ from 1" accepted [] and [1;1;1] as well
     as [1;1] -- and ATen refuses both ("dilation must be either a single int,
     or a tuple of two ints"), so the arity check every other H/W argument gets
     was the one thing standing between a refused node and a silent drop. *)
  let dilation = ints_arg esc ~default:[ 1; 1 ] node "dilation" in
  (match hw2 esc `Dilation dilation with
  | 1, 1 -> ()
  | _ -> malformed esc (`Unsupported_option { op; option = `Dilation dilation }));
  let ceil_mode = bool_arg esc ~default:false node "ceil_mode" in
  {
    Pool.MaxPool2d.ceil_mode;
    kernel;
    stride = pos_hw esc ~op ~param:`Stride stride;
    pad = nonneg_hw esc ~op ~param:`Padding padding;
  }

(* [used] is the innermost [rank] frame axes, so it has SIX entries once rank
   exceeds six — and then [d >= rank] admits d = 6 and [List.nth] raises
   [Failure "nth"]. The rank comes from a node's [tensor_values] metadata, which
   is untrusted model data and is NOT covered by [shape_of_sizes]'s own
   rank check: that one runs over graph inputs and captured tensors, not over an
   edge some node produced. Guarding here covers every caller
   (mean.dim, permute.default, unbind.int) rather than each arm separately, and
   reports the same row [shape_of_sizes] would for the same condition. *)
let used_axes_for esc ~tensor rank =
  if rank > 6 then
    malformed esc (`Bad_dimension { tensor; fault = `Rank_over_six })
  else List.filteri (fun i _ -> i >= 6 - rank) Axis.all

let axes_for_rank esc ~tensor rank dims =
  let used = used_axes_for esc ~tensor rank in
  List.map
    (fun d ->
      let d = if d < 0 then d + rank else d in
      if d < 0 || d >= rank then
        malformed esc (`Axis_out_of_range { axis = d; rank })
      else List.nth used d)
    dims

let native_perm esc ~tensor ~rank dims =
  let used = used_axes_for esc ~tensor rank in
  let outer = List.filter (fun a -> not (List.mem a used)) Axis.all in
  List.map (fun a -> (a, a)) outer
  @ List.mapi
      (fun i d ->
        let d = if d < 0 then d + rank else d in
        if d < 0 || d >= rank then
          malformed esc (`Axis_out_of_range { axis = d; rank });
        (List.nth used i, List.nth used d))
      dims

(* Shares [Aten_shape.resolve_view_size] with [Op_bridge] rather than
   re-deriving the [-1] convention: op3-impl.md F1 found this resolver
   accepted an invalid target silently (two [-1]s, a numel mismatch, a
   non-divisible inference) and F8 found its diagnostic named a tensor called
   "view" that never existed. Composed through [Err.Escape.or_throw], which
   exists precisely so a recursive walk can call an ordinary result-returning
   function without threading results through its own arms
   ([conv_in_channels] above is the same pattern: a bounded [int64] count
   inside the escape walk, reported as a typed row). *)
let resolve_view esc ~tensor shape size =
  let bad_view fault : error = `Bad_view { Bad_view.size; fault } in
  let numel =
    Err.Escape.or_throw esc
      (Err.map_error bad_view
         (Vec6.numel_bounded ~limit:Kernel.Limits.Hard.numel shape))
  in
  let resolved =
    Err.Escape.or_throw esc
      (Err.map_error
         (fun e -> bad_view (`Aten_shape e))
         (Aten_shape.resolve_view_size ~numel size))
  in
  shape_of_sizes esc tensor (List.map (fun x -> SymInt.Int x) resolved)

(* The shared resolver, wrapped in this module's own row. Beside [resolve_view]
   and for its reason: the arm that calls it runs inside the builder monad,
   where the ambient error type is [Graph_builder.error], so the widening has to
   happen out here where [error] is what a row can be. *)
let resolve_slice_arg esc ~extent ~start ~stop ~step =
  Err.Escape.or_throw esc
    (Err.map_error
       (fun e : error ->
         `Bad_slice { Bad_slice.start; stop; step; fault = `Aten_shape e })
       (Aten_shape.resolve_slice ~extent ~start ~stop ~step))

(* [aten.select.int]'s index, shared with [Op_bridge] the same way
   [resolve_slice_arg] shares [Aten_shape.resolve_slice]: ATen REJECTS an
   out-of-range index rather than clamping it, so this cannot reuse
   [resolve_slice_arg]'s bound. *)
let resolve_select_index esc ~extent ~index =
  Err.Escape.or_throw esc
    (Err.map_error
       (fun e : error ->
         `Bad_select { Bad_select.index; fault = `Aten_shape e })
       (Aten_shape.resolve_index ~extent ~index))

let lower program =
  Err.Escape.with_escape @@ fun esc ->
  Err.Escape.or_throw esc
  @@
  let graph : Pytorch_types.Graph.t =
    program.ExportedProgram.graph_module.graph
  in
  let sign = program.ExportedProgram.graph_module.GraphModule.signature in
  let source_specs =
    List.map
      (function
        | InputSpec.User_input { UserInputSpec.arg = Argument.Tensor a } ->
            (`Input, a.TensorArgument.name, None)
        | InputSpec.Parameter p ->
            (`Constant, p.arg.TensorArgument.name, Some p.parameter_name)
        | InputSpec.Buffer p ->
            (`Constant, p.arg.TensorArgument.name, Some p.buffer_name)
        | InputSpec.Tensor_constant p ->
            (`Constant, p.arg.TensorArgument.name, Some p.tensor_constant_name)
        | _ -> Err.Escape.throw esc (`Unsupported_input `Non_tensor))
      sign.GraphSignature.input_specs
  in
  (* Every SSA name any node READS, plus the graph's own outputs -- the
     complement of "dead". Computed once and lazily, so a graph containing no
     [native_layer_norm] never pays for it, and one containing 49 of them pays
     once rather than 49 times: the alternative, scanning the remaining nodes
     from inside the arm, is quadratic in the node count on exactly the models
     this row exists for.

     A node's own outputs are not reads, so they are not collected; only
     [inputs] and [graph.outputs] are. *)
  let reads =
    lazy
      (let add acc (a : Argument.t) =
         match a with
         | Argument.Tensor t -> String_map.add t.TensorArgument.name () acc
         | Argument.Optional_tensor (OptionalTensorArgument.Tensor t) ->
             String_map.add t.TensorArgument.name () acc
         | Argument.Tensors ts ->
             List.fold_left
               (fun acc (t : TensorArgument.t) ->
                 String_map.add t.TensorArgument.name () acc)
               acc ts
         | _ -> acc
       in
       List.fold_left
         (fun acc (n : Node.t) ->
           List.fold_left
             (fun acc (i : NamedArgument.t) -> add acc i.NamedArgument.arg)
             acc n.Node.inputs)
         (List.fold_left add String_map.empty graph.outputs)
         graph.nodes)
  in
  let tensor_origins = ref Tensor_id.Map.empty in
  let captured_targets = ref Tensor_id.Map.empty in
  let node_outputs = ref [] in
  let body =
    let open Graph_builder in
    let* env =
      List.fold_left
        (fun acc (kind, name, target) ->
          let* env = acc in
          let shape = tensor_shape esc graph name in
          let* id =
            match kind with
            | `Input -> input ~shape ()
            | `Constant -> constant ~shape ()
          in
          tensor_origins :=
            Tensor_id.Map.add id
              (Pt2_native_graph.Source
                 {
                   graph_path = [];
                   ssa_name = name;
                   meta = String_map.find_opt name graph.tensor_values;
                 })
              !tensor_origins;
          Option.iter
            (fun x ->
              captured_targets := Tensor_id.Map.add id x !captured_targets)
            target;
          return (String_map.add name id env))
        (return String_map.empty) source_specs
    in
    let lower_op env node =
      let get name = env_find esc env (tensor_name esc node name) in
      (* A Tensor-typed argument the exporter serialised as a bare scalar.
           MobileNet-v3's hardsigmoid is `add(x, 3)` / `div(x, 6)` with `as_int`
           arguments; the ATen path materialises those through [full_like]
           ([Interp_decode.tensor_or_scalar_arg]), and the native path routes
           them to the scalar-parameter ops instead, so no edge needs binding. *)
      let tensor_or_scalar name =
        match find_arg esc node name with
        | Argument.Tensor t -> `Tensor (env_find esc env t.TensorArgument.name)
        | Argument.Int i -> `Scalar (float_of_int i)
        | Argument.Float f -> `Scalar f
        | _ ->
            malformed esc
              (`Wrong_arg_kind
                 { op = node.target; arg = name; expected = `Tensor_or_scalar })
      in
      match node.target with
      (* The exact functional overload, kept separate from [convolution.default]
         rather than folded into it: they build different IR nodes, and the
         [Conv2d] one is what Native4D reads directly. Only the metadata and
         H/W decoding is shared, through [conv2d_params]. *)
      | "torch.ops.aten.conv2d.default" ->
          let params = conv2d_params esc graph node in
          let* x = permute perm_nchw_to_nhwc (get "input") in
          let* w = permute perm_oihw_to_conv_weight (get "weight") in
          let bias_name =
            optional_tensor_name ~absent_ok:true esc node "bias"
          in
          Option.iter
            (fun ssa ->
              require_rank esc graph ~ssa ~role:`Conv2d_bias ~expected:1)
            bias_name;
          let bias = Option.map (env_find esc env) bias_name in
          let* y = conv2d params ~x ~weight:w ?bias () in
          let* y = permute perm_nhwc_to_nchw y in
          return [ y ]
      (* Same decode and same relayouts as the arm above; only the padding
         contract differs, and the mode is carried into the IR unresolved. *)
      | "torch.ops.aten.conv2d.padding" ->
          let params = conv2d_padding_params esc graph node in
          let* x = permute perm_nchw_to_nhwc (get "input") in
          let* w = permute perm_oihw_to_conv_weight (get "weight") in
          let bias_name =
            optional_tensor_name ~absent_ok:true esc node "bias"
          in
          Option.iter
            (fun ssa ->
              require_rank esc graph ~ssa ~role:`Conv2d_padding_bias ~expected:1)
            bias_name;
          let bias = Option.map (env_find esc env) bias_name in
          let* y = conv2d_padding params ~x ~weight:w ?bias () in
          let* y = permute perm_nhwc_to_nchw y in
          return [ y ]
      | "torch.ops.aten.convolution.default" ->
          let params, _, _, _ = conv_params esc graph node in
          let* x = permute perm_nchw_to_nhwc (get "input") in
          let* w = permute perm_oihw_to_conv_weight (get "weight") in
          let bias_name = optional_tensor_name esc node "bias" in
          Option.iter
            (fun ssa ->
              require_rank esc graph ~ssa ~role:`Convolution_bias ~expected:1)
            bias_name;
          let bias = Option.map (env_find esc env) bias_name in
          let* y = convolution params ~x ~weight:w ?bias () in
          let* y = permute perm_nhwc_to_nchw y in
          return [ y ]
      | "torch.ops.aten._native_batch_norm_legit_no_training.default" ->
          let* x = permute perm_nchw_to_nhwc (get "input") in
          let params =
            { Norm.BatchNorm.channel = Axis.C; eps = float_arg esc node "eps" }
          in
          let* y =
            batch_norm params ~x
              ?weight:
                (Option.map (env_find esc env)
                   (optional_tensor_name esc node "weight"))
              ?bias:
                (Option.map (env_find esc env)
                   (optional_tensor_name esc node "bias"))
              ~running_mean:(get "running_mean")
              ~running_var:(get "running_var") ()
          in
          let* y = permute perm_nhwc_to_nchw y in
          return [ y ]
      (* [input] is right-aligned NCHW like [batch_norm]'s, so the same
         permute pair relays it around the op -- the serialized counterpart of
         [Op_bridge]'s [group_norm.default] arm: same checks, same node, same
         treatment of the optional operands (NO ones/zeros tensors when
         absent, for the reason the [layer_norm] arm below states). *)
      | "torch.ops.aten.group_norm.default" ->
          let* x = permute perm_nchw_to_nhwc (get "input") in
          let num_groups = int_arg esc node "num_groups" in
          let eps = float_arg esc ~default:1e-05 node "eps" in
          (* Decoded, not ignored, then discarded -- matching the bridge and
             matching ATen: a non-boolean here is a malformed node. *)
          let (_ : bool) = bool_arg esc ~default:true node "cudnn_enabled" in
          let groups =
            pos esc ~op:"group_norm.default" ~param:`Groups num_groups
          in
          let params = { Norm.GroupNorm.channel = Axis.C; groups; eps } in
          let affine name role =
            let ssa = optional_tensor_name ~absent_ok:true esc node name in
            Option.iter
              (fun ssa -> require_rank esc graph ~ssa ~role ~expected:1)
              ssa;
            Option.map (env_find esc env) ssa
          in
          let* y =
            group_norm params ~x
              ?weight:(affine "weight" `Group_norm_weight)
              ?bias:(affine "bias" `Group_norm_bias)
              ()
          in
          let* y = permute perm_nhwc_to_nchw y in
          return [ y ]
      (* [normalized_shape] is validated against the input, which is the check
         the bridge is missing: it reads only the LENGTH (op_bridge.ml:899) and
         never compares the extents, so a shape that names the wrong axes
         normalizes over the wrong ones and produces a plausible wrong answer.
         [trailing_axes] there does not check [k <= rank] either, and silently
         returns the whole axis list when it is exceeded. *)
      (* The FUNCTIONAL layer norm and its DECOMPOSED twin, in one body, and
         the serialized counterparts of [Op_bridge]'s arms: same checks, same
         node, same treatment of the optional operands. The only differences
         from the bridge are where the extents come from (static metadata here,
         a live tensor there) and that this one can meet an unresolved symbolic
         extent, which [static_sizes] already refuses.

         The two TARGETS differ in three things and in nothing else:
         [native_layer_norm]'s [eps] is required with no schema default, it has
         no [cudnn_enable], and it returns a 3-tuple whose trailing two elements
         this graph does not have. The arithmetic, the axis derivation, the
         normalized_shape validation and the affine handling are identical, so
         they are written once -- CLAUDE.md's rule against a second copy of a
         shape rule free to drift, applied to dispatch, as [view]/[_unsafe_view]
         already does. Both targets are named explicitly, there is no
         fallthrough, and every diagnostic below carries [op] or the resolved
         tensor name, so a failure still says which overload produced it. *)
      | ( "torch.ops.aten.layer_norm.default"
        | "torch.ops.aten.native_layer_norm.default" ) as target ->
          let op, eps =
            match target with
            | "torch.ops.aten.layer_norm.default" ->
                (* Decoded and then discarded, matching the bridge and matching
                   ATen: [layer_norm_symint] takes it as [bool /* cudnn_enable,
                   deprecated */] and drops it. Accepting only [false] would
                   reject the schema's own default. Decoding it is still what
                   rejects a non-boolean.

                   A required float WITH a schema default, so [float_arg
                   ~default] -- not rms_norm's [float_opt_arg], whose absent
                   case means "ATen picks the machine epsilon". *)
                let (_ : bool) =
                  bool_arg esc ~default:true node "cudnn_enable"
                in
                (Norm.Target.Layer_norm, float_arg esc ~default:1e-05 node "eps")
            | _ ->
                (* The decomposed form's [eps] has NO default in the schema, so
                   its absence is a malformed node rather than 1e-5. Every
                   corpus occurrence spells it 1e-06 explicitly. *)
                (Norm.Target.Native_layer_norm, float_arg esc node "eps")
          in
          (* The two dropped outputs, refused if anything reads one. Checked
             before the node is built, so the diagnostic names the op that
             cannot provide the statistic rather than the consumer that wanted
             it. The arity is checked here too: [materialized_output_names]
             keeps only the head, so a 2-tuple would pass the generic bind check
             and silently drop one fewer output than it should. *)
          let* () =
            match op with
            | Norm.Target.Layer_norm | Norm.Target.Rms_norm -> return ()
            | Norm.Target.Native_layer_norm ->
                let names = output_names esc node in
                (match names with
                | [ _; mean; rstd ] ->
                    List.iter
                      (fun (stat, ssa) ->
                        if String_map.mem ssa (Lazy.force reads) then
                          malformed esc
                            (`Live_layer_norm_stats
                               { Live_layer_norm_stats.op = target; stat; ssa }))
                      [ (`Mean, mean); (`Rstd, rstd) ]
                | _ ->
                    malformed esc
                      (`Output_arity
                         {
                           op = target;
                           serialized = List.length names;
                           derived = 3;
                         }));
                return ()
          in
          let x_name = tensor_name esc node "input" in
          let sizes =
            static_sizes esc ~tensor:x_name
              (tensor_meta esc graph ~ssa:x_name ~role:`Layer_norm_input)
          in
          let rank = List.length sizes in
          let normalized = ints_arg esc node "normalized_shape" in
          let k = List.length normalized in
          if k < 1 || k > rank then
            malformed esc (`Normalized_rank { op; rank; got = k });
          let trailing l = List.filteri (fun i _ -> i >= rank - k) l in
          let expected = trailing sizes in
          if expected <> normalized then
            malformed esc (`Normalized_shape { op; expected; got = normalized });
          let params =
            {
              Norm.LayerNorm.dims =
                trailing (used_axes_for esc ~tensor:x_name rank);
              eps;
            }
          in
          (* NO ones/zeros tensors when an affine operand is absent -- see the
             rms_norm arm below. Both importers must make the same choice or one
             of [Graph_ir]'s option arms becomes unreachable from one of them. *)
          let affine name role =
            let ssa = optional_tensor_name ~absent_ok:true esc node name in
            (* Rank [k], not rank 1: ATen indexes both affine operands by the
               whole normalized shape. *)
            Option.iter
              (fun ssa -> require_rank esc graph ~ssa ~role ~expected:k)
              ssa;
            Option.map (env_find esc env) ssa
          in
          let* y =
            layer_norm params ~x:(get "input")
              ?weight:(affine "weight" `Layer_norm_weight)
              ?bias:(affine "bias" `Layer_norm_bias)
              ()
          in
          return [ y ]
      | "torch.ops.aten.rms_norm.default" ->
          (* "input", not "self": the schema is
             rms_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight,
             float? eps), and [Op_bridge] reads the same name. *)
          let x_name = tensor_name esc node "input" in
          let sizes =
            static_sizes esc ~tensor:x_name
              (tensor_meta esc graph ~ssa:x_name ~role:`Rms_norm_input)
          in
          let rank = List.length sizes in
          let normalized = ints_arg esc node "normalized_shape" in
          let k = List.length normalized in
          if k < 1 || k > rank then
            malformed esc
              (`Normalized_rank { op = Norm.Target.Rms_norm; rank; got = k });
          let trailing l = List.filteri (fun i _ -> i >= rank - k) l in
          let expected = trailing sizes in
          if expected <> normalized then
            malformed esc
              (`Normalized_shape
                 { op = Norm.Target.Rms_norm; expected; got = normalized });
          let params =
            {
              Norm.RmsNorm.dims =
                trailing (used_axes_for esc ~tensor:x_name rank);
              eps =
                float_opt_arg esc ~default:Norm.RmsNorm.default_eps node "eps";
            }
          in
          (* NO ones tensor when the weight is absent. [Graph_ir]'s [Rms_norm]
             carries [weight : Tensor_ref.t option] and Native4D reads the
             option (lower.ml:293-299); synthesizing a constant would make the
             two importers build structurally different graphs for the same
             node and leave that arm unreachable from one of them. *)
          let weight_name =
            optional_tensor_name ~absent_ok:true esc node "weight"
          in
          (* Rank [k], not rank 1: ATen indexes the weight by the whole
             normalized shape, so a multi-axis normalization takes a
             multi-axis weight. *)
          Option.iter
            (fun ssa ->
              require_rank esc graph ~ssa ~role:`Rms_norm_weight ~expected:k)
            weight_name;
          let* y =
            rms_norm params ~x:(get "input")
              ?weight:(Option.map (env_find esc env) weight_name)
              ()
          in
          return [ y ]
      (* op8-impl.md commit 3: the serialized twin of [Op_bridge]'s arm --
         same checks, same typed rejections ([Attention.Sdpa.Reject], shared
         so the two importers cannot drift), same treatment of the optional
         mask. Q/K/V/mask rank is checked on the DECLARED metadata rank
         (F13), the only place it survives: [Tensor_sig.t] keeps none.
         [value.C <> query.C] and an inadmissible mask broadcast shape are
         flash-oracle boundaries (F4) [Attention.Sdpa.output_shape] already
         rejects via the eventual [Graph_builder.build] call, not re-checked
         here. *)
      | "torch.ops.aten.scaled_dot_product_attention.default" ->
          let query_name = tensor_name esc node "query" in
          let key_name = tensor_name esc node "key" in
          let value_name = tensor_name esc node "value" in
          require_rank esc graph ~ssa:query_name ~role:`Sdpa_query ~expected:4;
          require_rank esc graph ~ssa:key_name ~role:`Sdpa_key ~expected:4;
          require_rank esc graph ~ssa:value_name ~role:`Sdpa_value ~expected:4;
          let dropout_p = float_arg esc ~default:0.0 node "dropout_p" in
          if not (Float.equal dropout_p 0.0) then
            malformed esc
              (`Sdpa_reject (Attention.Sdpa.Reject.Dropout dropout_p));
          let is_causal = bool_arg esc ~default:false node "is_causal" in
          if is_causal then
            malformed esc (`Sdpa_reject Attention.Sdpa.Reject.Causal);
          let enable_gqa = bool_arg esc ~default:false node "enable_gqa" in
          if enable_gqa then
            malformed esc (`Sdpa_reject Attention.Sdpa.Reject.Gqa);
          let scale =
            match float_opt_arg_opt esc node "scale" with
            | None -> Attention.Sdpa.Scale.Default
            | Some s ->
                if not (Float.is_finite s) then
                  malformed esc
                    (`Sdpa_reject (Attention.Sdpa.Reject.Non_finite_scale s));
                if Float.compare s 0.0 < 0 then
                  malformed esc
                    (`Sdpa_reject (Attention.Sdpa.Reject.Negative_scale s));
                Attention.Sdpa.Scale.Explicit s
          in
          (* NO ones/zeros tensor for an absent mask -- [Graph_ir]'s [Sdpa]
             carries [mask : Tensor_ref.t option] and [Eval_op] fills it;
             materialising here would build a structurally different graph
             from [Op_bridge]'s for the same node. *)
          let mask_name =
            optional_tensor_name ~absent_ok:true esc node "attn_mask"
          in
          Option.iter
            (fun ssa ->
              let meta = tensor_meta esc graph ~ssa ~role:`Sdpa_mask in
              (match meta.TensorMeta.dtype with
              | ScalarType.BOOL ->
                  malformed esc
                    (`Sdpa_reject Attention.Sdpa.Reject.Boolean_mask)
              | _ -> ());
              let got = meta_rank meta in
              if got <> 2 && got <> 4 then
                malformed esc
                  (`Sdpa_reject
                     (Attention.Sdpa.Reject.Rank
                        {
                          arg_name = "sdpa attn_mask";
                          expected = [ 2; 4 ];
                          got;
                        })))
            mask_name;
          let params = { Attention.Sdpa.scale } in
          let* y =
            sdpa params ~query:(get "query") ~key:(get "key")
              ~value:(get "value")
              ?mask:(Option.map (env_find esc env) mask_name)
              ()
          in
          return [ y ]
      | "torch.ops.aten.pow.Tensor_Scalar" ->
          let exponent = required_scalar_arg esc node "exponent" in
          let* y = pow exponent (get "self") in
          return [ y ]
      | "torch.ops.aten.relu.default" ->
          let* y = relu (get "self") in
          return [ y ]
      | "torch.ops.aten.add.Tensor" -> (
          reject_alpha esc node;
          match tensor_or_scalar "other" with
          | `Tensor other ->
              let* y = add (get "self") other in
              return [ y ]
          | `Scalar s ->
              let* y = add_scalar s (get "self") in
              return [ y ])
      (* [x - s] legalizes to [x + (-s)]: IEEE negation is exact and the
         builder narrows to f32 on both spellings either way, so the two are
         bit-identical (op3-impl.md F7). No [sub_scalar] builder exists and
         none should: this negation is the whole legalization. *)
      | "torch.ops.aten.sub.Tensor" -> (
          reject_alpha esc node;
          match tensor_or_scalar "other" with
          | `Tensor other ->
              let* y = sub (get "self") other in
              return [ y ]
          | `Scalar s ->
              let* y = add_scalar (-.s) (get "self") in
              return [ y ])
      | "torch.ops.aten.clamp.default" ->
          let params : Pointwise.Clamp.params =
            {
              min = scalar_opt_arg esc node "min";
              max = scalar_opt_arg esc node "max";
            }
          in
          let* y = clamp params (get "self") in
          return [ y ]
      | "torch.ops.aten.clone.default" ->
          reject_memory_format esc node;
          let* y = clone (get "self") in
          return [ y ]
      | "torch.ops.aten.div.Tensor" -> (
          match tensor_or_scalar "other" with
          | `Tensor other ->
              let* y = div (get "self") other in
              return [ y ]
          | `Scalar s ->
              let* y = div_scalar s (get "self") in
              return [ y ])
      | "torch.ops.aten.hardsigmoid.default"
      | "torch.ops.aten.hardsigmoid_.default" ->
          let* y = hardsigmoid (get "self") in
          return [ y ]
      | "torch.ops.aten.hardswish.default" | "torch.ops.aten.hardswish_.default"
        ->
          let* y = hardswish (get "self") in
          return [ y ]
      | "torch.ops.aten.sigmoid.default" ->
          let* y = sigmoid (get "self") in
          return [ y ]
      | "torch.ops.aten.silu.default" | "torch.ops.aten.silu_.default" ->
          let* y = silu (get "self") in
          return [ y ]
      | "torch.ops.aten.gelu.default" ->
          let approximate = string_arg esc ~default:"none" node "approximate" in
          let approximate =
            match approximate with
            | "none" -> Pointwise.Gelu.Exact
            | "tanh" -> Pointwise.Gelu.Tanh
            | _ ->
                malformed esc
                  (`Unsupported_option
                     { op = node.target; option = `Approximate approximate })
          in
          let* y = gelu approximate (get "self") in
          return [ y ]
      | "torch.ops.aten.hardtanh.default" ->
          (* Schema defaults are -1/1; MobileNet-v2 always serialises 0/6. *)
          let params : Pointwise.Hardtanh.params =
            {
              min_val = scalar_arg esc ~default:(-1.) node "min_val";
              max_val = scalar_arg esc ~default:1. node "max_val";
            }
          in
          let* y = hardtanh params (get "self") in
          return [ y ]
      | "torch.ops.aten.mul.Tensor" -> (
          match tensor_or_scalar "other" with
          | `Tensor other ->
              let* y = mul (get "self") other in
              return [ y ]
          | `Scalar s ->
              let* y = mul_scalar s (get "self") in
              return [ y ])
      | "torch.ops.aten.mul.Scalar" ->
          let s = required_scalar_arg esc node "other" in
          let* y = mul_scalar s (get "self") in
          return [ y ]
      (* The functional overload has ONE output and takes the generic path:
         [materialized_output_names] must not gain it, since that list is for
         nodes whose trailing outputs are dropped. *)
      | "torch.ops.aten.max_pool2d.default" ->
          let* x = permute perm_nchw_to_nhwc (get "self") in
          let* y = max_pool2d (pool_params esc node) x in
          let* y = permute perm_nhwc_to_nchw y in
          return [ y ]
      | "torch.ops.aten.adaptive_avg_pool2d.default" ->
          let x_name = tensor_name esc node "self" in
          let got =
            meta_rank
              (tensor_meta esc graph ~ssa:x_name
                 ~role:`Adaptive_avg_pool2d_input)
          in
          if got <> 3 && got <> 4 then
            malformed esc (`Adaptive_pool_rank { tensor = x_name; got });
          let out_h, out_w =
            match ints_arg esc node "output_size" with
            | [ h; w ] -> (h, w)
            | xs ->
                malformed esc
                  (`Bad_arity
                     { Bad_arity.param = `Output_size; got = List.length xs })
          in
          let params =
            {
              Pool.AdaptiveAvgPool2d.output_size =
                {
                  h = pos esc ~op:node.target ~param:`Output_size out_h;
                  w = pos esc ~op:node.target ~param:`Output_size out_w;
                };
            }
          in
          let* x = permute perm_nchw_to_nhwc (get "self") in
          let* y = adaptive_avg_pool2d params x in
          let* y = permute perm_nhwc_to_nchw y in
          return [ y ]
      | "torch.ops.aten.max_pool2d_with_indices.default" ->
          let* x = permute perm_nchw_to_nhwc (get "self") in
          let* values, indices =
            max_pool2d_with_indices (pool_params esc node) x
          in
          let* () = discard indices in
          let* values = permute perm_nhwc_to_nchw values in
          return [ values ]
      | "torch.ops.aten.mean.dim" ->
          let x_name = tensor_name esc node "self" in
          let rank =
            meta_rank (tensor_meta esc graph ~ssa:x_name ~role:`Mean_input)
          in
          let params =
            {
              Reduce.Mean.dims =
                axes_for_rank esc ~tensor:x_name rank (ints_arg esc node "dim");
              keepdim = bool_arg esc node "keepdim";
            }
          in
          let* y = mean params (get "self") in
          return [ y ]
      | "torch.ops.aten.amax.default" ->
          let x_name = tensor_name esc node "self" in
          let rank =
            meta_rank (tensor_meta esc graph ~ssa:x_name ~role:`Amax_input)
          in
          let params =
            {
              Reduce.Amax.dims =
                axes_for_rank esc ~tensor:x_name rank (ints_arg esc node "dim");
              keepdim = bool_arg esc node "keepdim";
            }
          in
          let* y = amax params (get "self") in
          return [ y ]
      | "torch.ops.aten.linalg_vector_norm.default" ->
          let ord = scalar_arg esc ~default:2. node "ord" in
          if not (Float.equal ord 2.) then
            malformed esc
              (`Unsupported_option
                 { op = node.target; option = `Vector_norm_ord ord });
          reject_dtype esc node;
          let x_name = tensor_name esc node "self" in
          let rank =
            meta_rank
              (tensor_meta esc graph ~ssa:x_name ~role:`Vector_norm_input)
          in
          let params =
            {
              Reduce.Vector_norm.dims =
                axes_for_rank esc ~tensor:x_name rank (ints_arg esc node "dim");
              keepdim = bool_arg esc node "keepdim";
            }
          in
          let* y = vector_norm params (get "self") in
          return [ y ]
      | "torch.ops.aten.permute.default" ->
          let x_name = tensor_name esc node "self" in
          let rank =
            meta_rank (tensor_meta esc graph ~ssa:x_name ~role:`Permute_input)
          in
          let* y =
            permute
              (native_perm esc ~tensor:x_name ~rank (ints_arg esc node "dims"))
              (get "self")
          in
          return [ y ]
      (* Reuses [native_perm], the same permute machinery [permute.default]
         builds on, rather than a new builder -- outer padding axes stay
         identity because [native_perm] already does that. Equal dims are a
         real identity transpose, not special-cased away: [List.init] produces
         the identity list, and it lowers like any other permutation. *)
      | "torch.ops.aten.transpose.int" ->
          let x_name = tensor_name esc node "self" in
          let rank =
            meta_rank (tensor_meta esc graph ~ssa:x_name ~role:`Transpose_input)
          in
          let norm_dim d =
            let d = if d < 0 then d + rank else d in
            if d < 0 || d >= rank then
              malformed esc (`Axis_out_of_range { axis = d; rank });
            d
          in
          let d0 = norm_dim (int_arg esc node "dim0") in
          let d1 = norm_dim (int_arg esc node "dim1") in
          let dims =
            List.init rank (fun i ->
                if i = d0 then d1 else if i = d1 then d0 else i)
          in
          let* y =
            permute (native_perm esc ~tensor:x_name ~rank dims) (get "self")
          in
          return [ y ]
      (* The only arm whose output count is not fixed by the op. By the time it
         runs, [output_names] has already bounded and flattened the serialized
         names, so what is left is to normalize the axis, derive the count from
         already-validated metadata, and CHECK the two against each other before
         allocating anything.
         The check has to precede the builder, not merely precede binding: both
         the name list and the extent are model data, and [add_env]'s
         [Invalid_argument] is an invariant about this module rather than a
         report about the graph. *)
      (* The pad list is indexed FROM the innermost dimension, so the rank is
         load-bearing: with the wrong one the same list pads different axes and
         still produces a plausible graph. It comes from the serialized
         metadata here and from the live tensor on the bridge, but the
         un-reversal itself is [Pad.Pad.params_of_aten] on both sides. *)
      | "torch.ops.aten.pad.default" ->
          let x_name = tensor_name esc node "self" in
          let rank =
            meta_rank (tensor_meta esc graph ~ssa:x_name ~role:`Pad_input)
          in
          let params =
            Err.Escape.or_throw esc
              (Pad.Pad.params_of_aten ~rank ~pad:(ints_arg esc node "pad")
                 ~mode:(string_arg esc ~default:"constant" node "mode")
                 ~value:(float_opt_arg_opt esc node "value"))
          in
          let* y = pad params (get "self") in
          return [ y ]
      (* The variadic Native [Concat] op, direct -- see [Op_bridge]'s
         "torch.ops.aten.cat.default" arm for why the rank check runs BEFORE
         [Concat]'s own axis-agreement shape rule rather than being left to
         it. *)
      | "torch.ops.aten.cat.default" -> (
          match tensor_names_arg esc node "tensors" with
          | [] -> malformed esc (`Concat_no_tensors node.target)
          | name0 :: _ as names ->
              let rank =
                meta_rank (tensor_meta esc graph ~ssa:name0 ~role:`Concat_input)
              in
              List.iter
                (fun name ->
                  let got =
                    meta_rank
                      (tensor_meta esc graph ~ssa:name ~role:`Concat_input)
                  in
                  if got <> rank then
                    malformed esc
                      (`Concat_rank_mismatch
                         {
                           Concat_rank_mismatch.op = node.target;
                           first = rank;
                           other = got;
                         }))
                names;
              let d =
                let dim = int_arg esc ~default:0 node "dim" in
                let dim = if dim < 0 then dim + rank else dim in
                if dim < 0 || dim >= rank then
                  malformed esc (`Axis_out_of_range { axis = dim; rank });
                dim
              in
              let axis = List.nth (used_axes_for esc ~tensor:name0 rank) d in
              let* y =
                concat { Concat.Concat.axis }
                  (List.map (env_find esc env) names)
              in
              return [ y ])
      (* One [Stack] node: inserts a size-1 axis per operand at [axis], then
         joins them -- see [Op_bridge]'s arm for why (ATen's own definition),
         and [unsqueeze.default] below for the rank+1 [dim] convention this
         reuses via the [d > rank] bound (not [>=]). *)
      | "torch.ops.aten.stack.default" -> (
          match tensor_names_arg esc node "tensors" with
          | [] -> malformed esc (`Concat_no_tensors node.target)
          | name0 :: _ as names ->
              let rank =
                meta_rank (tensor_meta esc graph ~ssa:name0 ~role:`Stack_input)
              in
              List.iter
                (fun name ->
                  let got =
                    meta_rank
                      (tensor_meta esc graph ~ssa:name ~role:`Stack_input)
                  in
                  if got <> rank then
                    malformed esc
                      (`Concat_rank_mismatch
                         {
                           Concat_rank_mismatch.op = node.target;
                           first = rank;
                           other = got;
                         }))
                names;
              let d =
                let dim = int_arg esc ~default:0 node "dim" in
                let dim = if dim < 0 then dim + rank + 1 else dim in
                if dim < 0 || dim > rank then
                  malformed esc (`Axis_out_of_range { axis = dim; rank });
                dim
              in
              let axis =
                List.nth (used_axes_for esc ~tensor:name0 (rank + 1)) d
              in
              let* y =
                stack { Concat.Stack.axis } (List.map (env_find esc env) names)
              in
              return [ y ])
      (* One [Select] node: picks index [idx] along the normalized axis and
         drops it. The normalized ATen-level position [d] is needed once, as
         the frame axis via [used_axes_for], the same shape [transpose.int]'s
         [norm_dim] takes below. *)
      | "torch.ops.aten.select.int" ->
          let x_name = tensor_name esc node "self" in
          let rank =
            meta_rank (tensor_meta esc graph ~ssa:x_name ~role:`Select_input)
          in
          let d =
            let dim = int_arg esc node "dim" in
            let dim = if dim < 0 then dim + rank else dim in
            if dim < 0 || dim >= rank then
              malformed esc (`Axis_out_of_range { axis = dim; rank });
            dim
          in
          let axis = List.nth (used_axes_for esc ~tensor:x_name rank) d in
          let shape = tensor_shape esc graph x_name in
          let extent = Vec6.get shape axis in
          let idx =
            resolve_select_index esc ~extent ~index:(int_arg esc node "index")
          in
          let* y = select { Split.Select.axis; index = idx } (get "self") in
          return [ y ]
      (* Legalized to [Reshape] alone: inserting a size-1 axis never changes
         the linearized data order, so no [Slice] is needed, unlike
         [select.int]'s axis removal. [dim] is judged against rank+1 valid
         positions (the OUTPUT rank), so this does not reuse the local
         [norm_dim] pattern above verbatim -- the upper bound is relaxed by
         one, the same adjustment [Op_bridge.norm_unsqueeze_dim] makes. *)
      | "torch.ops.aten.unsqueeze.default" ->
          let x_name = tensor_name esc node "self" in
          let rank =
            meta_rank (tensor_meta esc graph ~ssa:x_name ~role:`Unsqueeze_input)
          in
          let d =
            let dim = int_arg esc node "dim" in
            let dim = if dim < 0 then dim + rank + 1 else dim in
            if dim < 0 || dim > rank then
              malformed esc (`Axis_out_of_range { axis = dim; rank });
            dim
          in
          let shape = tensor_shape esc graph x_name in
          let aten_list = Array.to_list (Aten_shape.to_aten ~rank shape) in
          let front = List.filteri (fun i _ -> i < d) aten_list in
          let back = List.filteri (fun i _ -> i >= d) aten_list in
          let out_sizes =
            List.map (fun x -> SymInt.Int x) (front @ [ 1 ] @ back)
          in
          let* y =
            reshape
              { Reshape.Reshape.shape = shape_of_sizes esc x_name out_sizes }
              (get "self")
          in
          return [ y ]
      (* Everything after [Aten_shape.resolve_slice] is shared with the bridge
         arm; what differs is where the extent comes from, and that is the point
         of a shared resolver rather than two normalizations. Here it is the
         SERIALIZED shape, which is also where an unresolved symbolic dimension
         is already refused; on the bridge it is the live tensor. *)
      | "torch.ops.aten.slice.Tensor" ->
          let x_name = tensor_name esc node "self" in
          let rank =
            meta_rank (tensor_meta esc graph ~ssa:x_name ~role:`Slice_input)
          in
          let axis =
            match
              axes_for_rank esc ~tensor:x_name rank
                [ int_arg esc ~default:0 node "dim" ]
            with
            | [ a ] -> a
            | _ -> invalid_arg "Native_interp: axes_for_rank lost its singleton"
          in
          let extent = Vec6.get (tensor_shape esc graph x_name) axis in
          let start = int_opt_arg_opt esc node "start" in
          let stop = int_opt_arg_opt esc node "end" in
          let step = int_arg esc ~default:1 node "step" in
          let bounds = resolve_slice_arg esc ~extent ~start ~stop ~step in
          let* y =
            slice
              {
                Split.Slice.axis;
                start = bounds.Aten_shape.Slice_bounds.start;
                stop = bounds.Aten_shape.Slice_bounds.stop;
                step = bounds.Aten_shape.Slice_bounds.step;
              }
              (get "self")
          in
          return [ y ]
      | "torch.ops.aten.unbind.int" ->
          let x_name = tensor_name esc node "self" in
          let rank =
            meta_rank (tensor_meta esc graph ~ssa:x_name ~role:`Unbind_input)
          in
          let axis =
            match
              axes_for_rank esc ~tensor:x_name rank
                [ int_arg esc ~default:0 node "dim" ]
            with
            | [ a ] -> a
            | _ -> invalid_arg "Native_interp: axes_for_rank lost its singleton"
          in
          (* No arity check here: [bind] does it against the ids actually
             produced, which is both stronger and total over ops. *)
          unbind { Split.Unbind.axis } (get "self")
      (* Divides [axis] into contiguous windows of [split_sizes], KEEPING the
         axis in every output, unlike [unbind.int]. Same shape as that arm --
         rank from the input's own metadata, [dim] resolved through
         [axes_for_rank] -- with [split_sizes] read as an int list the way
         [view.default]'s [size] is; [Split.Split_with_sizes.output_shapes] is
         what checks the sizes are positive and sum to the axis's extent, not
         this arm. No arity check here either, for the reason [unbind.int]'s
         comment gives. *)
      | "torch.ops.aten.split_with_sizes.default" ->
          let x_name = tensor_name esc node "self" in
          let rank =
            meta_rank
              (tensor_meta esc graph ~ssa:x_name ~role:`Split_with_sizes_input)
          in
          let axis =
            match
              axes_for_rank esc ~tensor:x_name rank
                [ int_arg esc ~default:0 node "dim" ]
            with
            | [ a ] -> a
            | _ -> invalid_arg "Native_interp: axes_for_rank lost its singleton"
          in
          let sizes = ints_arg esc node "split_sizes" in
          split_with_sizes { Split.Split_with_sizes.axis; sizes } (get "self")
      (* Both overloads share this body -- same argument names, same shared
         resolver, same reason op_bridge.ml's arm does (drift risk, exact
         target still visible in every diagnostic via [tensor]/[node.target]).
         [Identical] AFTER materialization; see .ai/native_aten_bridge_layout.md. *)
      | "torch.ops.aten.view.default" | "torch.ops.aten._unsafe_view.default" ->
          let tensor = tensor_name esc node "self" in
          let shape = tensor_shape esc graph tensor in
          let* y =
            reshape
              {
                Reshape.Reshape.shape =
                  resolve_view esc ~tensor shape (ints_arg esc node "size");
              }
              (get "self")
          in
          return [ y ]
      (* Not mergeable with [addmm.default] below, though both build a [Linear]:
         there [self] IS the bias and is required, here the bias is optional,
         and the two weights are transposes of each other. *)
      | "torch.ops.aten.linear.default" ->
          let w_name = tensor_name esc node "weight" in
          let _out_features, in_features =
            sizes_rank_2 esc ~tensor:w_name
              (static_sizes esc ~tensor:w_name
                 (tensor_meta esc graph ~ssa:w_name ~role:`Linear_weight))
          in
          let* w = permute perm_linear_weight (get "weight") in
          let bias_name =
            optional_tensor_name ~absent_ok:true esc node "bias"
          in
          Option.iter
            (fun ssa ->
              require_rank esc graph ~ssa ~role:`Linear_bias ~expected:1)
            bias_name;
          let bias = Option.map (env_find esc env) bias_name in
          (* No check that the input's trailing extent matches [in_features]:
             [Linear.output_shape] already compares both the activation's and
             the weight's C against it (linear.ml:69-84), and a second copy of
             that rule here is one that could drift from it. *)
          let* y =
            linear
              {
                Linear.Linear.in_features =
                  dim_extent esc ~tensor:w_name in_features;
              }
              ~x:(get "input") ~weight:w ?bias ()
          in
          return [ y ]
      | "torch.ops.aten.addmm.default" ->
          let w_name = tensor_name esc node "mat2" in
          let in_features =
            match
              static_sizes esc ~tensor:w_name
                (tensor_meta esc graph ~ssa:w_name ~role:`Addmm_weight)
            with
            | n :: _ -> n
            | [] ->
                (* A rank-zero mat2, which this arm cannot read a feature count
                   from. It used to report [`Symbolic], which was simply not
                   true; [`Expected_rank] arrived with linear.default's own rank
                   check and is the accurate row. Only the RANK-ZERO case is
                   rejected here -- addmm reads dim 0 and has never required
                   rank two, so demanding it would be a behaviour change rather
                   than a corrected diagnostic. *)
                malformed esc
                  (`Bad_dimension
                     {
                       tensor = w_name;
                       fault = `Expected_rank { expected = 2; got = 0 };
                     })
          in
          let* w = permute perm_addmm_weight (get "mat2") in
          let* y =
            linear
              {
                Linear.Linear.in_features =
                  dim_extent esc ~tensor:w_name in_features;
              }
              ~x:(get "mat1") ~weight:w ~bias:(get "self") ()
          in
          return [ y ]
      | target -> Err.Escape.throw esc (`Unsupported_operator target)
    in
    let lower_node index env node =
      let names = materialized_output_names esc node in
      let bind ids =
        (* THE arity check, here rather than in any operator arm, and against the
           ids the builder actually produced rather than against anything
           derived. Two reasons, both learned the hard way:

           [output_names] flattens [Argument.Tensors] for EVERY node, not just
           the ops that return one — so a serialized `relu` carrying an
           `as_tensors` output reaches this point with two names and one id.
           Before flattening, that shape was refused as
           [`Non_tensor_node_output]; an arm-local check would restore the hole
           for every op it does not cover.

           And a check against metadata-derived counts is not the same property:
           a node whose declared [tensor_values] shape disagrees with the shape
           Native infers passes it and still arrives here mismatched.

           [add_env]'s [Invalid_argument] stays an invariant about this module,
           and this is what keeps a malformed export from reaching it — as does
           [List.iter2] just below, which would otherwise raise first. *)
        if List.compare_lengths names ids <> 0 then
          malformed esc
            (`Output_arity
               {
                 op = node.Pytorch_types.Node.target;
                 serialized = List.length names;
                 derived = List.length ids;
               });
        List.iter2
          (fun name id ->
            tensor_origins :=
              Tensor_id.Map.add id
                (Pt2_native_graph.Source
                   {
                     graph_path = [];
                     ssa_name = name;
                     meta = String_map.find_opt name graph.tensor_values;
                   })
                !tensor_origins)
          names ids;
        node_outputs := (index, ids) :: !node_outputs;
        add_env env names ids
      in
      if is_nontrivial_node node then
        let* ids = group ~label:node.target (lower_op env node) in
        return (bind ids)
      else
        let* ids = lower_op env node in
        return (bind ids)
    in
    let* env =
      List.fold_left
        (fun acc (index, node) ->
          let* env = acc in
          lower_node index env node)
        (return env)
        (List.mapi (fun i n -> (i, n)) graph.nodes)
    in
    (* The second place a `Tensor[]` has to flatten, through the same bounded
       helper: a graph may return the whole list an unbind produced. The budget
       is shared across the graph's outputs, so the bound is on the AGGREGATE
       interface width rather than on each list separately. *)
    let outputs =
      List.map (env_find esc env)
        (flatten_outputs esc
           ~on_bad_kind:(fun () -> malformed esc `Non_tensor_graph_output)
           graph.outputs)
    in
    return outputs
  in
  match
    Graph_builder.build ~name:"pt2" ~outputs:Fun.id body
    |> Err.map_error ~pos:__POS__ (fun e -> `Build e)
  with
  | Error _ as e -> e
  | Ok native_graph ->
      let node_origins =
        List.fold_left
          (fun acc (source_index, ids) ->
            let origin_node = List.nth graph.nodes source_index in
            List.fold_left
              (fun acc (node : Graph_ir.node) ->
                if
                  List.exists
                    (fun id -> List.mem id node.Graph_ir.Node.outputs)
                    ids
                then
                  Node_id.Map.add node.Graph_ir.Node.id
                    [
                      {
                        Pt2_native_graph.Node_origin.graph_path = [];
                        index = source_index;
                        target = origin_node.target;
                        name = origin_node.name;
                        metadata = origin_node.metadata;
                      };
                    ]
                    acc
                else acc)
              acc native_graph.Graph_ir.Graph.nodes)
          Node_id.Map.empty !node_outputs
      in
      Pt2_native_graph.make ~graph:native_graph ~tensor_origins:!tensor_origins
        ~node_origins ~captured_targets:!captured_targets
      |> Err.map_error ~pos:__POS__ (fun e -> `Provenance e)

let lower_archive archive = lower (Pt2_archive.program archive)

let tensor_of_pt2 (tensor : Pt2_tensor.t) =
  let open Err.Syntax in
  match tensor.dtype with
  | Pt2_dtype.Float32 -> (
      (* [shape_of_sizes] is written for graph metadata, so it throws into the
         lowering row; this is the only caller outside [lower], so it owns the
         frame and re-labels what comes out. Only the [malformed] rows can
         arrive, and re-labelling them keeps which row it was. *)
      let* shape =
        Err.Escape.with_escape (fun esc ->
            shape_of_sizes esc "tensor"
              (List.map (fun x -> SymInt.Int x) tensor.sizes))
        |> Err.map_error ~pos:__POS__ (function
          | #malformed as e -> `Tensor_bridge (e :> tensor_bridge)
          | e -> e)
      in
      if List.compare_lengths tensor.sizes tensor.strides <> 0 then
        Err.fail
          (`Tensor_bridge
             (`Rank_mismatch
                {
                  Rank_mismatch.sizes = List.length tensor.sizes;
                  strides = List.length tensor.strides;
                }))
      else
        let storage_index coord =
          List.fold_left ( + ) tensor.storage_offset
            (List.mapi
               (fun i stride ->
                 let axis =
                   List.nth Axis.all (6 - List.length tensor.sizes + i)
                 in
                 Dim.to_int (Vec6.get coord axis) * stride)
               tensor.strides)
        in
        let data_len = Bytes.length tensor.data in
        (* Bound the WHOLE reachable index range once, in int64, before
           materializing. [storage_index] itself runs per element in plain int,
           where a product or sum of in-range factors can still overflow -- and
           js_of_ocaml's int is 32 bits, so that wrap would be silent. Checking
           the extremes up front proves every coordinate in between is in range,
           so the inner loop stays int and stays fast. Strides may be negative
           (PyTorch stores channels-last inputs non-contiguously), hence both
           ends rather than just the maximum. See [[js_backends_design]]. *)
        (* The arithmetic that computes the bound must itself be checked, or the
           bound is decorative: [sizes=[2]] with [strides=[1 lsl 61]] makes
           [hi * 4 + 4] wrap negative, sail past a naive comparison, and then the
           per-element offset wraps to 0 and silently re-reads element 0. So each
           multiply and add is overflow-tested, and the byte bound is checked by
           DIVIDING the capacity rather than scaling [hi] up. *)
        let exception Range_overflow in
        let mul64 a b =
          if Int64.equal b 0L then 0L
          else
            let r = Int64.mul a b in
            if Int64.equal (Int64.div r b) a then r else raise Range_overflow
        in
        let add64 a b =
          let r = Int64.add a b in
          (* Overflow iff the operands agree in sign and the result does not. *)
          if
            Int64.compare (Int64.logxor a b) 0L >= 0
            && Int64.compare (Int64.logxor a r) 0L < 0
          then raise Range_overflow
          else r
        in
        let range =
          try
            let o = Int64.of_int tensor.storage_offset in
            Ok
              (List.fold_left2
                 (fun (lo, hi) size stride ->
                   let span =
                     mul64
                       (Int64.of_int (max 0 (size - 1)))
                       (Int64.of_int stride)
                   in
                   if Int64.compare span 0L < 0 then (add64 lo span, hi)
                   else (lo, add64 hi span))
                 (o, o) tensor.sizes tensor.strides)
          with Range_overflow -> Error ()
        in
        let out_of_range (lo, hi) =
          (* Every shape has extent >= 1, so at least one element is always read
             and fewer than 4 bytes is always an error. [(data_len - 4) / 4] is
             the largest admissible index, computed without scaling [hi]. *)
          data_len < 4
          || Int64.compare lo 0L < 0
          || Int64.compare hi (Int64.of_int ((data_len - 4) / 4)) > 0
        in
        match range with
        | Error () -> Err.fail (`Tensor_bridge `Storage_index_overflow)
        | Ok (lo, hi) when out_of_range (lo, hi) ->
            Err.fail
              (`Tensor_bridge
                 (`Storage_out_of_range
                    { Storage_range.lo; hi; data_bytes = data_len }))
        | Ok _ -> (
            try
              Err.return
                (Tensor.materialize shape (fun coord ->
                     let offset = storage_index coord * 4 in
                     Int32.float_of_bits (Bytes.get_int32_le tensor.data offset)))
            with Invalid_argument m ->
              Err.fail (`Tensor_bridge (`Materialize_failed m))))
  | dtype -> Err.fail (`Tensor_bridge (`Unsupported_dtype dtype))

let run ?hooks archive ~input =
  let open Err.Syntax in
  let* lowered = lower_archive archive in
  let graph = lowered.Pt2_native_graph.graph in
  let eval_hooks =
    Option.map
      (fun (Hooks h) ->
        Eval_direct.Hooks
          {
            on_start = (fun node -> h.on_start lowered node);
            on_end = (fun node state -> h.on_end lowered node state);
          })
      hooks
  in
  let* input = tensor_of_pt2 input in
  let user_ids =
    List.filter
      (fun id -> Graph_ir.input_kind graph id = Graph_ir.Input.Input)
      graph.Graph_ir.Graph.inputs
  in
  let* inputs =
    match user_ids with
    | [ id ] -> Err.return [ (id, input) ]
    | ids ->
        Err.fail
          (`Unsupported_input (`Not_exactly_one_user_input (List.length ids)))
  in
  let used_constants =
    List.concat_map
      (fun node -> Graph_ir.operands node.Graph_ir.Node.op)
      graph.Graph_ir.Graph.nodes
  in
  let* constants =
    Err.List.map
      (fun (id, target) ->
        let* raw =
          Pt2_archive.load_captured_tensor archive target
          |> Err.map_error ~pos:__POS__ (fun e -> `Tensor_bridge (`Archive e))
        in
        let+ tensor = tensor_of_pt2 raw in
        (id, tensor))
      (List.filter
         (fun (id, _) -> List.mem id used_constants)
         (Tensor_id.Map.bindings lowered.captured_targets))
  in
  let* env =
    Eval_direct.run ?hooks:eval_hooks ~constants graph ~inputs
    |> Err.map_error ~pos:__POS__ (fun e -> `Eval e)
  in
  Err.List.map
    (fun id ->
      Tensor_id.Map.find_opt id env |> Err.of_option (`Output_not_evaluated id))
    graph.Graph_ir.Graph.outputs

(* ---- transforming, and running the result --------------------------------- *)

type transformed =
  | Transformed : {
      constants : Tensor.packed Tensor_id.Map.t;
      constant_store : Constant_store.t;
      derived : (Tensor_id.t * string list) list;
      graph : Graph_ir.graph;
      lens : 'b Pt2_native_graph.lens;
      nodes_before : int;
      audits : Pass.Audit_log.t;
      composed : Map_verify.Report.t option;
    }
      -> transformed

type loaded = { from_state : int; from_archive : int; from_plan : int }

let load_captured archive target =
  let open Err.Syntax in
  let* raw =
    Pt2_archive.load_captured_tensor archive target
    |> Err.map_error ~pos:__POS__ (fun e -> `Tensor_bridge (`Archive e))
  in
  tensor_of_pt2 raw

let capture_resolver archive capture =
  load_captured archive (Const_ssa.Capture.to_string capture)
  |> Err.map_error (fun _ -> `Missing_capture capture)

(* The PT2 names a destination constant derives from: its provenance sources,
   resolved in the sidecar the importer built. Asked only of an edge with no
   archive path of its own — a folded weight — which is exactly where "where did
   this come from" has no other answer. *)
let derivations lens sidecar (graph : Graph_ir.graph) =
  let open Err.Syntax in
  Err.List.fold_left
    (fun acc id ->
      if Graph_ir.input_kind graph id <> Graph_ir.Input.Constant then
        Err.return acc
      else
        let+ target =
          Pt2_native_graph.captured_target lens id
          |> Err.map_error ~pos:__POS__ (fun e -> `Lens e)
        in
        match target with
        | Some _ -> acc
        | None -> (
            let names =
              List.filter_map
                (fun src ->
                  match
                    Tensor_id.Map.find_opt src
                      sidecar.Pt2_native_graph.tensor_origins
                  with
                  | Some (Pt2_native_graph.Source o) ->
                      Some o.Pt2_native_graph.Tensor_origin.ssa_name
                  | Some Pt2_native_graph.Derived | None -> None)
                (Pt2_native_graph.provenance_sources lens id)
            in
            match names with [] -> acc | _ -> (id, names) :: acc))
    [] graph.Graph_ir.Graph.inputs

(* Everything downstream of lowering, over a graph the caller already has.

   Split out of [transform] because the archive is not a prerequisite for any of
   it: a caller holding a [Pt2_native_graph.t] — from a payload-free
   [model.json], or from a graph it built — can transform, verify and pack
   without an archive to read constants from, which [transform] would have
   demanded. [~constants] is that seed, as a MAP rather than the association
   list [Rewrite.origin] takes: an id appearing twice with different payloads is
   not a state this entry point should have to define an answer for. *)
let transform_lowered ?(constants = Tensor_id.Map.empty) ?verify ?verify_budget
    ?verify_probe ?max_verified_steps ?max_verify_clusters ?trace
    ?max_trace_entries ?max_audit_reports lowered ~passes =
  let open Err.Syntax in
  let source = lowered.Pt2_native_graph.graph in
  let seeded = Tensor_id.Map.bindings constants in
  let transform_error e = `Transform ((e : Rewrite.error) :> Pass.error) in
  let* constant_store =
    Tensor_id.Map.fold
      (fun id target acc ->
        let* store = acc in
        match Tensor_id.Map.find_opt id source.Graph_ir.Graph.tensors with
        | None -> Err.return store
        | Some tensor ->
            Constant_store.bind_captured store ~tensor
              (Const_ssa.Capture.of_string target)
            |> Err.map_error (fun e -> (e :> Rewrite.error))
            |> Err.map_error ~pos:__POS__ transform_error)
      lowered.Pt2_native_graph.captured_targets
      (Err.return Constant_store.empty)
  in
  let* (Rewrite.Origin origin) =
    Rewrite.origin ~constant_store ~constants:seeded source
    |> Err.map_error ~pos:__POS__ transform_error
  in
  let* {
         Pass.audits;
         trace = _;
         next_index = _;
         step = Rewrite.Step (rewritten, rewrite_map);
       } =
    Pass.run_reporting ?verify ?verify_budget ?verify_probe ?max_verified_steps
      ?trace ?max_trace_entries ?max_audit_reports origin passes
    |> Err.map_error ~pos:__POS__ (fun e -> `Transform e)
  in
  let* (Rewrite.Step (packed, pack_map)) =
    Rewrite.pack rewritten |> Err.map_error ~pos:__POS__ transform_error
  in
  let graph = Rewrite.graph packed in
  let composed_map = Graph_map.compose rewrite_map pack_map in
  let* lens =
    Pt2_native_graph.lens lowered ~src:origin composed_map ~dst:packed
    |> Err.map_error ~pos:__POS__ (fun e -> `Lens e)
  in
  (* The per-pass audits say what each rewrite established; this says what
     survived all of them, in the FINAL graph's ids — which is what lets a
     printed node carry its own verdict. A cluster here also names every origin
     edge that collapsed into one destination edge, so a node several passes
     rewrote reads as the one claim they add up to rather than as a pile of
     intermediate ones. *)
  let* composed =
    match verify with
    | None -> Err.return None
    | Some policy ->
        let* report =
          Map_verify.run ?budget:verify_budget ?probe:verify_probe
            ?max_clusters:max_verify_clusters composed_map
            ~src:(Rewrite.snapshot origin)
            ~src_constants:(Rewrite.constants origin)
            ~src_constant_store:(Rewrite.constant_store origin)
            ~dst:(Rewrite.snapshot packed)
            ~dst_constants:(Rewrite.constants packed)
            ~dst_constant_store:(Rewrite.constant_store packed)
          |> Err.map_error ~pos:__POS__ (fun e -> `Verify e)
        in
        (* The policy applies here too. Composition and terminal packing are the
           two steps no per-pass check covers — a refutation introduced by
           [Graph_map.compose] or [Rewrite.pack] appears in this report and
           nowhere else — so computing it and not judging it would print the
           failure inline and still exit successfully. *)
        if Map_verify.Policy.accepts policy report then Err.return (Some report)
        else
          Err.fail
            (`Transform
               (`Verification
                  {
                    Pass.Verification.pass = "compose+pack";
                    problem = Pass.Verification.Rejected report;
                  }))
  in
  let+ derived = derivations lens lowered graph in
  Transformed
    {
      composed;
      constants = Rewrite.constants packed;
      constant_store = Rewrite.constant_store packed;
      derived = List.rev derived;
      graph;
      lens;
      nodes_before = List.length source.Graph_ir.Graph.nodes;
      audits;
    }

(* Lower, optionally bind the payloads a node reads, then hand the rest to
   [transform_lowered]. The signature is unchanged, so every existing caller is
   unaffected by the split. *)
let preload archive lowered =
  let open Err.Syntax in
  let source = lowered.Pt2_native_graph.graph in
  let read_by_a_node =
    List.concat_map
      (fun (n : Graph_ir.node) -> Graph_ir.operands n.Graph_ir.Node.op)
      source.Graph_ir.Graph.nodes
    |> Tensor_id.Set.of_list
  in
  (* Only what a node reads: an archive holds buffers nothing evaluates —
     resnet18's int64 [num_batches_tracked] among them — and loading one would
     fail on a dtype the engine has no reason to support. *)
  let+ seeded =
    Err.List.map
      (fun (id, target) ->
        let+ payload = load_captured archive target in
        (id, payload))
      (List.filter
         (fun (id, _) -> Tensor_id.Set.mem id read_by_a_node)
         (Tensor_id.Map.bindings lowered.Pt2_native_graph.captured_targets))
  in
  Tensor_id.Map.of_seq (List.to_seq seeded)

let transform ?preload:(want_payloads = false) ?verify ?verify_budget
    ?verify_probe archive ~passes =
  let open Err.Syntax in
  let* lowered = lower_archive archive in
  let* constants =
    if want_payloads then preload archive lowered
    else Err.return Tensor_id.Map.empty
  in
  transform_lowered ~constants ?verify ?verify_budget ?verify_probe lowered
    ~passes

(* Payloads for the constants the graph actually reads, state before archive.
   An edge with neither is simply absent; [Eval_direct] is the one that decides
   whether that matters, and says which edge if it does. *)
let constants_for archive ~lens ~graph ~computed =
  let open Err.Syntax in
  let used =
    List.concat_map
      (fun (n : Graph_ir.node) -> Graph_ir.operands n.Graph_ir.Node.op)
      graph.Graph_ir.Graph.nodes
    |> Tensor_id.Set.of_list
  in
  let+ loaded =
    Err.List.fold_left
      (fun acc id ->
        if not (Tensor_id.Set.mem id used) then Err.return acc
        else
          match Tensor_id.Map.find_opt id computed with
          | Some payload -> Err.return ((id, payload, `State) :: acc)
          | None -> (
              let* target =
                Pt2_native_graph.captured_target lens id
                |> Err.map_error ~pos:__POS__ (fun e -> `Lens e)
              in
              match target with
              | None -> Err.return acc
              | Some target ->
                  let+ payload = load_captured archive target in
                  (id, payload, `Archive) :: acc))
      []
      (List.filter
         (fun id -> Graph_ir.input_kind graph id = Graph_ir.Input.Constant)
         graph.Graph_ir.Graph.inputs)
  in
  let count source =
    List.length (List.filter (fun (_, _, s) -> s = source) loaded)
  in
  ( List.rev_map (fun (id, payload, _) -> (id, payload)) loaded,
    { from_state = count `State; from_archive = count `Archive; from_plan = 0 }
  )

let evaluate archive (Transformed t) ~input =
  let open Err.Syntax in
  let* input = tensor_of_pt2 input in
  let* store, materialized =
    Const_ssa_materialize.materialize (capture_resolver archive)
      t.constant_store
    |> Err.map_error ~pos:__POS__ (fun e -> `Materialize e)
  in
  let* constants, loaded =
    constants_for archive ~lens:t.lens ~graph:t.graph
      ~computed:(Constant_store.materialized store)
  in
  let user_ids =
    List.filter
      (fun id -> Graph_ir.input_kind t.graph id = Graph_ir.Input.Input)
      t.graph.Graph_ir.Graph.inputs
  in
  let* inputs =
    match user_ids with
    | [ id ] -> Err.return [ (id, input) ]
    | ids ->
        Err.fail
          (`Unsupported_input (`Not_exactly_one_user_input (List.length ids)))
  in
  let* env =
    Eval_direct.run ~constants t.graph ~inputs
    |> Err.map_error ~pos:__POS__ (fun e -> `Eval e)
  in
  let+ outputs =
    Err.List.map
      (fun id ->
        Tensor_id.Map.find_opt id env
        |> Err.of_option (`Output_not_evaluated id))
      t.graph.Graph_ir.Graph.outputs
  in
  (outputs, { loaded with from_plan = materialized.applies })
