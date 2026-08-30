(* Error payload types and printer for [Op_bridge], split from op_bridge.ml. Purely data plus deterministic printing -- no dependency on
   the argument-decoding or per-op dispatch machinery, so this compiles
   first. This file is not published under a [.mli] of its own: the library
   is [(wrapped false)] and [Op_bridge]'s external surface is just
   [dispatch]/[pp_error]/[error] (see op_bridge.ml), so there is no separate
   contract to keep legible here. *)

type aten_env = Interp_decode.env
type invalid_hw_arg = { name : string; values : int list }

(* [cause] is the bridge's OWN row, not a rendering of it: [Tensor_bridge]
   used to hand back a string and this wrapped it, so a caller could read what
   went wrong but never branch on it. *)
type tensor_bridge_error = { arg_name : string; cause : Tensor_bridge.error }

(* Own modules with the payload named [t], per CLAUDE.md's record convention.
   The top-level records above predate it and are not a pattern to extend. *)

(* Shared by every arm that resolves a decoded dim through
   [Aten_shape.axis_of_dim]: [op] is what distinguishes one arm's fault from
   another's (unbind.int, mean.dim, permute.default, transpose.int, ...),
   [dim] is the ORIGINAL value as decoded (not normalized), and [rank] is the
   operand's rank. *)
module Invalid_dim = struct
  type t = { op : string; dim : int; rank : int }
end

(* [permute.default]/[transpose.int] build a full permutation from a decoded
   dims list; a list whose length disagrees with the operand's rank is a
   distinct fault from any single dim being out of range. *)
module Dims_count = struct
  type t = { op : string; rank : int; got : int }
end

(* [cat.default]/[stack.default]: ATen requires every tensor in the list to
   share one rank (there is no broadcasting between list entries), checked
   here rather than left to [Concat]'s own shape rule -- that rule compares
   FRAME axes, and two different ranks land their data on different frame
   axes under [Aten_shape.of_aten]'s right-alignment, so a rank disagreement
   would surface there as a confusing off-axis extent mismatch instead of
   the rank fault it actually is. *)
module Concat_rank_mismatch = struct
  type t = { op : string; first : int; other : int }
end

(* Carries ATen's OWN dtype, read off the source tensor before conversion,
   rather than a Native [Payload.fmt]. Two reasons: [Payload.fmt] is a
   three-parameter GADT and cannot be a bare field type (its existential is
   [Payload.packed_fmt]), and translating a dtype into a Native format purely to
   report it would be a conversion inside an error. *)
module Unsupported_input_dtype = struct
  type t = { arg_name : string; dtype : Aten_scalar_type.t }
end

(* The one pooling option [Pool.MaxPool2d.params] has no field for
   ([ceil_mode] IS represented -- see [Pool.MaxPool2d.params.ceil_mode]).
   Own module because the row carries the op alongside the value, and [op]
   would collide with the other records here. *)
(* The two [normalized_shape] faults, shared by every normalisation. The arm
   read the LENGTH of [normalized_shape] and nothing else, so a shape naming
   extents the input does not have normalized over the wrong axes and returned a
   plausible wrong answer; and [trailing_axes] silently returns the whole axis
   list when [k > rank], so an over-long shape normalized over everything.

   [op] names which normalisation, because three targets now share these rows
   and the message used to say "rms_norm" for all of them. *)
module Normalized_rank = struct
  type t = { op : Norm.Target.t; rank : int; got : int }
end

module Normalized_shape = struct
  type t = { op : Norm.Target.t; expected : int list; got : int list }
end

(* An optional operand's declared RANK, which right-alignment into the six-axis
   frame erases before any shared shape rule can see it. *)
module Operand_rank = struct
  type t = { arg_name : string; expected : int; got : int }
end

(* [matmul.default]'s remaining unsupported shape family, now that both the
   batch-less case (`.ai/matmul_softmax_design.md` §4, binds to the existing
   [Bmm] node) and the batched/multi-head case (§5, binds to the new
   [Batched_matmul] node) are supported: either operand is rank<2 (the
   vector-matrix/matrix-vector/vector-vector forms ATen gives a different
   output-rank rule, §6) or the two operands have different rank. Carries
   both raw ATen shapes -- not just "which check failed" -- since a rank
   mismatch and a too-low rank are two different reasons a reader needs to
   see the actual shapes to tell apart. *)
module Matmul_unsupported_shape = struct
  type t = { self_shape : int array; other_shape : int array }
end

module Pool_unsupported = struct
  type option = Dilation of int list | Divisor_override of int
  type t = { op : string; option : option }
end

module Adaptive_pool_rank = struct
  type t = { got : int }
end

(* sdpa's typed rejection boundary is [Attention.Sdpa.Reject], shared with
   [Native_interp] (op8-impl.md commit 3) for [Op_config.Bad]'s reason: the
   two importers must reject the same values. *)

(* `index.Tensor`'s locked list-acceptance rule (`.ai/index_tensor_design.md`
   round 3): [indices] is accepted iff its length equals [self]'s ATen rank,
   exactly one entry is a Long-dtype tensor of ATen rank exactly 1, and every
   other entry is an explicit [None]. Every fault names what was actually
   found, not just that the rule failed -- round 9's own required proof that
   the rank-1 restriction is a real, provable typed rejection. *)
module Index_list = struct
  type fault =
    | Length_mismatch of { expected : int; got : int }
    | Multiple_live_entries of int list (* positions, in order found *)
    | No_live_entry
    | Wrong_dtype of { position : int; dtype : Aten_scalar_type.t }
    | Wrong_rank of { position : int; rank : int }

  type t = { fault : fault }
end

type error =
  [ `Adaptive_pool_rank of Adaptive_pool_rank.t
  | `Addmm_invalid_weight_rank of int array
  | `Aten_shape of Aten_shape.error
  | `Bad_config of Op_config.Bad.t
  | `Bad_pad_list of Pad.Pad.Bad_pad_list.t
  | `Build of Graph_builder.error
  | `Concat_no_tensors of string
  | `Concat_rank_mismatch of Concat_rank_mismatch.t
  | `Conv2d_invalid_weight_rank of int array
  | `Conv2d_padding_invalid_weight_rank of int array
  | `Convolution_invalid_weight_rank of int array
  | `Decode of Interp_decode.error
  | `Dims_count of Dims_count.t
  | `Index_list of Index_list.t
  | `Invalid_dim of Invalid_dim.t
  | `Invalid_hw_arg of invalid_hw_arg
  | `Linear_invalid_weight_rank of int array
  | `Matmul_unsupported_shape of Matmul_unsupported_shape.t
  | `Normalized_rank of Normalized_rank.t
  | `Normalized_shape of Normalized_shape.t
  | `Operand_rank of Operand_rank.t
  | `Pool_unsupported of Pool_unsupported.t
  | `Sdpa_reject of Attention.Sdpa.Reject.t
  | `Tensor_bridge of tensor_bridge_error
  | `Unsupported_input_dtype of Unsupported_input_dtype.t
  | `Unsupported_memory_format of [ `Channels_last | `Channels_last_3d ]
  | `Unsupported_padding_mode of string
  | `Validation_failure of string ]

(* Deliberately not [Fmt.brackets], which boxes its content and so may
   line-wrap; the original bare "[%s]" (String.concat) never did, regardless
   of width. *)
let pp_int_list ppf xs =
  Fmt.pf ppf "[%a]" (Fmt.list ~sep:(Fmt.any ", ") Fmt.int) xs

let pp_int_array ppf xs = pp_int_list ppf (Array.to_list xs)

let pp_error ppf : [< error ] -> unit = function
  | `Adaptive_pool_rank { Adaptive_pool_rank.got } ->
      Fmt.pf ppf
        "adaptive_avg_pool2d input must be rank-3 (CHW) or rank-4 (NCHW), got \
         rank-%d"
        got
  | `Addmm_invalid_weight_rank shape ->
      Fmt.pf ppf "addmm: mat2 must be rank-2, got shape %a" pp_int_array shape
  | `Aten_shape e -> Aten_shape.pp_error ppf e
  | `Bad_config e -> Op_config.Bad.pp ppf e
  | `Bad_pad_list e -> Pad.Pad.Bad_pad_list.pp ppf e
  | `Build e -> Graph_builder.pp_error ppf e
  | `Concat_no_tensors op -> Fmt.pf ppf "%s: at least one tensor is required" op
  | `Concat_rank_mismatch { Concat_rank_mismatch.op; first; other } ->
      Fmt.pf ppf "%s: every tensor must have the same rank: %d vs %d" op first
        other
  | `Conv2d_invalid_weight_rank shape ->
      Fmt.pf ppf "conv2d: weight must be rank-4, got shape %a" pp_int_array
        shape
  | `Conv2d_padding_invalid_weight_rank shape ->
      Fmt.pf ppf "conv2d.padding: weight must be rank-4, got shape %a"
        pp_int_array shape
  | `Convolution_invalid_weight_rank shape ->
      Fmt.pf ppf "convolution: weight must be rank-4, got shape %a" pp_int_array
        shape
  | `Decode e -> Interp_decode.pp_error ppf e
  | `Dims_count { Dims_count.op; rank; got } ->
      Fmt.pf ppf "%s: expected %d dims, got %d" op rank got
  | `Index_list { Index_list.fault } -> (
      match fault with
      | Index_list.Length_mismatch { expected; got } ->
          Fmt.pf ppf
            "index.Tensor: indices has %d entries, expected %d (self's rank)"
            got expected
      | Index_list.Multiple_live_entries positions ->
          Fmt.pf ppf
            "index.Tensor: indices has more than one live entry, at positions \
             %a"
            pp_int_list positions
      | Index_list.No_live_entry ->
          Fmt.string ppf "index.Tensor: indices has no live (non-None) entry"
      | Index_list.Wrong_dtype { position; dtype } ->
          Fmt.pf ppf "index.Tensor: indices[%d] must be Long, got %s" position
            (Aten_scalar_type.to_string dtype)
      | Index_list.Wrong_rank { position; rank } ->
          Fmt.pf ppf "index.Tensor: indices[%d] must be rank 1, got rank %d"
            position rank)
  | `Invalid_dim { Invalid_dim.op; dim; rank } ->
      Fmt.pf ppf "%s: invalid dimension %d for rank %d" op dim rank
  | `Invalid_hw_arg { name; values } ->
      Fmt.pf ppf "%s: expected [h; w] or [v], got %a" name pp_int_list values
  | `Linear_invalid_weight_rank shape ->
      Fmt.pf ppf "linear: weight must be rank-2, got shape %a" pp_int_array
        shape
  | `Matmul_unsupported_shape
      { Matmul_unsupported_shape.self_shape; other_shape } ->
      Fmt.pf ppf
        "matmul.default: both operands must be rank>=2 and of equal rank, got \
         self=%a other=%a"
        pp_int_array self_shape pp_int_array other_shape
  | `Normalized_rank { Normalized_rank.op; rank; got } ->
      Fmt.pf ppf
        "%a: normalized_shape has %d entries, outside [1, %d] for this rank"
        Norm.Target.pp op got rank
  | `Normalized_shape { Normalized_shape.op; expected; got } ->
      Fmt.pf ppf
        "%a: normalized_shape %a does not match the input's trailing extents %a"
        Norm.Target.pp op pp_int_list got pp_int_list expected
  | `Operand_rank { Operand_rank.arg_name; expected; got } ->
      Fmt.pf ppf "%s must be rank-%d, got rank-%d" arg_name expected got
  | `Pool_unsupported { Pool_unsupported.op; option } -> (
      match option with
      | Pool_unsupported.Dilation d ->
          Fmt.pf ppf "%s: dilation=%a is not supported (only 1)" op pp_int_list
            d
      | Pool_unsupported.Divisor_override d ->
          Fmt.pf ppf "%s: divisor_override=%d is not supported (only none)" op d
      )
  | `Sdpa_reject e -> Attention.Sdpa.Reject.pp ppf e
  | `Tensor_bridge { arg_name; cause } ->
      Fmt.pf ppf "%s: %a" arg_name Tensor_bridge.pp_error cause
  | `Unsupported_input_dtype { Unsupported_input_dtype.arg_name; dtype } ->
      Fmt.pf ppf "%s: the native engine computes in f32, got %s" arg_name
        (Aten_scalar_type.to_string dtype)
  | `Unsupported_memory_format mf ->
      let name =
        match mf with
        | `Channels_last -> "channels_last"
        | `Channels_last_3d -> "channels_last_3d"
      in
      Fmt.pf ppf "clone: memory_format=%s is not supported" name
  | `Unsupported_padding_mode s ->
      Fmt.pf ppf "padding mode %S is neither \"valid\" nor \"same\"" s
  | `Validation_failure msg -> Fmt.string ppf msg
