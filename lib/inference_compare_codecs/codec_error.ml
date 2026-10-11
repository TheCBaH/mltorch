(* One closed error vocabulary for the codec layer. Every payload is data: a
   count, an identifier or an index, never prose. *)

type shape_fault =
  | Axis_out_of_range of { axis : int; rank : int }
  | Duplicate_axis of int
  | Element_count_changed of { source : int64; target : int64 }
  | Extent_mismatch of { axis : int; left : int64; right : int64 }
  | Negative_extent of int64
  | Permutation_arity of { expected : int; actual : int }
  | Rank_mismatch of { left : int; right : int }
  | Slice_out_of_range of {
      axis : int;
      start : int64;
      length : int64;
      extent : int64;
    }

type t =
  | Arity of { expected : int; actual : int }
  | Byte_length_mismatch of { expected : int64; actual : int64 }
  | Cast_overflow of { element : int64 }
  | Cast_value_unavailable of { element : int64 }
  | Codec_mismatch of {
      expected : Inference_compare_schema.Id.Codec.t;
      actual : Inference_compare_schema.Id.Codec.t;
    }
  | Count_overflow
  | Cyclic_recipe of Inference_compare_schema.Id.Conversion_node.t
  | Digest_mismatch
  | Duplicate_node of Inference_compare_schema.Id.Conversion_node.t
  | Hasher_unavailable of Inference_compare_schema.Id.Digest_algorithm.t
  | Lossy_recipe
  | Resource_exceeded of { requested : int64; limit : int64 }
  | Shape of shape_fault
  | Spec_mismatch
  | Undeclared_lossy_cast
  | Unknown_boundary of Inference_compare_schema.Tensor_ref.t
  | Unknown_canonical of Inference_compare_schema.Id.Canonical_value.t
  | Unknown_codec of Inference_compare_schema.Id.Codec.t
  | Unknown_node of Inference_compare_schema.Id.Conversion_node.t
  | Unknown_operation of Inference_compare_schema.Id.Codec_operation.t
  | Unsupported_cast of {
      source : Inference_compare_schema.Id.Codec.t;
      target : Inference_compare_schema.Id.Codec.t;
    }
  | Unsupported_overflow
  | Unsupported_rounding
  | Value_released

let pp_shape_fault ppf = function
  | Axis_out_of_range { axis; rank } ->
      Fmt.pf ppf "axis %d out of range for rank %d" axis rank
  | Duplicate_axis a -> Fmt.pf ppf "duplicate axis %d" a
  | Element_count_changed { source; target } ->
      Fmt.pf ppf "element count changed from %Ld to %Ld" source target
  | Extent_mismatch { axis; left; right } ->
      Fmt.pf ppf "extent mismatch on axis %d: %Ld vs %Ld" axis left right
  | Negative_extent d -> Fmt.pf ppf "negative extent %Ld" d
  | Permutation_arity { expected; actual } ->
      Fmt.pf ppf "permutation has %d axes, rank is %d" actual expected
  | Rank_mismatch { left; right } ->
      Fmt.pf ppf "rank mismatch: %d vs %d" left right
  | Slice_out_of_range { axis; start; length; extent } ->
      Fmt.pf ppf "slice [%Ld, +%Ld) outside extent %Ld on axis %d" start length
        extent axis

let pp ppf = function
  | Arity { expected; actual } ->
      Fmt.pf ppf "arity: expected %d, got %d" expected actual
  | Byte_length_mismatch { expected; actual } ->
      Fmt.pf ppf "byte length: expected %Ld, got %Ld" expected actual
  | Cast_overflow { element } ->
      Fmt.pf ppf "cast overflow at element %Ld" element
  | Cast_value_unavailable { element } ->
      Fmt.pf ppf "cast of a nonfinite value at element %Ld" element
  | Codec_mismatch { expected; actual } ->
      Fmt.pf ppf "codec: expected %a, got %a"
        Inference_compare_schema.Id.Codec.pp expected
        Inference_compare_schema.Id.Codec.pp actual
  | Count_overflow -> Fmt.string ppf "element count overflow"
  | Cyclic_recipe n ->
      Fmt.pf ppf "cyclic recipe at %a"
        Inference_compare_schema.Id.Conversion_node.pp n
  | Digest_mismatch -> Fmt.string ppf "digest mismatch"
  | Duplicate_node n ->
      Fmt.pf ppf "duplicate node %a"
        Inference_compare_schema.Id.Conversion_node.pp n
  | Hasher_unavailable a ->
      Fmt.pf ppf "digest algorithm %a unavailable"
        Inference_compare_schema.Id.Digest_algorithm.pp a
  | Lossy_recipe -> Fmt.string ppf "recipe declared lossless but is lossy"
  | Resource_exceeded { requested; limit } ->
      Fmt.pf ppf "resource: requested %Ld, limit %Ld" requested limit
  | Shape f -> pp_shape_fault ppf f
  | Spec_mismatch -> Fmt.string ppf "tensor spec mismatch"
  | Undeclared_lossy_cast -> Fmt.string ppf "lossy cast in a lossless recipe"
  | Unknown_boundary r ->
      Fmt.pf ppf "unknown boundary tensor %a"
        Inference_compare_schema.Tensor_ref.pp r
  | Unknown_canonical c ->
      Fmt.pf ppf "unknown canonical value %a"
        Inference_compare_schema.Id.Canonical_value.pp c
  | Unknown_codec c ->
      Fmt.pf ppf "unknown codec %a" Inference_compare_schema.Id.Codec.pp c
  | Unknown_node n ->
      Fmt.pf ppf "unknown node %a"
        Inference_compare_schema.Id.Conversion_node.pp n
  | Unknown_operation o ->
      Fmt.pf ppf "unknown operation %a"
        Inference_compare_schema.Id.Codec_operation.pp o
  | Unsupported_cast { source; target } ->
      Fmt.pf ppf "unsupported cast %a -> %a"
        Inference_compare_schema.Id.Codec.pp source
        Inference_compare_schema.Id.Codec.pp target
  | Unsupported_overflow -> Fmt.string ppf "unsupported overflow policy"
  | Unsupported_rounding -> Fmt.string ppf "unsupported rounding"
  | Value_released -> Fmt.string ppf "value released"
