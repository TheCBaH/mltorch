(* The element types the real engine pair exchanges. The coordinator's codec
   registry knows f32, f64, i32, i64, u8 and bool; anything else is refused
   when a model is described, not when a value reaches an engine. *)

open Inference_compare_schema
module Dtype = Pt2_checkpoint_map.Dtype

let codec_name : Dtype.t -> string option = function
  | F32 -> Some "f32"
  | F64 -> Some "f64"
  | I32 -> Some "i32"
  | I64 -> Some "i64"
  | BOOL -> Some "bool"
  | U8 -> Some "u8"
  | _ -> None

let of_scalar_type (t : Pytorch_types.ScalarType.t) : Dtype.t option =
  match t with
  | FLOAT -> Some F32
  | DOUBLE -> Some F64
  | INT -> Some I32
  | LONG -> Some I64
  | BOOL -> Some BOOL
  | BYTE -> Some U8
  | _ -> None

let codec_of_name n =
  match Id.Codec.of_string n with
  | Ok c -> c
  | Error _ -> invalid_arg "Dtypes.codec"

let layout =
  match Id.Layout.of_string "row_major" with
  | Ok l -> l
  | Error _ -> invalid_arg "Dtypes.layout"

let spec ~(dtype : Dtype.t) ~(shape : int64 list) : Tensor_spec.t option =
  Option.map
    (fun n ->
      {
        Tensor_spec.dtype =
          { codec = codec_of_name n; parameters_digest = None };
        logical_shape = shape;
        layout_id = layout;
        value_semantics = true;
      })
    (codec_name dtype)

let dtype_of_codec (c : Id.Codec.t) : Dtype.t option =
  match Id.Codec.to_string c with
  | "f32" -> Some F32
  | "f64" -> Some F64
  | "i32" -> Some I32
  | "i64" -> Some I64
  | "bool" -> Some BOOL
  | "u8" -> Some U8
  | _ -> None
