module W = Wire
module Operation = W.Text (Id.Codec_operation)

type overflow = Reject | Saturate | Wrap
type rounding = Nearest_even | Toward_negative | Toward_positive | Toward_zero

type t =
  | Cast of { target : Dtype.t; rounding : rounding; overflow : overflow }
  | Codec_operation of {
      name : Id.Codec_operation.t;
      implementation_digest : Digest.Implementation.t;
    }
  | Concat of int
  | Permute of int list
  | Reshape of int64 list
  | Slice of { axis : int; start : int64; length : int64 }

let decode_rounding : rounding W.decoder =
 fun j ->
  let ( let* ) = W.( let* ) in
  let* s = W.string j in
  match s with
  | "nearest_even" -> Ok Nearest_even
  | "toward_negative" -> Ok Toward_negative
  | "toward_positive" -> Ok Toward_positive
  | "toward_zero" -> Ok Toward_zero
  | tag -> W.unknown_tag ~kind:"rounding" tag

let encode_rounding r =
  W.enc_string
    (match r with
    | Nearest_even -> "nearest_even"
    | Toward_negative -> "toward_negative"
    | Toward_positive -> "toward_positive"
    | Toward_zero -> "toward_zero")

let decode_overflow : overflow W.decoder =
 fun j ->
  let ( let* ) = W.( let* ) in
  let* s = W.string j in
  match s with
  | "reject" -> Ok Reject
  | "saturate" -> Ok Saturate
  | "wrap" -> Ok Wrap
  | tag -> W.unknown_tag ~kind:"overflow" tag

let encode_overflow o =
  W.enc_string
    (match o with
    | Reject -> "reject"
    | Saturate -> "saturate"
    | Wrap -> "wrap")

let decode : t W.decoder =
  W.variant (fun tag o ->
      let ( let* ) = W.( let* ) in
      match tag with
      | "cast" ->
          let* target = W.Obj.required o "target" Dtype.decode in
          let* rounding = W.Obj.required o "rounding" decode_rounding in
          let* overflow = W.Obj.required o "overflow" decode_overflow in
          Ok (Cast { target; rounding; overflow })
      | "codec_operation" ->
          let* name = W.Obj.required o "name" Operation.decode in
          let* implementation_digest =
            W.Obj.required o "implementation_digest" Wire_digest.implementation
          in
          Ok (Codec_operation { name; implementation_digest })
      | "concat" ->
          let* axis = W.Obj.required o "axis" W.nonneg_int in
          Ok (Concat axis)
      | "permute" ->
          let* axes = W.Obj.required o "axes" (W.list W.nonneg_int) in
          Ok (Permute axes)
      | "reshape" ->
          let* shape = W.Obj.required o "shape" (W.list W.nonneg_int64) in
          Ok (Reshape shape)
      | "slice" ->
          let* axis = W.Obj.required o "axis" W.nonneg_int in
          let* start = W.Obj.required o "start" W.nonneg_int64 in
          let* length = W.Obj.required o "length" W.nonneg_int64 in
          Ok (Slice { axis; start; length })
      | tag -> W.unknown_tag ~kind:"conversion_step" tag)

let encode = function
  | Cast { target; rounding; overflow } ->
      W.tagged "cast"
        [
          W.field "target" Dtype.encode target;
          W.field "rounding" encode_rounding rounding;
          W.field "overflow" encode_overflow overflow;
        ]
  | Codec_operation { name; implementation_digest } ->
      W.tagged "codec_operation"
        [
          W.field "name" Operation.encode name;
          W.field "implementation_digest" Wire_digest.encode_implementation
            implementation_digest;
        ]
  | Concat axis -> W.tagged "concat" [ W.field "axis" W.enc_int axis ]
  | Permute axes ->
      W.tagged "permute" [ W.field "axes" (W.enc_list W.enc_int) axes ]
  | Reshape shape ->
      W.tagged "reshape" [ W.field "shape" (W.enc_list W.enc_int64) shape ]
  | Slice { axis; start; length } ->
      W.tagged "slice"
        [
          W.field "axis" W.enc_int axis;
          W.field "start" W.enc_int64 start;
          W.field "length" W.enc_int64 length;
        ]
