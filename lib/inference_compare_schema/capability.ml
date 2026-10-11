module W = Wire

(* Admission metadata: what a codec, a conversion operation or a metric
   implementation declares about itself. A provider's description never
   supplies executable policy -- it only lets validation refuse early -- and
   an unsupported capability is stated, not left absent. *)

type element =
  | Boolean
  | Float_binary32
  | Float_binary64
  | Signed_integer
  | Unsigned_integer

type support = Supported | Unsupported

module Codec = struct
  type t = {
    codec : Id.Codec.t;
    element : element;
    byte_width : int;
    element_inspection : support;
    exact_values : support;
  }
end

module Operation = struct
  type t = {
    operation : Id.Codec_operation.t;
    implementation_digest : Digest.Implementation.t;
    arity : int;
    lossless : support;
  }
end

type threshold_direction = At_least | At_most | Declared

module Metric = struct
  type t = {
    comparator : Id.Comparator.t;
    implementation_digest : Digest.Implementation.t;
    threshold_direction : threshold_direction;
    finite_operands_only : bool;
  }
end

module Conversion = struct
  type t = { source : Id.Codec.t; target : Id.Codec.t; lossless : support }
end

let decode_element, encode_element =
  W.enum ~kind:"element"
    [
      (Boolean, "boolean");
      (Float_binary32, "float_binary32");
      (Float_binary64, "float_binary64");
      (Signed_integer, "signed_integer");
      (Unsigned_integer, "unsigned_integer");
    ]

let decode_support, encode_support =
  W.enum ~kind:"support"
    [ (Supported, "supported"); (Unsupported, "unsupported") ]

let decode_direction, encode_direction =
  W.enum ~kind:"threshold_direction"
    [ (At_least, "at_least"); (At_most, "at_most"); (Declared, "declared") ]

module Codec_id = W.Text (Id.Codec)
module Operation_id = W.Text (Id.Codec_operation)
module Comparator_id = W.Text (Id.Comparator)

let positive_width : int W.decoder =
 fun j ->
  let ( let* ) = W.( let* ) in
  let* v = W.int j in
  if v < 1 then W.fail (W.Constraint Out_of_range) else Ok v

let decode_codec : Codec.t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* codec = W.Obj.required o "codec" Codec_id.decode in
      let* element = W.Obj.required o "element" decode_element in
      let* byte_width = W.Obj.required o "byte_width" positive_width in
      let* element_inspection =
        W.Obj.required o "element_inspection" decode_support
      in
      let* exact_values = W.Obj.required o "exact_values" decode_support in
      Ok { Codec.codec; element; byte_width; element_inspection; exact_values })

let encode_codec (c : Codec.t) =
  W.obj
    [
      W.field "codec" Codec_id.encode c.codec;
      W.field "element" encode_element c.element;
      W.field "byte_width" W.enc_int c.byte_width;
      W.field "element_inspection" encode_support c.element_inspection;
      W.field "exact_values" encode_support c.exact_values;
    ]

let decode_operation : Operation.t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* operation = W.Obj.required o "operation" Operation_id.decode in
      let* implementation_digest =
        W.Obj.required o "implementation_digest" Wire_digest.implementation
      in
      let* arity = W.Obj.required o "arity" positive_width in
      let* lossless = W.Obj.required o "lossless" decode_support in
      Ok { Operation.operation; implementation_digest; arity; lossless })

let encode_operation (op : Operation.t) =
  W.obj
    [
      W.field "operation" Operation_id.encode op.operation;
      W.field "implementation_digest" Wire_digest.encode_implementation
        op.implementation_digest;
      W.field "arity" W.enc_int op.arity;
      W.field "lossless" encode_support op.lossless;
    ]

let decode_metric : Metric.t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* comparator = W.Obj.required o "comparator" Comparator_id.decode in
      let* implementation_digest =
        W.Obj.required o "implementation_digest" Wire_digest.implementation
      in
      let* threshold_direction =
        W.Obj.required o "threshold_direction" decode_direction
      in
      let* finite_operands_only =
        W.Obj.required o "finite_operands_only" W.bool
      in
      Ok
        {
          Metric.comparator;
          implementation_digest;
          threshold_direction;
          finite_operands_only;
        })

let encode_metric (m : Metric.t) =
  W.obj
    [
      W.field "comparator" Comparator_id.encode m.comparator;
      W.field "implementation_digest" Wire_digest.encode_implementation
        m.implementation_digest;
      W.field "threshold_direction" encode_direction m.threshold_direction;
      W.field "finite_operands_only" W.enc_bool m.finite_operands_only;
    ]

let decode_conversion : Conversion.t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* source = W.Obj.required o "source" Codec_id.decode in
      let* target = W.Obj.required o "target" Codec_id.decode in
      let* lossless = W.Obj.required o "lossless" decode_support in
      Ok { Conversion.source; target; lossless })

let encode_conversion (c : Conversion.t) =
  W.obj
    [
      W.field "source" Codec_id.encode c.source;
      W.field "target" Codec_id.encode c.target;
      W.field "lossless" encode_support c.lossless;
    ]
