(* Builders for the numerical-comparison tests. *)

open Inference_compare_schema
module Codecs = Inference_compare_codecs
module C = Codec_fixtures
module L = Codecs.Logical_value

let ctx = C.context ()

let f32_of_bits bits_list =
  let b = Bytes.create (4 * List.length bits_list) in
  List.iteri
    (fun i bits -> Bytes.set_int32_le b (i * 4) (Int64.to_int32 bits))
    bits_list;
  C.ok
    (L.make ctx
       (C.f32 [ Int64.of_int (List.length bits_list) ])
       (Bytes.to_string b))

let f64 values =
  C.floats ctx (C.f64 [ Int64.of_int (List.length values) ]) values

let f32 values =
  C.floats ctx (C.f32 [ Int64.of_int (List.length values) ]) values

let f64_shaped shape values = C.floats ctx (C.f64 shape) values

let i64 values =
  C.integers ctx (C.i64 [ Int64.of_int (List.length values) ]) values

let float_policy ?(atol = 0.0) ?(rtol = 0.0) ?(scale = Float_policy.Right_scale)
    ?(arithmetic = Float_policy.Binary64) ?(nan = Float_policy.Reject)
    ?(signed_zero = Float_policy.Equal) ?(required = Float_policy.Tolerance) ()
    : Float_policy.t =
  {
    atol;
    rtol;
    scale;
    arithmetic;
    nan;
    infinity = Same_sign;
    signed_zero;
    required;
  }

let policy ?(element = None)
    ?(direction = Difference_statistics.Right_minus_left)
    ?(variance = Difference_statistics.Population)
    ?(relative_scale = Float_policy.Right_scale) ?(metrics = []) () :
    Comparison_policy.t =
  {
    element;
    statistics = { direction; variance };
    relative_scale;
    tensor_metrics = metrics;
  }

let floating ?atol ?rtol ?scale ?arithmetic ?nan ?signed_zero ?required () =
  Some
    (Element_policy.Floating
       (float_policy ?atol ?rtol ?scale ?arithmetic ?nan ?signed_zero ?required
          ()))

let pp_error = Inference_compare_metrics.Element_compare.pp_error

let element ?max_samples p ~left ~right =
  match
    Inference_compare_metrics.Element_compare.compare ?max_samples ctx p ~left
      ~right
  with
  | Ok r -> r
  | Error e ->
      Format.kasprintf failwith "element compare: %a" pp_error
        (Err.Error.kind e)

let g = Fmt.float

let pp_scalar ppf : Difference_statistics.scalar -> unit = function
  | Finite_value v -> Fmt.pf ppf "%.6g" v
  | Unavailable Empty_finite_set -> Fmt.string ppf "n/a(empty)"
  | Unavailable Insufficient_samples -> Fmt.string ppf "n/a(insufficient)"
  | Unavailable Numeric_overflow -> Fmt.string ppf "n/a(overflow)"
  | Unavailable Unsupported_dtype -> Fmt.string ppf "n/a(dtype)"

let pp_mag ppf : Difference_measurement.magnitude -> unit = function
  | Finite v -> Fmt.pf ppf "%.6g" v
  | Unbounded -> Fmt.string ppf "unbounded"
  | Unavailable Arithmetic_overflow -> Fmt.string ppf "overflow"
  | Unavailable Nonfinite_operands -> Fmt.string ppf "nonfinite"
  | Unavailable Zero_over_zero -> Fmt.string ppf "0/0"

let pp_stats ppf (s : Difference_statistics.t) =
  Fmt.pf ppf "pairs=%Ld +%Ld -%Ld 0:%Ld bias=%a mae=%a rmse=%a std=%a"
    s.finite_pairs s.positive_errors s.negative_errors s.zero_errors pp_scalar
    s.bias pp_scalar s.mean_absolute_error pp_scalar s.root_mean_square_error
    pp_scalar s.standard_deviation
