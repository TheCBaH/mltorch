module W = Wire

type arithmetic = Binary32 | Binary64
type infinity = Same_sign
type nan = Both_nan | Reject | Same_bits
type requirement = Bitwise | Tolerance
type scale = Left_scale | Right_scale | Symmetric_max_scale
type signed_zero = Distinct | Equal

type t = {
  atol : float;
  rtol : float;
  scale : scale;
  arithmetic : arithmetic;
  nan : nan;
  infinity : infinity;
  signed_zero : signed_zero;
  required : requirement;
}

let decode_arithmetic, encode_arithmetic =
  W.enum ~kind:"arithmetic" [ (Binary32, "binary32"); (Binary64, "binary64") ]

let decode_infinity, encode_infinity =
  W.enum ~kind:"infinity" [ (Same_sign, "same_sign") ]

let decode_nan, encode_nan =
  W.enum ~kind:"nan"
    [ (Both_nan, "both_nan"); (Reject, "reject"); (Same_bits, "same_bits") ]

let decode_requirement, encode_requirement =
  W.enum ~kind:"requirement" [ (Bitwise, "bitwise"); (Tolerance, "tolerance") ]

let decode_scale, encode_scale =
  W.enum ~kind:"scale"
    [
      (Left_scale, "left");
      (Right_scale, "right");
      (Symmetric_max_scale, "symmetric_max");
    ]

let decode_signed_zero, encode_signed_zero =
  W.enum ~kind:"signed_zero" [ (Distinct, "distinct"); (Equal, "equal") ]

let equal a b =
  Float.equal a.atol b.atol && Float.equal a.rtol b.rtol && a.scale = b.scale
  && a.arithmetic = b.arithmetic
  && a.nan = b.nan && a.infinity = b.infinity
  && a.signed_zero = b.signed_zero
  && a.required = b.required

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* atol = W.Obj.required o "atol" W.nonneg_float in
      let* rtol = W.Obj.required o "rtol" W.nonneg_float in
      let* scale = W.Obj.required o "scale" decode_scale in
      let* arithmetic = W.Obj.required o "arithmetic" decode_arithmetic in
      let* nan = W.Obj.required o "nan" decode_nan in
      let* infinity = W.Obj.required o "infinity" decode_infinity in
      let* signed_zero = W.Obj.required o "signed_zero" decode_signed_zero in
      let* required = W.Obj.required o "required" decode_requirement in
      Ok { atol; rtol; scale; arithmetic; nan; infinity; signed_zero; required })

let encode p =
  W.obj
    [
      W.field "atol" W.enc_float p.atol;
      W.field "rtol" W.enc_float p.rtol;
      W.field "scale" encode_scale p.scale;
      W.field "arithmetic" encode_arithmetic p.arithmetic;
      W.field "nan" encode_nan p.nan;
      W.field "infinity" encode_infinity p.infinity;
      W.field "signed_zero" encode_signed_zero p.signed_zero;
      W.field "required" encode_requirement p.required;
    ]
