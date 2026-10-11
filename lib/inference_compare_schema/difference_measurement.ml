module W = Wire

type unavailable = Arithmetic_overflow | Nonfinite_operands | Zero_over_zero
type magnitude = Finite of float | Unavailable of unavailable | Unbounded
type sample = { index : int64 list; absolute : magnitude; relative : magnitude }

let decode_unavailable, encode_unavailable =
  W.enum ~kind:"unavailable"
    [
      (Arithmetic_overflow, "arithmetic_overflow");
      (Nonfinite_operands, "nonfinite_operands");
      (Zero_over_zero, "zero_over_zero");
    ]

let decode_magnitude : magnitude W.decoder =
  W.variant (fun tag o ->
      let ( let* ) = W.( let* ) in
      match tag with
      | "finite" ->
          let* v = W.Obj.required o "value" W.float in
          Ok (Finite v)
      | "unavailable" ->
          let* r = W.Obj.required o "reason" decode_unavailable in
          Ok (Unavailable r)
      | "unbounded" -> Ok Unbounded
      | tag -> W.unknown_tag ~kind:"magnitude" tag)

let encode_magnitude = function
  | Finite v -> W.tagged "finite" [ W.field "value" W.enc_float v ]
  | Unavailable r ->
      W.tagged "unavailable" [ W.field "reason" encode_unavailable r ]
  | Unbounded -> W.tagged "unbounded" []

let decode_sample : sample W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* index = W.Obj.required o "index" (W.list W.nonneg_int64) in
      let* absolute = W.Obj.required o "absolute" decode_magnitude in
      let* relative = W.Obj.required o "relative" decode_magnitude in
      Ok { index; absolute; relative })

let encode_sample s =
  W.obj
    [
      W.field "index" (W.enc_list W.enc_int64) s.index;
      W.field "absolute" encode_magnitude s.absolute;
      W.field "relative" encode_magnitude s.relative;
    ]
