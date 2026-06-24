(* Affine (asymmetric) quantization: real = scale * (q - zero_point). Granularity
   is per-tensor or per-channel along C (indexed by the channel coordinate). The
   scale is a float and the zero_point / quanta are plain signed ints (domain data
   in [qmin, qmax], not Dim sizes). See .ai/native_tensor_design.md §3. *)

type t =
  | Per_tensor of { scale : float; zero_point : int }
  | Per_channel of { scale : float array; zero_point : int array }
(* length = C *)

let params t ~c =
  match t with
  | Per_tensor { scale; zero_point } -> (scale, zero_point)
  | Per_channel { scale; zero_point } -> (scale.(c), zero_point.(c))

let dequantize t ~c ~q =
  let scale, zp = params t ~c in
  scale *. float_of_int (q - zp)

let quantize t ~c ~qmin ~qmax x =
  let scale, zp = params t ~c in
  let v = int_of_float (Float.round (x /. scale)) + zp in
  if v < qmin then qmin else if v > qmax then qmax else v

let pp fmt = function
  | Per_tensor { scale; zero_point } ->
      Format.fprintf fmt "Per_tensor s=%g zp=%d" scale zero_point
  | Per_channel { scale; _ } ->
      Format.fprintf fmt "Per_channel(%d)" (Array.length scale)
