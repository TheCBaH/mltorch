(* Affine (asymmetric) quantization: real = scale * (q - zero_point). Granularity
   is per-tensor or per-channel along C (indexed by the channel coordinate). Both
   the scale and the zero point are binary32 values (each satisfies
   [round32 x = x]), held as [float]: a producer's binary64 scale is rounded
   once at the boundary ({!round_scale}), never inside a kernel, so a
   dequantization is operations over binary32 constants. The quanta [q] are
   plain signed ints in [qmin, qmax]. See .ai/native_tensor_design.md §3. *)

type t =
  | Per_channel of { scale : float array; zero_point : float array }
  | Per_tensor of { scale : float; zero_point : float }
(* equal lengths, by construction *)

type error =
  [ `Quant_array_lengths of int * int
  | `Quant_not_binary32 of [ `Scale | `Zero_point ] * float ]

let pp_error fmt : [< error ] -> unit = function
  | `Quant_array_lengths (s, z) ->
      Fmt.pf fmt "quant scale/zero_point lengths differ: %d vs %d" s z
  | `Quant_not_binary32 (which, v) ->
      Fmt.pf fmt "quant %s %h is not a binary32 value"
        (match which with `Scale -> "scale" | `Zero_point -> "zero_point")
        v

let round32 x = Int32.float_of_bits (Int32.bits_of_float x)

(* A binary32 value: finite, or an infinity or NaN no quantization uses. *)
let is_binary32 x =
  Float.is_finite x && Core.Float_bits.equal_exact (round32 x) x

let round_scale x =
  let r = round32 x in
  (r, not (Core.Float_bits.equal_exact r x))

let check which x =
  if is_binary32 x then Ok () else Error (`Quant_not_binary32 (which, x))

let per_tensor_checked ~scale ~zero_point =
  match (check `Scale scale, check `Zero_point zero_point) with
  | Ok (), Ok () -> Err.return (Per_tensor { scale; zero_point })
  | (Error _ as e), _ | _, (Error _ as e) -> (
      match e with Error k -> Err.fail k | Ok () -> assert false)

let per_tensor ~scale ~zero_point =
  match Err.payload (per_tensor_checked ~scale ~zero_point) with
  | Ok t -> t
  | Error e ->
      invalid_arg (Fmt.str "Quant.per_tensor: %a" pp_error (e :> error))

(* [Array.copy] on both: the caller keeps its arrays and may mutate them, and
   nothing below hands the stored ones back out. *)
let per_channel ~scale ~zero_point =
  let ls = Array.length scale and lz = Array.length zero_point in
  if ls <> lz then Err.fail (`Quant_array_lengths (ls, lz))
  else
    let bad which a =
      Array.fold_left
        (fun acc x ->
          match acc with
          | Some _ -> acc
          | None ->
              if is_binary32 x then None
              else Some (`Quant_not_binary32 (which, x)))
        None a
    in
    match (bad `Scale scale, bad `Zero_point zero_point) with
    | Some e, _ | None, Some e -> Err.fail e
    | None, None ->
        Err.return
          (Per_channel
             { scale = Array.copy scale; zero_point = Array.copy zero_point })

let channel_count = function
  | Per_channel { scale; _ } -> Some (Array.length scale)
  | Per_tensor _ -> None

let equal a b =
  match (a, b) with
  | Per_channel x, Per_channel y ->
      Array.length x.scale = Array.length y.scale
      (* [for_all2] would raise on a length mismatch, which the guard above
            already rules out; the explicit check keeps that local. *)
      && Array.for_all2 Core.Float_bits.equal_exact x.scale y.scale
      && Array.for_all2 Core.Float_bits.equal_exact x.zero_point y.zero_point
  | Per_channel _, Per_tensor _ | Per_tensor _, Per_channel _ -> false
  | Per_tensor x, Per_tensor y ->
      Core.Float_bits.equal_exact x.scale y.scale
      && Core.Float_bits.equal_exact x.zero_point y.zero_point

let of_err_for_jsont r =
  match Err.export ~pos:__POS__ r with
  | Ok v -> v
  | Error k -> Jsont.Error.msgf Jsont.Meta.none "%a" pp_error k

let params t ~(c : Dim.index Dim.t) =
  match t with
  | Per_channel { scale; zero_point } ->
      (scale.((c :> int)), zero_point.((c :> int)))
  | Per_tensor { scale; zero_point } -> (scale, zero_point)

(* The binary64 reference: [scale * (q - zero_point)] over the binary32
   parameters. [q - zero_point] is exact when the zero point is an integer
   (the only kind a producer gives), and the product of a binary32 scale and a
   difference of at most 25 significant bits is exact too. *)
let dequantize t ~c ~q =
  let scale, zero_point = params t ~c in
  scale *. (float_of_int q -. zero_point)

(* The binary32 meaning, as explicit operations each rounded once: [q] to float
   (exact: [|q| < 2^15]), minus the zero point, times the scale, in that order. *)
let dequantize32 t ~c ~q =
  let scale, zero_point = params t ~c in
  round32 (round32 (float_of_int q -. zero_point) *. scale)

(* Round half to even, on a finite value. *)
let round_half_even v =
  let r = Float.round v in
  if Float.abs (v -. Float.trunc v) = 0.5 then 2. *. Float.round (v /. 2.)
  else r

(* [round (x / scale + zero_point)] half to even, clamped to [qmin, qmax] in
   float before any conversion to an integer: an infinity or a value far
   outside the range saturates, and a NaN takes the level the zero point
   names (clamped), so no [int] is ever made from an unspecified conversion. *)
let quantize t ~c ~qmin ~qmax x =
  let scale, zero_point = params t ~c in
  let lo = float_of_int qmin and hi = float_of_int qmax in
  let clamp v = if v < lo then lo else if v > hi then hi else v in
  let v = (x /. scale) +. zero_point in
  let level =
    if Float.is_nan v then clamp (round_half_even zero_point)
    else clamp (round_half_even v)
  in
  int_of_float level

let pp fmt = function
  | Per_channel { scale; _ } ->
      Format.fprintf fmt "Per_channel(%d)" (Array.length scale)
  | Per_tensor { scale; zero_point } ->
      Format.fprintf fmt "Per_tensor s=%g zero_point=%g" scale zero_point

(* An integer zero point is written as the integer it always was; any other
   binary32 value as a binary32 number. *)
let enc_zero_point z =
  if Float.is_integer z && Float.abs z < 1073741824. then
    Json_util.jint (int_of_float z)
  else Json_util.enc Json_util.f32_jsont z

let jsont : t Jsont.t =
  Jsont.map ~kind:"quant"
    ~dec:(fun json ->
      Json_util.union ~kind:"quant"
        [
          ( "Per_channel",
            fun v ->
              let ms = Json_util.req_obj v "Per_channel" in
              let get k c = Json_util.req_field ms k c "Per_channel" in
              let scale =
                Array.of_list (get "scale" (Jsont.list Json_util.f32_jsont))
              in
              let zero_point =
                Array.of_list
                  (get "zero_point" (Jsont.list Json_util.f32_jsont))
              in
              (* Through the constructor, so a document with mismatched arrays
                 is rejected here rather than raising later out of [params]. *)
              of_err_for_jsont (per_channel ~scale ~zero_point) );
          ( "Per_tensor",
            fun v ->
              let ms = Json_util.req_obj v "Per_tensor" in
              let get k c = Json_util.req_field ms k c "Per_tensor" in
              let scale = get "scale" Json_util.f32_jsont in
              let zero_point = get "zero_point" Json_util.f32_jsont in
              per_tensor ~scale ~zero_point );
        ]
        json)
    ~enc:(fun q ->
      match q with
      | Per_channel { scale; zero_point } ->
          Json_util.single ~case:"Per_channel"
            (Json_util.jobj
               [
                 ( "scale",
                   Json_util.jarr
                     (List.map
                        (Json_util.enc Json_util.f32_jsont)
                        (Array.to_list scale)) );
                 ( "zero_point",
                   Json_util.jarr
                     (List.map enc_zero_point (Array.to_list zero_point)) );
               ])
      | Per_tensor { scale; zero_point } ->
          Json_util.single ~case:"Per_tensor"
            (Json_util.jobj
               [
                 ("scale", Json_util.enc Json_util.f32_jsont scale);
                 ("zero_point", enc_zero_point zero_point);
               ]))
    Jsont.json
