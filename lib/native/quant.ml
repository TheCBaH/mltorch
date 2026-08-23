(* Affine (asymmetric) quantization: real = scale * (q - zero_point). Granularity
   is per-tensor or per-channel along C (indexed by the channel coordinate). The
   scale is a float and the zero_point / quanta are plain signed ints (domain data
   in [qmin, qmax], not Dim sizes). See .ai/native_tensor_design.md §3. *)

type t =
  | Per_tensor of { scale : float; zero_point : int }
  | Per_channel of { scale : float array; zero_point : int array }
(* equal lengths, by construction *)

type error = [ `Quant_array_lengths of int * int ]

let pp_error fmt : [< error ] -> unit = function
  | `Quant_array_lengths (s, z) ->
      Fmt.pf fmt "quant scale/zero_point lengths differ: %d vs %d" s z

let per_tensor ~scale ~zero_point = Per_tensor { scale; zero_point }

(* [Array.copy] on both: the caller keeps its arrays and may mutate them, and
   nothing below hands the stored ones back out. *)
let per_channel ~scale ~zero_point =
  let ls = Array.length scale and lz = Array.length zero_point in
  if ls <> lz then Err.fail (`Quant_array_lengths (ls, lz))
  else
    Err.return
      (Per_channel
         { scale = Array.copy scale; zero_point = Array.copy zero_point })

let channel_count = function
  | Per_tensor _ -> None
  | Per_channel { scale; _ } -> Some (Array.length scale)

let equal a b =
  match (a, b) with
  | Per_tensor x, Per_tensor y ->
      Core.Float_bits.equal_exact x.scale y.scale && x.zero_point = y.zero_point
  | Per_channel x, Per_channel y ->
      Array.length x.scale = Array.length y.scale
      (* [for_all2] would raise on a length mismatch, which the guard above
            already rules out; the explicit check keeps that local. *)
      && Array.for_all2 Core.Float_bits.equal_exact x.scale y.scale
      && Array.for_all2 ( = ) x.zero_point y.zero_point
  | Per_tensor _, Per_channel _ | Per_channel _, Per_tensor _ -> false

let of_err_for_jsont r =
  match Err.export ~pos:__POS__ r with
  | Ok v -> v
  | Error k -> Jsont.Error.msgf Jsont.Meta.none "%a" pp_error k

let params t ~c =
  match t with
  | Per_tensor { scale; zero_point } -> (scale, zero_point)
  | Per_channel { scale; zero_point } -> (scale.(c), zero_point.(c))

let dequantize t ~c ~q =
  let scale, zero_point = params t ~c in
  scale *. float_of_int (q - zero_point)

let quantize t ~c ~qmin ~qmax x =
  let scale, zero_point = params t ~c in
  let v = int_of_float (Float.round (x /. scale)) + zero_point in
  if v < qmin then qmin else if v > qmax then qmax else v

let pp fmt = function
  | Per_tensor { scale; zero_point } ->
      Format.fprintf fmt "Per_tensor s=%g zero_point=%d" scale zero_point
  | Per_channel { scale; _ } ->
      Format.fprintf fmt "Per_channel(%d)" (Array.length scale)

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
                Array.of_list (get "zero_point" (Jsont.list Jsont.int))
              in
              (* Through the constructor, so a document with mismatched arrays
                 is rejected here rather than raising later out of [params]. *)
              of_err_for_jsont (per_channel ~scale ~zero_point) );
          ( "Per_tensor",
            fun v ->
              let ms = Json_util.req_obj v "Per_tensor" in
              let get k c = Json_util.req_field ms k c "Per_tensor" in
              let scale = get "scale" Json_util.f32_jsont in
              let zero_point = get "zero_point" Jsont.int in
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
                     (List.map Json_util.jint (Array.to_list zero_point)) );
               ])
      | Per_tensor { scale; zero_point } ->
          Json_util.single ~case:"Per_tensor"
            (Json_util.jobj
               [
                 ("scale", Json_util.enc Json_util.f32_jsont scale);
                 ("zero_point", Json_util.jint zero_point);
               ]))
    Jsont.json
