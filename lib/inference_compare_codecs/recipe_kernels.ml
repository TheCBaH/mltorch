open Inference_compare_schema
module E = Element_codec

let prod = List.fold_left ( * ) 1

let permute ~width ~axes ~shape src =
  let rank = List.length shape in
  let shape_a = Array.of_list shape in
  let axes_a = Array.of_list axes in
  let in_strides = Array.of_list (Shape_math.strides shape) in
  let out_dims = Array.init rank (fun j -> shape_a.(axes_a.(j))) in
  let total = prod shape in
  let dst = Bytes.create (total * width) in
  (* An odometer over the output index; [offset] tracks the matching input
     element so no per-element division is needed. *)
  let idx = Array.make rank 0 in
  let offset = ref 0 in
  for o = 0 to total - 1 do
    Bytes.blit_string src (!offset * width) dst (o * width) width;
    let j = ref (rank - 1) in
    let continue = ref true in
    while !continue && !j >= 0 do
      let s = in_strides.(axes_a.(!j)) in
      idx.(!j) <- idx.(!j) + 1;
      offset := !offset + s;
      if idx.(!j) < out_dims.(!j) then continue := false
      else begin
        offset := !offset - (idx.(!j) * s);
        idx.(!j) <- 0;
        decr j
      end
    done
  done;
  Bytes.unsafe_to_string dst

let split_at axis shape =
  let rec go i before = function
    | [] -> (List.rev before, 0, [])
    | d :: after when i = axis -> (List.rev before, d, after)
    | d :: rest -> go (i + 1) (d :: before) rest
  in
  go 0 [] shape

let slice ~width ~axis ~start ~length ~shape src =
  let before, extent, after = split_at axis shape in
  let outer = prod before and inner = prod after * width in
  let dst = Buffer.create (outer * length * inner) in
  for o = 0 to outer - 1 do
    Buffer.add_substring dst src
      (((o * extent) + start) * inner)
      (length * inner)
  done;
  Buffer.contents dst

let concat ~width ~axis operands =
  match operands with
  | [] -> ""
  | (shape0, _) :: _ ->
      let before, _, after = split_at axis shape0 in
      let outer = prod before and inner = prod after * width in
      let extents =
        List.map
          (fun (shape, _) ->
            let _, e, _ = split_at axis shape in
            e)
          operands
      in
      let dst = Buffer.create 64 in
      for o = 0 to outer - 1 do
        List.iter2
          (fun e (_, bytes) ->
            Buffer.add_substring dst bytes (o * e * inner) (e * inner))
          extents operands
      done;
      Buffer.contents dst

(* A source element read as an exact integer or an exact double. *)
type value = I of int64 | F of float

let read (c : E.t) s i =
  if E.is_float c then F (E.float_value c s i) else I (E.integer_value c s i)

(* Inclusive integer range of an integer codec as int64 bounds, or [None] for
   the full int64 range (where range checks are vacuous). *)
let range (c : E.t) =
  match (c.element, c.byte_width) with
  | Boolean, _ -> Some (0L, 1L)
  | Unsigned_integer, 1 -> Some (0L, 255L)
  | Unsigned_integer, 2 -> Some (0L, 65535L)
  | Unsigned_integer, 4 -> Some (0L, 0xFFFF_FFFFL)
  | Signed_integer, 1 -> Some (-128L, 127L)
  | Signed_integer, 2 -> Some (-32768L, 32767L)
  | Signed_integer, 4 -> Some (-2147483648L, 2147483647L)
  | _ -> None

let wrap (c : E.t) v =
  match (c.element, c.byte_width) with
  | Signed_integer, w when w < 8 ->
      let shift = 64 - (8 * w) in
      Int64.shift_right (Int64.shift_left v shift) shift
  | Unsigned_integer, w when w < 8 ->
      Int64.logand v (Int64.sub (Int64.shift_left 1L (8 * w)) 1L)
  | Boolean, _ -> Int64.logand v 1L
  | _ -> v

let round_float (r : Conversion_step.rounding) f =
  match r with
  | Toward_zero -> Float.trunc f
  | Toward_negative -> Float.floor f
  | Toward_positive -> Float.ceil f
  | Nearest_even ->
      (* [Float.round] rounds halves away from zero; correct the exact-half
         case to the even neighbour. *)
      let r = Float.round f in
      if Float.abs (f -. Float.trunc f) = 0.5 then
        let t = Float.trunc f in
        if Float.rem t 2.0 = 0.0 then t else r
      else r

(* binary32 -> binary64 bit pattern. Every finite value and infinity widens
   exactly through a double; a NaN is widened field-wise so its payload and
   quiet bit are not altered by the hardware conversion. *)
let widen32 bits =
  let sign = Int64.shift_left (Int64.shift_right_logical bits 31) 63 in
  let exp =
    Int64.to_int (Int64.logand (Int64.shift_right_logical bits 23) 0xFFL)
  in
  let man = Int64.logand bits 0x7FFFFFL in
  if exp = 0xFF then
    Int64.logor sign
      (Int64.logor (Int64.shift_left 0x7FFL 52) (Int64.shift_left man 29))
  else Int64.bits_of_float (Int32.float_of_bits (Int64.to_int32 bits))

(* The inverse for a NaN: the top 23 mantissa bits. A payload that lives only
   in the dropped low bits would otherwise become an infinity, so the quiet bit
   is set in that case. *)
let narrow64_nan bits =
  let sign = Int64.to_int32 (Int64.shift_right_logical bits 63) in
  let man = Int64.logand bits 0xF_FFFF_FFFF_FFFFL in
  let top = Int64.to_int32 (Int64.shift_right_logical man 29) in
  let top = if Int32.equal top 0l then 0x400000l else top in
  Int32.logor (Int32.shift_left sign 31) (Int32.logor 0x7F800000l top)

let cast ~(source : E.t) ~(target : E.t) ~rounding ~overflow ~count src =
  let dst = Bytes.create (count * target.byte_width) in
  let exception Stop of Codec_error.t in
  let int_target i (v : int64) =
    (* [v] is the exact integer result before the range policy. *)
    let v =
      match range target with
      | None -> v
      | Some (lo, hi) -> (
          if Int64.compare v lo >= 0 && Int64.compare v hi <= 0 then v
          else
            match (overflow : Conversion_step.overflow) with
            | Reject ->
                raise (Stop (Cast_overflow { element = Int64.of_int i }))
            | Saturate -> if Int64.compare v lo < 0 then lo else hi
            | Wrap -> wrap target v)
    in
    E.set_bits target dst i v
  in
  try
    for i = 0 to count - 1 do
      match (read source src i, target.element) with
      | I v, (Signed_integer | Unsigned_integer | Boolean) -> int_target i v
      | I v, Float_binary64 ->
          E.set_bits target dst i (Int64.bits_of_float (Int64.to_float v))
      | I v, Float_binary32 ->
          (* Sources are at most 32 bits (see {!Cast_law}), so the double is
              exact and the single rounding below is the only one. *)
          E.set_bits target dst i
            (Int64.of_int32 (Int32.bits_of_float (Int64.to_float v)))
      | F f, (Signed_integer | Unsigned_integer | Boolean) -> (
          if not (Float.is_finite f) then
            raise (Stop (Cast_value_unavailable { element = Int64.of_int i }))
          else
            let r = round_float rounding f in
            (* 2^63 is exactly representable; int64's maximum is not, so the
                upper test is [>= 2^63]. *)
            if r >= 9223372036854775808.0 then
              match overflow with
              | Saturate -> int_target i Int64.max_int
              | _ -> raise (Stop (Cast_overflow { element = Int64.of_int i }))
            else if r < -9223372036854775808.0 then
              match overflow with
              | Saturate -> int_target i Int64.min_int
              | _ -> raise (Stop (Cast_overflow { element = Int64.of_int i }))
            else
              match overflow with
              | Wrap ->
                  raise (Stop (Cast_overflow { element = Int64.of_int i }))
              | Reject | Saturate -> int_target i (Int64.of_float r))
      | F _, Float_binary64 ->
          (* Raw bits, not a round trip through a double: a signalling NaN
              keeps its payload and its signalling bit. *)
          let raw = E.bits source src i in
          E.set_bits target dst i
            (if source.element = Float_binary32 then widen32 raw else raw)
      | F _, Float_binary32 when source.element = Float_binary32 ->
          E.set_bits target dst i (E.bits source src i)
      | F f, Float_binary32 when Float.is_nan f ->
          E.set_bits target dst i
            (Int64.of_int32 (narrow64_nan (E.bits source src i)))
      | F f, Float_binary32 ->
          let bits = Int32.bits_of_float f in
          let narrowed = Int32.float_of_bits bits in
          if Float.is_finite f && not (Float.is_finite narrowed) then
            match overflow with
            | Saturate ->
                let m = Int32.float_of_bits 0x7F7FFFFFl in
                let m = if f < 0.0 then -.m else m in
                E.set_bits target dst i (Int64.of_int32 (Int32.bits_of_float m))
            | Reject | Wrap ->
                raise (Stop (Cast_overflow { element = Int64.of_int i }))
          else E.set_bits target dst i (Int64.of_int32 bits)
    done;
    Ok (Bytes.unsafe_to_string dst)
  with Stop e -> Error e

let _ = Id.Codec.equal
