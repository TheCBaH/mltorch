let bf16_to_f32_halves u = (u, 0)

let f16_to_f32_halves u =
  let sign = (u lsr 15) land 1 in
  let exp = (u lsr 10) land 0x1F in
  let mant = u land 0x3FF in
  let pack e m =
    (* binary32: sign | 8-bit exponent | 23-bit mantissa, as two 16-bit halves. *)
    ((sign lsl 15) lor (e lsl 7) lor (m lsr 16), m land 0xFFFF)
  in
  if exp = 0 then
    if mant = 0 then pack 0 0
    else begin
      (* Subnormal: shift the leading one up to bit 10; each shift lowers the
         exponent. A binary16 subnormal is mant * 2^-24, a binary32 normal
         (1.f) * 2^(e - 127). *)
      let rec normalize m k =
        if m land 0x400 <> 0 then (m, k) else normalize (m lsl 1) (k + 1)
      in
      let m, k = normalize mant 0 in
      pack (113 - k) ((m land 0x3FF) lsl 13)
    end
  else if exp = 0x1F then
    if mant = 0 then pack 0xFF 0 else pack 0xFF ((mant lsl 13) lor 0x400000)
  else pack (exp + 112) (mant lsl 13)

let widen from ~(src : Pt2_storage.t) ~(dst : Pt2_storage.t) =
  let halves =
    match (from : Dtype.t) with
    | BF16 -> bf16_to_f32_halves
    | F16 -> f16_to_f32_halves
    | _ -> invalid_arg "Widen.widen: only BF16 and F16 widen to F32"
  in
  let n = Bigarray.Array1.dim src in
  if n land 1 = 1 || Bigarray.Array1.dim dst <> 2 * n then
    invalid_arg "Widen.widen: destination must be twice the source length";
  let get = Bigarray.Array1.unsafe_get and set = Bigarray.Array1.unsafe_set in
  for i = 0 to (n / 2) - 1 do
    let u =
      Char.code (get src (2 * i)) lor (Char.code (get src ((2 * i) + 1)) lsl 8)
    in
    let hi, lo = halves u in
    set dst (4 * i) (Char.unsafe_chr (lo land 0xFF));
    set dst ((4 * i) + 1) (Char.unsafe_chr (lo lsr 8));
    set dst ((4 * i) + 2) (Char.unsafe_chr (hi land 0xFF));
    set dst ((4 * i) + 3) (Char.unsafe_chr (hi lsr 8))
  done
