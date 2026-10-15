(* The binary32 transcendentals against the correctly rounded binary64 libm
   result, in ulps, over a stratified sample (every 4093rd bit pattern) of the
   binary32 line. The bounds are frozen: a change that loosens one fails here.
   The full 2^32 sweeps are [accuracy.exe], recorded in the design record. *)

let ordered x =
  let b = Int32.to_int (Int32.bits_of_float x) land 0xFFFFFFFF in
  if b land 0x80000000 <> 0 then -(b land 0x7FFFFFFF) else b

let sweep ?(lo = neg_infinity) ?(hi = infinity) ~stride f g =
  let worst = ref 0 and n = ref 0 and bits = ref 0 in
  while !bits < 0x1_0000_0000 do
    let x = Int32.float_of_bits (Int32.of_int !bits) in
    if Float.is_finite x && x >= lo && x <= hi then (
      incr n;
      let want = F32_math.round32 (g x) and got = f x in
      let d =
        if Float.is_nan want && Float.is_nan got then 0
        else if Float.is_nan want || Float.is_nan got then max_int
        else abs (ordered want - ordered got)
      in
      worst := max !worst d);
    bits := !bits + stride
  done;
  (!n, !worst)

let%expect_test "accuracy, frozen" =
  let show name (n, w) = Fmt.pr "%-22s %7d inputs, max %d ulp@." name n w in
  show "exp" (sweep ~stride:4093 F32_math.Value.exp Stdlib.exp);
  show "log" (sweep ~stride:4093 F32_math.Value.log Stdlib.log);
  show "sin |x| <= 100"
    (sweep ~lo:(-100.) ~hi:100. ~stride:997 F32_math.Value.sin Stdlib.sin);
  show "cos |x| <= 100"
    (sweep ~lo:(-100.) ~hi:100. ~stride:997 F32_math.Value.cos Stdlib.cos);
  show "sin |x| <= 8192"
    (sweep ~lo:(-8192.) ~hi:8192. ~stride:4093 F32_math.Value.sin Stdlib.sin);
  show "cos |x| <= 8192"
    (sweep ~lo:(-8192.) ~hi:8192. ~stride:4093 F32_math.Value.cos Stdlib.cos);
  [%expect
    {|
    exp                    1045246 inputs, max 1 ulp
    log                    1045246 inputs, max 1 ulp
    sin |x| <= 100         2247550 inputs, max 1 ulp
    cos |x| <= 100         2247550 inputs, max 1 ulp
    sin |x| <= 8192         573861 inputs, max 8 ulp
    cos |x| <= 8192         573861 inputs, max 1 ulp |}]

(* Special values: what the C library gives, in the places a binary32 function
   defines one. A NaN prints without its sign, which is the host's: 0/0 is
   negative on x86-64 and positive on AArch64. *)
let%expect_test "special values" =
  let show name f xs =
    Fmt.pr "%s:" name;
    List.iter
      (fun x ->
        let y = f x in
        if Float.is_nan y then Fmt.pr " nan" else Fmt.pr " %h" y)
      xs;
    Fmt.pr "@."
  in
  let xs = [ 0.; -0.; 1.; -1.; infinity; neg_infinity; nan; 1e-45; 3e38 ] in
  show "exp" F32_math.Value.exp xs;
  show "log" F32_math.Value.log xs;
  show "sin" F32_math.Value.sin [ 0.; -0.; 1.; nan ];
  show "cos" F32_math.Value.cos [ 0.; -0.; 1.; nan ];
  show "erf" F32_math.Value.erf [ 0.; -0.; 1.; -1.; 3.; nan ];
  [%expect
    {|
    exp: 0x1p+0 0x1p+0 0x1.5bf0a8p+1 0x1.78b564p-2 infinity 0x0p+0 nan 0x1p+0 infinity
    log: -infinity -infinity 0x0p+0 nan infinity nan nan -0x1.9d1dap+6 0x1.62632cp+6
    sin: 0x0p+0 0x0p+0 0x1.aed548p-1 nan
    cos: 0x1p+0 0x1p+0 0x1.14a28p-1 nan
    erf: 0x0p+0 0x0p+0 0x1.af7678p-1 -0x1.af7678p-1 0x1.fffd1ap-1 nan |}]
