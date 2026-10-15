(* [accuracy FUNCTION STRIDE [LO HI]]: the largest distance in binary32 ulps
   between [F32_math.Value.FUNCTION] and the correctly rounded binary64 libm
   result, over every STRIDE-th binary32 bit pattern in [LO, HI] (default the
   whole line), with the count of inputs off by one ulp or more. Stride 1 is the
   full 2^32 sweep. *)

let ordered x =
  (* a monotone integer for finite and infinite binary32 values *)
  let b = Int32.to_int (Int32.bits_of_float x) land 0xFFFFFFFF in
  if b land 0x80000000 <> 0 then -(b land 0x7FFFFFFF) else b

let ulps a b = abs (ordered a - ordered b)

let functions =
  [
    ("exp", F32_math.Value.exp, Stdlib.exp);
    ("log", F32_math.Value.log, Stdlib.log);
    ("sin", F32_math.Value.sin, Stdlib.sin);
    ("cos", F32_math.Value.cos, Stdlib.cos);
  ]

let () =
  let name = Sys.argv.(1) and stride = int_of_string Sys.argv.(2) in
  let lo =
    if Array.length Sys.argv > 3 then float_of_string Sys.argv.(3)
    else neg_infinity
  in
  let hi =
    if Array.length Sys.argv > 4 then float_of_string Sys.argv.(4) else infinity
  in
  let _, f, g = List.find (fun (n, _, _) -> n = name) functions in
  let worst = ref 0 and at = ref 0. and off = ref 0 and n = ref 0 in
  let bits = ref 0 in
  while !bits < 0x1_0000_0000 do
    let x = Int32.float_of_bits (Int32.of_int !bits) in
    if Float.is_finite x && x >= lo && x <= hi then (
      incr n;
      let want = F32_math.round32 (g x) and got = f x in
      let d =
        if Float.is_nan want && Float.is_nan got then 0
        else if Float.is_nan want || Float.is_nan got then max_int
        else ulps want got
      in
      if d > 0 then incr off;
      if d > !worst then (
        worst := d;
        at := x));
    bits := !bits + stride
  done;
  Printf.printf "%s: %d inputs, max %d ulp at %h, off by one or more: %d\n" name
    !n !worst !at !off
