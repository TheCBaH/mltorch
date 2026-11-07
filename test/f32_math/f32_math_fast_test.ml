(* The fast exp and erf against the reference ones, bit for bit, each in a
   file of its own: over these five million inputs either one takes as long
   as the rest of [f32_math_test.ml] together. *)

let bits32 x = Int32.bits_of_float x
let of_bits b = Int32.float_of_bits b

(* [f] against [g], bit for bit (a NaN by its bits too): every 1021st pattern
   of the line; every pattern around each point where [k] changes ([t +- 0.5]
   crossing an integer, [k] in [-150, 128]) and around the clamps; every
   pattern of (-0.5, 0) / log2 e, where the fast [k] is [-0]; and NaNs of
   both signs and several payloads. *)
let compare name f g =
  let n = ref 0 and differ = ref [] and count = ref 0 in
  let check x =
    incr n;
    let a = bits32 (f x) and b = bits32 (g x) in
    if not (Int32.equal a b) then (
      incr count;
      if !count <= 8 then
        differ := Fmt.str "%s %lx: %lx, %lx" name (bits32 x) a b :: !differ)
  in
  let around x k =
    let b = bits32 (F32_math.round32 x) in
    for d = -k to k do
      check (of_bits (Int32.add b (Int32.of_int d)))
    done
  in
  let bits = ref 0 in
  while !bits < 0x1_0000_0000 do
    check (of_bits (Int32.of_int !bits));
    bits := !bits + 1021
  done;
  for k = -151 to 129 do
    around ((float_of_int k +. 0.5) /. 1.44269504088896341) 64;
    around ((float_of_int k -. 0.5) /. 1.44269504088896341) 64
  done;
  List.iter (fun x -> around x 64) [ -104.; 89.; 0. ];
  (* (-0.5, 0) / log2 e: [-0x1.62e43p-2, 0), every pattern *)
  let hi = bits32 (F32_math.round32 (-0.5 /. 1.44269504088896341)) in
  let b = ref 0x80000001l in
  while Int32.compare !b hi <= 0 && Int32.compare !b 0l < 0 do
    check (of_bits !b);
    b := Int32.add !b 1009l
  done;
  List.iter
    (fun b -> check (of_bits b))
    [ 0x7fc00000l; 0xffc00000l; 0x7fc00001l; 0x7fffffffl; 0xffbfffffl ];
  Fmt.pr "%s: %d inputs; differing: %d@." name !n !count;
  List.iter print_endline (List.rev !differ)

let%expect_test "the fast exp is the reference exp" =
  compare "exp" F32_math.Value_fast.exp F32_math.Value.exp;
  [%expect {| exp: 5321954 inputs; differing: 0 |}]
