(* The owned [exp] against the host libm, bit for bit: the model is the
   specification the native realizations are checked against, so it has to be
   the function the reference computes. *)

open Machine_ir

let same a b =
  if Float.is_nan a then Float.is_nan b
  else Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

let differences inputs =
  List.fold_left
    (fun n x -> if same (Mir_exp.model x) (Stdlib.exp x) then n else n + 1)
    0 inputs

let specials =
  [
    0.;
    -0.;
    1.;
    -1.;
    Float.nan;
    Float.infinity;
    Float.neg_infinity;
    0x1p-54;
    0x1p-55;
    -0x1p-54;
    -0x1p-55;
    Float.min_float;
    5e-324;
    511.99999999999994;
    512.;
    -512.;
    709.782712893384;
    709.782712893385;
    710.;
    1023.9999999999999;
    1024.;
    -745.1332191019411;
    -745.1332191019412;
    -708.3964185322641;
    -708.4;
    -1023.9999999999999;
    -1024.;
    1e300;
    -1e300;
    Float.max_float;
  ]

(* Inputs whose scaled value lands exactly on a half integer, where rounding
   half away from zero and half to even choose different reduction steps. *)
let ties =
  List.concat_map
    (fun k ->
      let x0 = (float_of_int k +. 0.5) /. Mir_exp.invln2n in
      List.filter
        (fun x ->
          let z = x *. Mir_exp.invln2n in
          Float.abs (z -. Float.trunc z) = 0.5)
        [
          Float.pred (Float.pred x0);
          Float.pred x0;
          x0;
          Float.succ x0;
          Float.succ (Float.succ x0);
        ])
    (List.init 2001 (fun k -> k - 1000))

(* Inputs whose scaled value falls just short of a half integer below a power
   of two, where adding one half would round up to the wrong integer. *)
let binade_edges =
  List.concat_map
    (fun n ->
      List.concat_map
        (fun sign ->
          let x0 = sign *. (Float.ldexp 1. n -. 0.5) /. Mir_exp.invln2n in
          let rec walk k x acc =
            if k = 0 then acc else walk (k - 1) (Float.pred x) (x :: acc)
          in
          let rec climb k x =
            if k = 0 then x else climb (k - 1) (Float.succ x)
          in
          walk 400 (climb 200 x0) [])
        [ 1.; -1. ])
    (List.init 18 Fun.id)

let uniform seed lo hi n =
  let st = Random.State.make [| seed |] in
  List.init n (fun _ -> lo +. Random.State.float st (hi -. lo))

let patterns seed n =
  let st = Random.State.make [| seed |] in
  List.init n (fun _ -> Int64.float_of_bits (Random.State.bits64 st))

let%expect_test "owned exp equals the host libm" =
  let report name inputs =
    Fmt.pr "%s: %d inputs, %d differences@." name (List.length inputs)
      (differences inputs)
  in
  report "specials" specials;
  report "unit interval" (uniform 1 (-1.) 1. 200_000);
  report "model range" (uniform 2 (-20.) 20. 400_000);
  report "wide" (uniform 3 (-745.2) 709.8 400_000);
  report "special-case band" (uniform 4 (-1024.) 1024. 200_000);
  report "half-integer reductions" ties;
  report "binade edges" binade_edges;
  report "bit patterns" (patterns 5 400_000);
  [%expect
    {|
    specials: 30 inputs, 0 differences
    unit interval: 200000 inputs, 0 differences
    model range: 400000 inputs, 0 differences
    wide: 400000 inputs, 0 differences
    special-case band: 200000 inputs, 0 differences
    half-integer reductions: 2015 inputs, 0 differences
    binade edges: 14400 inputs, 0 differences
    bit patterns: 400000 inputs, 0 differences |}]
