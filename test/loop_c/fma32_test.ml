open Loop_ir

(* [Loop_numerics.fma32] against the host's [fmaf], bit for bit, over the edge
   values and a large generated sample. The sample is built so double rounding
   would show: products whose exact sum with c lands within a binary64 ulp of a
   binary32 midpoint. *)

let c_source =
  String.concat "\n"
    [
      "#include <math.h>";
      "#include <stdint.h>";
      "#include <stdio.h>";
      "#include <string.h>";
      "int main(int argc, char **argv) {";
      "  (void)argc;";
      "  FILE *in = fopen(argv[1], \"rb\"), *out = fopen(argv[2], \"wb\");";
      "  float t[3];";
      "  while (fread(t, sizeof t[0], 3, in) == 3) {";
      "    float r = fmaf(t[0], t[1], t[2]);";
      "    uint32_t b; memcpy(&b, &r, 4);";
      "    fwrite(&b, 4, 1, out);";
      "  }";
      "  fclose(out); return 0;";
      "}";
      "";
    ]

let specials =
  [|
    0.;
    -0.;
    1.;
    -1.;
    0.5;
    3.;
    1.0000001;
    16777216.;
    16777217.;
    3.4028235e38;
    -3.4028235e38;
    1.17549435e-38;
    1e-45;
    infinity;
    neg_infinity;
    nan;
    1.1920929e-07;
    8388608.;
    8388609.;
    0.1;
    0.3;
    1e20;
    1e-20;
  |]

let samples =
  let r = Loop_numerics.round32 in
  let st = ref 0x2545F4914F6CDD1DL in
  let next () =
    (st := Int64.(logxor !st (shift_left !st 13)));
    (st := Int64.(logxor !st (shift_right_logical !st 7)));
    (st := Int64.(logxor !st (shift_left !st 17)));
    !st
  in
  let f32_of_bits b = Int32.float_of_bits (Int64.to_int32 b) in
  let random () =
    (* any bit pattern: every exponent, subnormals, nonfinite included *)
    f32_of_bits (next ())
  in
  let moderate () =
    (* moderate exponents, where the sum is near a rounding boundary often *)
    Int32.float_of_bits
      (Int32.logor 0x3F000000l
         (Int32.logand (Int64.to_int32 (next ())) 0x00FFFFFFl))
    *. if Int64.logand (next ()) 1L = 0L then 1. else -1.
  in
  let acc = ref [] in
  Array.iter
    (fun a ->
      Array.iter
        (fun b -> Array.iter (fun c -> acc := (a, b, c) :: !acc) specials)
        specials)
    specials;
  for _ = 1 to 20000 do
    acc := (random (), random (), random ()) :: !acc
  done;
  for _ = 1 to 40000 do
    let a = moderate () and b = moderate () in
    (* c close to -a*b so cancellation exposes the low bits *)
    let c =
      Loop_numerics.round32
        (-.(a *. b)
        *. (1. +. (float_of_int (Int64.to_int (Int64.rem (next ()) 7L)) *. 1e-7))
        )
    in
    acc := (a, b, c) :: !acc
  done;
  (* Constructed double-rounding cases. With a = (1 + 2^-12) 2^(m-24), b = 1 -
     2^-12 + 2^-24 and c = 2^m the exact sum is 2^m + 2^(m-24) + 2^(m-60): a
     hair above the midpoint of two binary32 neighbours, below half a binary64
     ulp. Rounded to binary64 first it lands ON the midpoint and ties to even,
     which is the wrong neighbour; fmaf rounds up. *)
  for m = -20 to 20 do
    let two k = Float.ldexp 1. k in
    let a = (1. +. two (-12)) *. two (m - 24)
    and b = 1. -. two (-12) +. two (-24)
    and c = two m in
    acc := (a, b, c) :: (-.a, b, -.c) :: !acc
  done;
  (* binary32 operands, as the host receives them *)
  List.rev_map (fun (a, b, c) -> (r a, r b, r c)) !acc

let%expect_test "fma32 equals the host's fmaf, bit for bit" =
  let dir = Loop_c_exec.Proc.temp_dir "fma32" in
  Fun.protect ~finally:(fun () -> Loop_c_exec.Proc.remove_tree dir) @@ fun () ->
  let exe =
    match Loop_c_exec.compile c_source with
    | Ok e -> e
    | Error (`C_compile m | `C_host m) -> failwith m
  in
  let input = Filename.concat dir "in.bin"
  and output = Filename.concat dir "out.bin" in
  let buf = Buffer.create (12 * List.length samples) in
  List.iter
    (fun (a, b, c) ->
      List.iter
        (fun x -> Buffer.add_int32_le buf (Int32.bits_of_float x))
        [ a; b; c ])
    samples;
  Loop_c_exec.Proc.write_file input (Buffer.contents buf);
  (match Loop_c_exec.Proc.run [ exe; input; output ] with
  | Ok (Loop_c_exec.Proc.Exited 0, _) -> ()
  | _ -> failwith "fmaf host failed");
  let got = Loop_c_exec.Proc.read_file output in
  let bad = ref 0 and nan_cases = ref 0 in
  List.iteri
    (fun i (a, b, c) ->
      let want = Int32.float_of_bits (String.get_int32_le got (4 * i)) in
      let have = Loop_numerics.fma32 a b c in
      if Float.is_nan want then (
        incr nan_cases;
        if not (Float.is_nan have) then incr bad)
      else if Int32.bits_of_float want <> Int32.bits_of_float have then (
        if !bad < 5 then
          Fmt.pr "MISMATCH fma(%h, %h, %h): host %h, ours %h@." a b c want have;
        incr bad))
    samples;
  Fmt.pr "%d samples (%d NaN results), %d mismatches@." (List.length samples)
    !nan_cases !bad;
  [%expect {| 72249 samples (2010 NaN results), 0 mismatches |}]

(* The check must be able to fail: the textbook shortcut, round32 of the double
   fma, is NOT fmaf, and the sample finds it. *)
let%expect_test "the double-rounding shortcut is caught by the same sample" =
  let wrong = ref 0 in
  List.iter
    (fun (a, b, c) ->
      let shortcut = Loop_numerics.round32 (Float.fma a b c) in
      let ours = Loop_numerics.fma32 a b c in
      if
        (not (Float.is_nan ours))
        && Int32.bits_of_float shortcut <> Int32.bits_of_float ours
      then incr wrong)
    samples;
  Fmt.pr "shortcut differs on %s@."
    (if !wrong > 0 then "some samples" else "NONE");
  [%expect {| shortcut differs on some samples |}]
