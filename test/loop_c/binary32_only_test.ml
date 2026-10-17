open Loop_ir

(* The binary32-only helpers ({!Loop_c_dialect.binary32_only}) in the text the
   F-only RV32 target is given, compiled by the host compiler and checked bit
   for bit: [f32_from_i64] against the host's own [(float)] of an [int64_t],
   [i64_from_f32] and [i64_from_f32_failure] against the binary64 helpers they
   stand in for, over the boundaries and a seeded sweep. *)

module R = Loop_c_runtime

let dialect = Loop_c_dialect.Compcert_ilp32

let c_source =
  String.concat "\n"
    [
      R.prelude_in dialect;
      R.helpers ~dialect
        R.Name.
          [
            Bf16_to_f32;
            Bf16_to_float;
            F16_to_f32;
            F16_to_float;
            F32_from_i64;
            I64_from_float;
            I64_from_float_failure;
            I64_from_f32;
            I64_from_f32_failure;
          ];
      "extern void *fopen(const char *, const char *);";
      "extern unsigned long fread(void *, unsigned long, unsigned long, void \
       *);";
      "extern int printf(const char *, ...);";
      "extern int memcmp(const void *, const void *, unsigned long);";
      "int main(int argc, char **argv) {";
      "  (void)argc;";
      "  void *in = fopen(argv[1], \"rb\");";
      "  long n = 0, bad = 0;";
      "  if (argv[2][0] == 'h') {";
      "    for (uint32_t h = 0; h < 65536; h++) {";
      "      float a = f16_to_f32((uint16_t)h), b = \
       (float)f16_to_float((uint16_t)h);";
      "      float c = bf16_to_f32((uint16_t)h), d = \
       (float)bf16_to_float((uint16_t)h);";
      "      n++;";
      "      if (memcmp(&a, &b, sizeof a) != 0 && !(a != a && b != b)) bad++;";
      "      if (memcmp(&c, &d, sizeof c) != 0 && !(c != c && d != d)) bad++;";
      "    }";
      "    printf(\"%ld f16 and bf16 patterns, %ld differ\\n\", n, bad);";
      "  } else if (argv[2][0] == 'i') {";
      "    int64_t v;";
      "    while (fread(&v, sizeof v, 1, in) == 1) {";
      "      float a = f32_from_i64(v), b = (float)v;";
      "      n++;";
      "      if (memcmp(&a, &b, sizeof a) != 0) {";
      "        if (bad++ < 5) printf(\"%lld: %a, host %a\\n\", (long long)v, \
       (double)a, (double)b);";
      "      }";
      "    }";
      "    printf(\"%ld int64 inputs, %ld differ\\n\", n, bad);";
      "  } else {";
      "    float x;";
      "    while (fread(&x, sizeof x, 1, in) == 1) {";
      "      struct model_error a, b;";
      "      memset(&a, 0, sizeof a); memset(&b, 0, sizeof b);";
      "      int64_t p = i64_from_f32(x), q = i64_from_float((double)x);";
      "      int ra = i64_from_f32_failure(&a, x);";
      "      int rb = i64_from_float_failure(&b, (double)x);";
      "      n++;";
      "      if (p != q || ra != rb || memcmp(&a, &b, sizeof a) != 0) {";
      "        if (bad++ < 5) printf(\"%a: %lld, binary64 %lld\\n\", \
       (double)x, (long long)p, (long long)q);";
      "      }";
      "    }";
      "    printf(\"%ld float inputs, %ld differ\\n\", n, bad);";
      "  }";
      "  return 0;";
      "}";
      "";
    ]

let rng = ref 0x2545F4914F6CDD1DL

let next () =
  (rng := Int64.(logxor !rng (shift_left !rng 13)));
  (rng := Int64.(logxor !rng (shift_right_logical !rng 7)));
  (rng := Int64.(logxor !rng (shift_left !rng 17)));
  !rng

let two k = Int64.shift_left 1L k

(* Each power of two from 2^23 up, with its neighbours, the halfway points of a
   binary32 at that magnitude and their neighbours, both signs; then random
   values of every bit length. *)
let int64s =
  let around v = [ Int64.pred v; v; Int64.succ v ] in
  let at k =
    let p = two k in
    let half = if k >= 24 then two (k - 24) else 0L in
    around p
    @ around (Int64.add p half)
    @ around (Int64.add p (Int64.mul 3L half))
  in
  let fixed =
    [ 0L; 1L; -1L; Int64.max_int; Int64.min_int; Int64.succ Int64.min_int ]
    @ List.concat_map at (List.init 40 (fun i -> 23 + i))
  in
  let random =
    List.init 200_000 (fun _ ->
        let bits = 1 + Int64.to_int (Int64.unsigned_rem (next ()) 63L) in
        Int64.shift_right (next ()) (64 - bits))
  in
  List.concat_map (fun v -> [ v; Int64.neg v ]) (fixed @ random)

(* Every float around the conversion's limits (±2^31, ±2^63) and both
   infinities and NaN; then random bit patterns and random values of
   magnitude 2^20 to 2^70. *)
let floats =
  let bits f = Int32.bits_of_float f in
  let around f =
    let b = bits f in
    List.map Int32.float_of_bits [ Int32.pred b; b; Int32.succ b ]
  in
  let fixed =
    [ 0.; -0.; 0.5; -0.5; 1.; -1.; infinity; neg_infinity; nan ]
    @ List.concat_map around
        [
          2147483648.; -2147483648.; 9223372036854775808.; -9223372036854775808.;
        ]
  in
  let random =
    List.init 200_000 (fun _ -> Int32.float_of_bits (Int64.to_int32 (next ())))
  in
  let ranged =
    List.init 200_000 (fun _ ->
        let e = 20 + Int64.to_int (Int64.unsigned_rem (next ()) 51L) in
        let m = Int64.to_float (Int64.logand (next ()) 0xFFFFFFL) in
        let s = if Int64.logand (next ()) 1L = 0L then 1. else -1. in
        Int32.float_of_bits (bits (s *. Float.ldexp (1. +. (m /. 16777216.)) e)))
  in
  fixed @ random @ ranged

let run mode write =
  let dir = Loop_c_exec.Proc.temp_dir "binary32_only" in
  Fun.protect ~finally:(fun () -> Loop_c_exec.Proc.remove_tree dir) @@ fun () ->
  let exe =
    match Loop_c_exec.compile c_source with
    | Ok e -> e
    | Error (`C_compile m | `C_host m) -> failwith m
  in
  let input = Filename.concat dir "in.bin" in
  let buf = Buffer.create 4096 in
  write buf;
  Loop_c_exec.Proc.write_file input (Buffer.contents buf);
  match Loop_c_exec.Proc.run [ exe; input; mode ] with
  | Ok (Loop_c_exec.Proc.Exited 0, out) -> print_string out
  | _ -> print_endline "the test program failed"

(* Every pattern: the binary32 decoders give the bits the binary64 ones round
   to (any NaN equal to any NaN). *)
let%expect_test "the f16 and bf16 decoders without binary64" =
  run "h" ignore;
  [%expect {| 65536 f16 and bf16 patterns, 0 differ |}]

let%expect_test "f32_from_i64 rounds as the host's conversion does" =
  run "i" (fun buf -> List.iter (Buffer.add_int64_le buf) int64s);
  [%expect {| 400732 int64 inputs, 0 differ |}]

let%expect_test "i64_from_f32 and its failure record equal the binary64 helpers"
    =
  run "f" (fun buf ->
      List.iter
        (fun f -> Buffer.add_int32_le buf (Int32.bits_of_float f))
        floats);
  [%expect {| 400021 float inputs, 0 differ |}]
