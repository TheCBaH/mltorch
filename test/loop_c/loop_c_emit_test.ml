open Loop_ir
open Loop_fixtures
open Loop_programs

let kernel_of p =
  match Err.payload (Loop_c.kernel ~name:"kernel" p) with
  | Ok k -> k
  | Error _ -> failwith "refused"

let%expect_test "the doubling kernel's C, and that emitting twice is identical"
    =
  let a = kernel_of doubling and b = kernel_of doubling in
  print_string a.Loop_c.source;
  Fmt.pr "identical: %b; helpers: %d; local doubles: %Ld@."
    (String.equal a.Loop_c.source b.Loop_c.source)
    (List.length a.Loop_c.helpers)
    a.Loop_c.local_doubles;
  [%expect
    {|
    static int kernel(struct model_error *err, double *local, const float *b0, float *b1) {
      (void)err; (void)local;
      (void)b0;
      (void)b1;
      {
        for (int64_t i0 = ((int64_t)0); i0 < ((int64_t)4); i0++) {
          b1[i0] = (float)((double)b0[i0] * (0x1p+1));
        }
      }
      return 0;
    }
    identical: true; helpers: 0; local doubles: 0 |}]

let%expect_test "a quantized buffer is a typed refusal, not a bad kernel" =
  let q = buffer 0 (shape_w 4) (Payload.Fmt Payload.I8) Loop_buffer.Input in
  (match Err.payload (Loop_c.kernel ~name:"k" (program ~buffers:[ q ] [])) with
  | Ok _ -> print_endline "accepted"
  | Error e -> Fmt.pr "%a@." Loop_c.pp_error e);
  [%expect {| t0: format i8 has no C implementation |}]

(* Every half-precision pattern, decoded by the C helper and by the reference,
   compared by bits (a NaN by being a NaN: a payload is not preserved). *)
let%expect_test "f16 and bf16 decode agrees with Half for all 65536 patterns" =
  let text =
    String.concat "\n"
      [
        Loop_c_runtime.prelude;
        Loop_c_runtime.helpers
          [
            Loop_c_runtime.Name.F16_to_float; Loop_c_runtime.Name.Bf16_to_float;
          ];
        "#include <stdio.h>";
        "int main(void) {";
        "  for (uint32_t h = 0; h < 65536; h++) {";
        "    double a = f16_to_float((uint16_t)h), b = \
         bf16_to_float((uint16_t)h);";
        "    fwrite(&a, sizeof a, 1, stdout); fwrite(&b, sizeof b, 1, stdout);";
        "  }";
        "  return 0;";
        "}";
        "";
      ]
  in
  let exe =
    match Loop_c_exec.compile text with
    | Ok exe -> exe
    | Error (`C_compile m | `C_host m) -> failwith m
  in
  let out = Filename.temp_file "half" ".bin" in
  let rc = Sys.command (Filename.quote_command exe [] ~stdout:out) in
  let s = Loop_c_exec.Proc.read_file out in
  Sys.remove out;
  let bad = ref 0 in
  for h = 0 to 65535 do
    let get k =
      Int64.float_of_bits (String.get_int64_le s (((h * 2) + k) * 8))
    in
    let same x y =
      if Float.is_nan y then Float.is_nan x
      else Int64.equal (Int64.bits_of_float x) (Int64.bits_of_float y)
    in
    if not (same (get 0) (Half.Half.to_float h)) then incr bad;
    if not (same (get 1) (Half.Bf16.to_float h)) then incr bad
  done;
  Fmt.pr "exit %d, mismatches: %d@." rc !bad;
  [%expect {| exit 0, mismatches: 0 |}]
