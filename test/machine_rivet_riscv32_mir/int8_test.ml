(* Integer contractions as generated RVV functions, called directly under
   qemu: a caller hands the function A, B and C from the slot table, and C's
   32-bit integers are compared with the descriptor's exact reference, for
   every pairing of signed and unsigned operands, B's columns contiguous or
   strided, and shapes with row, column and k tails; then A's row stride
   mutated, which the runs must catch. No recognized loop nest is an integer
   product yet, so the functions are reached this way. *)

module A = Machine_rivet_riscv32.Rv32_asm
module H = Machine_rivet_riscv32.Rv32_harness
module K = Coralnpu_kernels.Rvv_contraction
module C = Contraction

let entry = "int8_entry"

let caller d =
  A.module_ "int8_caller" (fun () ->
      let i = A.insn and r = A.reg in
      [
        A.text;
        A.align 4;
        A.global entry;
        A.label entry;
        i "addi" [ r "sp"; r "sp"; A.imm (-16) ];
        i "sw" [ r "ra"; A.mem "sp" 12 ];
        i "mv" [ r "t0"; r "a0" ];
        i "lw" [ r "a0"; A.mem "t0" 0 ];
        i "lw" [ r "a1"; A.mem "t0" 4 ];
        i "lw" [ r "a2"; A.mem "t0" 8 ];
        i "call" [ A.sym (C.symbol d) ];
        i "lw" [ r "ra"; A.mem "sp" 12 ];
        i "addi" [ r "sp"; r "sp"; A.imm 16 ];
        i "li" [ r "a0"; A.imm 0 ];
        i "ret" [];
      ])

(* Deterministic 8-bit values, the range's ends first. *)
let value sign seed e =
  let lo, span =
    match sign with C.Sign.Signed -> (-128, 256) | C.Sign.Unsigned -> (0, 256)
  in
  if e = 0 then lo
  else if e = 1 then lo + span - 1
  else
    let h = ((seed * 7919) + (e * 104729)) * 2654435761 land 0x3fffffff in
    lo + (h mod span)

let bytes_of sign seed count =
  String.init count (fun e -> Char.chr (value sign seed e land 0xFF))

let descriptor ~signs:(sa, sb) ~strided ~m ~n ~k =
  Result.get_ok
    (Err.payload
       (C.make ~batch:1L ~m ~n ~k
          ~a:{ C.Operand.batch = 0L; outer = k; k = 1L }
          ~b:
            (if strided then { C.Operand.batch = 0L; outer = k; k = 1L }
             else { C.Operand.batch = 0L; outer = 1L; k = n })
          ~c:{ C.Output.batch = 0L; row = n; col = 1L }
          ~format:(C.Element.I8 { a = sa; b = sb })
          ~seed:C.Seed.Zero ~numerics:C.Numerics.Ordered))

let run ?mutation ~vlen d =
  let m = Int64.to_int d.C.m and n = Int64.to_int d.C.n in
  let k = Int64.to_int d.C.k in
  let sa, sb =
    match d.C.format with
    | C.Element.I8 { a; b } -> (a, b)
    | C.Element.Bf16 | C.Element.F32 -> assert false
  in
  let a = bytes_of sa 1 (m * k) and b = bytes_of sb 2 (n * k) in
  let expected = Bytes.make (4 * m * n) '\000' in
  let read sign s off =
    let v = Char.code s.[Int64.to_int off] in
    float_of_int
      (match sign with
      | C.Sign.Signed when v >= 128 -> v - 256
      | C.Sign.Signed | C.Sign.Unsigned -> v)
  in
  C.eval d ~a:(read sa a) ~b:(read sb b)
    ~c:(fun _ -> 0.)
    ~write:(fun off x ->
      Bytes.set_int32_le expected (4 * Int64.to_int off) (Int32.of_float x));
  let ( let* ) = Result.bind in
  let* helper = K.module_ ?mutation d in
  let* call = caller d in
  let* prepared =
    H.prepare ~profile:Machine_rivet_riscv32.Rv32_profile.Rv32imf_zve32f_ilp32f
      ~entry ~kernel:[ call; helper ]
      [
        { H.size = m * k; align = 16; placement = H.Placement.In };
        { H.size = n * k; align = 16; placement = H.Placement.In };
        { H.size = 4 * m * n; align = 16; placement = H.Placement.Out };
      ]
  in
  let* o =
    H.run
      ~cpu:
        (Fmt.str
           "rv32,d=false,zve32f=true,vlen=%d,elen=32,rvv_ta_all_1s=true,rvv_ma_all_1s=true"
           vlen) prepared ~bound:(function
      | 0 -> Some a
      | 1 -> Some b
      | _ -> None)
  in
  Ok (List.assoc 2 o.H.outputs = Bytes.to_string expected)

let%expect_test "integer contractions under qemu" =
  let show name = function
    | Ok true -> Fmt.pr "%s: exact@." name
    | Ok false -> Fmt.pr "%s: DIFFERS@." name
    | Error e -> Fmt.pr "%s: %s@." name e
  in
  List.iter
    (fun vlen ->
      List.iter
        (fun (m, n, k) ->
          List.iter
            (fun signs ->
              List.iter
                (fun strided ->
                  let d = descriptor ~signs ~strided ~m ~n ~k in
                  show
                    (Fmt.str "vlen %d, %Ldx%Ld k%Ld %s, B %s" vlen m n k
                       (C.Element.name d.C.format)
                       (if strided then "strided" else "contiguous"))
                    (run ~vlen d))
                [ false; true ])
            C.Sign.
              [
                (Signed, Signed);
                (Signed, Unsigned);
                (Unsigned, Signed);
                (Unsigned, Unsigned);
              ])
        [ (9L, 32L, 64L); (7L, 29L, 63L) ])
    [ 128; 1024 ];
  show "A row stride"
    (run ~mutation:K.Mutation.A_row_stride ~vlen:128
       (descriptor
          ~signs:C.Sign.(Signed, Signed)
          ~strided:true ~m:7L ~n:29L ~k:63L));
  [%expect
    {|
    vlen 128, 9x32 k64 i8ss, B contiguous: exact
    vlen 128, 9x32 k64 i8ss, B strided: exact
    vlen 128, 9x32 k64 i8su, B contiguous: exact
    vlen 128, 9x32 k64 i8su, B strided: exact
    vlen 128, 9x32 k64 i8us, B contiguous: exact
    vlen 128, 9x32 k64 i8us, B strided: exact
    vlen 128, 9x32 k64 i8uu, B contiguous: exact
    vlen 128, 9x32 k64 i8uu, B strided: exact
    vlen 128, 7x29 k63 i8ss, B contiguous: exact
    vlen 128, 7x29 k63 i8ss, B strided: exact
    vlen 128, 7x29 k63 i8su, B contiguous: exact
    vlen 128, 7x29 k63 i8su, B strided: exact
    vlen 128, 7x29 k63 i8us, B contiguous: exact
    vlen 128, 7x29 k63 i8us, B strided: exact
    vlen 128, 7x29 k63 i8uu, B contiguous: exact
    vlen 128, 7x29 k63 i8uu, B strided: exact
    vlen 1024, 9x32 k64 i8ss, B contiguous: exact
    vlen 1024, 9x32 k64 i8ss, B strided: exact
    vlen 1024, 9x32 k64 i8su, B contiguous: exact
    vlen 1024, 9x32 k64 i8su, B strided: exact
    vlen 1024, 9x32 k64 i8us, B contiguous: exact
    vlen 1024, 9x32 k64 i8us, B strided: exact
    vlen 1024, 9x32 k64 i8uu, B contiguous: exact
    vlen 1024, 9x32 k64 i8uu, B strided: exact
    vlen 1024, 7x29 k63 i8ss, B contiguous: exact
    vlen 1024, 7x29 k63 i8ss, B strided: exact
    vlen 1024, 7x29 k63 i8su, B contiguous: exact
    vlen 1024, 7x29 k63 i8su, B strided: exact
    vlen 1024, 7x29 k63 i8us, B contiguous: exact
    vlen 1024, 7x29 k63 i8us, B strided: exact
    vlen 1024, 7x29 k63 i8uu, B contiguous: exact
    vlen 1024, 7x29 k63 i8uu, B strided: exact
    A row stride: DIFFERS |}]
