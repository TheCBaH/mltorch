(* Integer contractions on the Coral RTL, called directly: the RVV function
   on rvv-core-mini-axi and the matrix engine's on vme-core-mini-axi, every
   pairing of signed and unsigned operands, C's 32-bit integers against the
   descriptor's exact reference, with the cycles each takes. *)

module A = Machine_rivet_riscv32.Rv32_asm
module C = Contraction
module L = Coral_layout

let dir = Sys.getenv "CORALNPU_DATA"
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

let value sign seed e =
  let lo = match sign with C.Sign.Signed -> -128 | C.Sign.Unsigned -> 0 in
  if e = 0 then lo
  else if e = 1 then lo + 255
  else
    let h = ((seed * 7919) + (e * 104729)) * 2654435761 land 0x3fffffff in
    lo + (h mod 256)

let bytes_of sign seed count =
  String.init count (fun e -> Char.chr (value sign seed e land 0xFF))

let descriptor (sa, sb) ~m ~n ~k =
  Result.get_ok
    (Err.payload
       (C.make ~batch:1L ~m ~n ~k
          ~a:{ C.Operand.batch = 0L; outer = k; k = 1L }
          ~b:{ C.Operand.batch = 0L; outer = 1L; k = n }
          ~c:{ C.Output.batch = 0L; row = n; col = 1L }
          ~format:(C.Element.I8 { a = sa; b = sb })
          ~seed:C.Seed.Zero ~numerics:C.Numerics.Ordered))

let run ~configuration ~engine d =
  let m = Int64.to_int d.C.m and n = Int64.to_int d.C.n in
  let k = Int64.to_int d.C.k in
  let sa, sb =
    match d.C.format with
    | C.Element.I8 { a; b } -> (a, b)
    | C.Element.Bf16 | C.Element.F32 -> assert false
  in
  let a = bytes_of sa 1 (m * k) and b = bytes_of sb 2 (n * k) in
  let read sign s off =
    let v = Char.code s.[Int64.to_int off] in
    float_of_int
      (match sign with
      | C.Sign.Signed when v >= 128 -> v - 256
      | C.Sign.Signed | C.Sign.Unsigned -> v)
  in
  let expected = Bytes.make (4 * m * n) '\000' in
  C.eval d ~a:(read sa a) ~b:(read sb b)
    ~c:(fun _ -> 0.)
    ~write:(fun off x ->
      Bytes.set_int32_le expected (4 * Int64.to_int off) (Int32.of_float x));
  let ( let* ) = Result.bind in
  let* profile =
    Result.map_error
      (Fmt.str "%a" Coral_profile.pp_error)
      (Err.payload (Coral_profile.load ~dir configuration))
  in
  let* helper =
    if engine then Coralnpu_kernels_zvt.Zvt_contraction.module_ d
    else Coralnpu_kernels.Rvv_contraction.module_ d
  in
  let* call = caller d in
  let slot placement size =
    {
      L.Slot.align = 16L;
      home = L.Home.Any;
      placement;
      size = Int64.of_int size;
    }
  in
  let* image =
    Result.map_error
      (Fmt.str "%a" Coral_image.pp_error)
      (Err.payload
         (Coral_image.build
            ~rv32_profile:
              Machine_rivet_riscv32.Rv32_profile.Rv32imf_zve32f_ilp32f
            ~matrix:engine profile ~entry ~kernel:[ call; helper ]
            [
              slot L.Placement.In (m * k);
              slot L.Placement.In (n * k);
              slot L.Placement.Out (4 * m * n);
            ]))
  in
  let* o =
    Result.map_error
      (Fmt.str "%a" Coral_run.Run_error.pp)
      (Err.payload
         (Coral_run.run ~flags:[ "--cycles=50000000" ] ~dir image
            ~bound:(function
           | 0 -> Some a
           | 1 -> Some b
           | _ -> None)))
  in
  Ok
    ( List.assoc 2 o.Coral_run.outputs = Bytes.to_string expected,
      o.Coral_run.mcycle )

let%expect_test "integer contractions on the vector unit and the matrix engine"
    =
  List.iter
    (fun (m, n, k) ->
      List.iter
        (fun signs ->
          let d = descriptor signs ~m ~n ~k in
          List.iter
            (fun (unit, configuration, engine) ->
              match run ~configuration ~engine d with
              | Ok (exact, cycles) ->
                  Fmt.pr "%Ldx%Ld k%Ld %s, %s: %s, %Ld cycles, %.2f ops/cycle@."
                    m n k
                    (C.Element.name d.C.format)
                    unit
                    (if exact then "exact" else "DIFFERS")
                    cycles
                    (2. *. Int64.to_float (C.macs d) /. Int64.to_float cycles)
              | Error e ->
                  Fmt.pr "%Ldx%Ld k%Ld %s, %s: %s@." m n k
                    (C.Element.name d.C.format)
                    unit e)
            [
              ("vector unit", "rvv-core-mini-axi", false);
              ("matrix engine", "vme-core-mini-axi", true);
            ])
        C.Sign.
          [
            (Signed, Signed);
            (Signed, Unsigned);
            (Unsigned, Signed);
            (Unsigned, Unsigned);
          ])
    [ (32L, 32L, 64L); (19L, 29L, 61L) ];
  [%expect
    {|
    32x32 k64 i8ss, vector unit: exact, 12855 cycles, 10.20 ops/cycle
    32x32 k64 i8ss, matrix engine: exact, 5907 cycles, 22.19 ops/cycle
    32x32 k64 i8su, vector unit: exact, 12855 cycles, 10.20 ops/cycle
    32x32 k64 i8su, matrix engine: exact, 5907 cycles, 22.19 ops/cycle
    32x32 k64 i8us, vector unit: exact, 12855 cycles, 10.20 ops/cycle
    32x32 k64 i8us, matrix engine: exact, 5907 cycles, 22.19 ops/cycle
    32x32 k64 i8uu, vector unit: exact, 12855 cycles, 10.20 ops/cycle
    32x32 k64 i8uu, matrix engine: exact, 5907 cycles, 22.19 ops/cycle
    19x29 k61 i8ss, vector unit: exact, 7665 cycles, 8.77 ops/cycle
    19x29 k61 i8ss, matrix engine: exact, 4151 cycles, 16.19 ops/cycle
    19x29 k61 i8su, vector unit: exact, 7665 cycles, 8.77 ops/cycle
    19x29 k61 i8su, matrix engine: exact, 4151 cycles, 16.19 ops/cycle
    19x29 k61 i8us, vector unit: exact, 7665 cycles, 8.77 ops/cycle
    19x29 k61 i8us, matrix engine: exact, 4151 cycles, 16.19 ops/cycle
    19x29 k61 i8uu, vector unit: exact, 7665 cycles, 8.77 ops/cycle
    19x29 k61 i8uu, matrix engine: exact, 4151 cycles, 16.19 ops/cycle |}]
