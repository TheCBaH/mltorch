(* Seeded contractions as generated RVV functions under qemu: C's own 32-bit
   words start the accumulators of a bfloat16 or an integer product, bitwise
   equal to [Contraction.eval] (the reference's rounding order for a bfloat16
   product, exact for an integer one); then the seed zeroed, which the runs
   must catch. *)

module H = Machine_rivet_riscv32.Rv32_harness
module K = Coralnpu_kernels.Rvv_contraction
module C = Contraction

let entry = "seed_entry"

let caller d =
  let module A = Machine_rivet_riscv32.Rv32_asm in
  A.module_ "seed_caller" (fun () ->
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

let hash seed e = ((seed * 7919) + (e * 104729)) * 2654435761 land 0x3fffffff

(* a bfloat16 near one, any sign *)
let bf16 seed e =
  let h = hash seed e in
  ((h land 1) lsl 15)
  lor ((120 + ((h lsr 8) mod 16)) lsl 7)
  lor ((h lsr 1) land 0x7F)

let bf16_bytes seed count =
  let b = Bytes.create (2 * count) in
  for e = 0 to count - 1 do
    Bytes.set_uint16_le b (2 * e) (bf16 seed e)
  done;
  Bytes.to_string b

let i8 sign seed e =
  let lo = match sign with C.Sign.Signed -> -128 | C.Sign.Unsigned -> 0 in
  if e = 0 then lo else if e = 1 then lo + 255 else lo + (hash seed e mod 256)

let i8_bytes sign seed count =
  String.init count (fun e -> Char.chr (i8 sign seed e land 0xFF))

let descriptor ~format ~numerics ~m ~n ~k =
  Result.get_ok
    (Err.payload
       (C.make ~batch:1L ~m ~n ~k
          ~a:{ C.Operand.batch = 0L; outer = k; k = 1L }
          ~b:{ C.Operand.batch = 0L; outer = 1L; k = n }
          ~c:{ C.Output.batch = 0L; row = n; col = 1L }
          ~format ~seed:C.Seed.Tensor ~numerics))

(* [Ok true] when C is bitwise the reference's *)
let run ?mutation d =
  let m = Int64.to_int d.C.m and n = Int64.to_int d.C.n in
  let k = Int64.to_int d.C.k in
  let a, b, size, read_a, read_b, seed, word, to_word =
    match d.C.format with
    | C.Element.I8 { a = sa; b = sb } ->
        let read sign s off =
          let v = Char.code s.[Int64.to_int off] in
          float_of_int
            (match sign with
            | C.Sign.Signed when v >= 128 -> v - 256
            | C.Sign.Signed | C.Sign.Unsigned -> v)
        in
        let a = i8_bytes sa 1 (m * k) and b = i8_bytes sb 2 (n * k) in
        let seed = Bytes.create (4 * m * n) in
        for e = 0 to (m * n) - 1 do
          Bytes.set_int32_le seed (4 * e)
            (Int32.of_int ((hash 3 e mod 20001) - 10000))
        done;
        ( a,
          b,
          1,
          read sa a,
          read sb b,
          Bytes.to_string seed,
          (fun s off -> Int32.to_float (String.get_int32_le s (4 * off))),
          fun x -> Int32.of_float x )
    | C.Element.Bf16 | C.Element.F32 ->
        let read s off =
          Int32.float_of_bits
            (Int32.shift_left
               (Int32.of_int (String.get_uint16_le s (2 * Int64.to_int off)))
               16)
        in
        let a = bf16_bytes 1 (m * k) and b = bf16_bytes 2 (n * k) in
        let seed = Bytes.create (4 * m * n) in
        for e = 0 to (m * n) - 1 do
          Bytes.set_int32_le seed (4 * e)
            (Int32.bits_of_float
               (Int32.float_of_bits
                  (Int32.of_int
                     ((0x3F800000 + (hash 3 e mod 0x400000))
                     lor ((hash 4 e land 1) lsl 31)))))
        done;
        ( a,
          b,
          2,
          read a,
          read b,
          Bytes.to_string seed,
          (fun s off -> Int32.float_of_bits (String.get_int32_le s (4 * off))),
          fun x -> Int32.bits_of_float x )
  in
  let expected = Bytes.make (4 * m * n) '\000' in
  C.eval d ~a:read_a ~b:read_b
    ~c:(fun off -> word seed (Int64.to_int off))
    ~write:(fun off x ->
      Bytes.set_int32_le expected (4 * Int64.to_int off) (to_word x));
  let ( let* ) = Result.bind in
  let* helper =
    Result.map_error Coralnpu_kernels.Kernel_refusal.to_string
      (K.module_ ?mutation d)
  in
  let* call = caller d in
  let* prepared =
    H.prepare ~profile:Machine_rivet_riscv32.Rv32_profile.Rv32imf_zve32f_ilp32f
      ~entry ~kernel:[ call; helper ]
      [
        Machine_rivet_riscv32_test.Harness_slot.make
          ~size:(size * m * k)
          ~align:16 H.Placement.In;
        Machine_rivet_riscv32_test.Harness_slot.make
          ~size:(size * n * k)
          ~align:16 H.Placement.In;
        Machine_rivet_riscv32_test.Harness_slot.make
          ~size:(4 * m * n)
          ~align:16 H.Placement.Out;
      ]
  in
  let* o =
    H.run
      ~cpu:
        "rv32,d=false,zve32f=true,vlen=128,elen=32,rvv_ta_all_1s=true,rvv_ma_all_1s=true"
      prepared ~bound:(fun slot_ ->
        match (slot_ : Core.Table_slot.t :> int) with
        | 0 -> Some a
        | 1 -> Some b
        | _ -> Some seed)
  in
  Ok
    (List.assoc (Core.Table_slot.of_int 2) o.H.outputs
    = Bytes.to_string expected)

let show name = function
  | Ok true -> Fmt.pr "%s: bitwise@." name
  | Ok false -> Fmt.pr "%s: DIFFERS@." name
  | Error e -> Fmt.pr "%s: %s@." name e

let shapes = [ (9L, 32L, 64L); (7L, 29L, 63L) ]

let%expect_test "seeded bfloat16 and integer products under qemu" =
  List.iter
    (fun (m, n, k) ->
      List.iter
        (fun numerics ->
          let d = descriptor ~format:C.Element.Bf16 ~numerics ~m ~n ~k in
          show
            (Fmt.str "%Ldx%Ld k%Ld bf16 %s" m n k (C.Numerics.name numerics))
            (run d))
        C.Numerics.[ Fused_sequential; Ordered ];
      List.iter
        (fun signs ->
          let d =
            descriptor
              ~format:(C.Element.I8 { a = fst signs; b = snd signs })
              ~numerics:C.Numerics.Ordered ~m ~n ~k
          in
          show
            (Fmt.str "%Ldx%Ld k%Ld %s" m n k (C.Element.name d.C.format))
            (run d))
        C.Sign.
          [
            (Signed, Signed);
            (Signed, Unsigned);
            (Unsigned, Signed);
            (Unsigned, Unsigned);
          ])
    shapes;
  [%expect
    {|
    9x32 k64 bf16 fused_sequential: bitwise
    9x32 k64 bf16 ordered: bitwise
    9x32 k64 i8ss: bitwise
    9x32 k64 i8su: bitwise
    9x32 k64 i8us: bitwise
    9x32 k64 i8uu: bitwise
    7x29 k63 bf16 fused_sequential: bitwise
    7x29 k63 bf16 ordered: bitwise
    7x29 k63 i8ss: bitwise
    7x29 k63 i8su: bitwise
    7x29 k63 i8us: bitwise
    7x29 k63 i8uu: bitwise |}]

let%expect_test "a zeroed seed is red under qemu" =
  show "bf16 ordered"
    (run ~mutation:K.Mutation.Seed_zeroed
       (descriptor ~format:C.Element.Bf16 ~numerics:C.Numerics.Ordered ~m:7L
          ~n:29L ~k:63L));
  show "i8ss"
    (run ~mutation:K.Mutation.Seed_zeroed
       (descriptor
          ~format:C.Element.(I8 { a = C.Sign.Signed; b = C.Sign.Signed })
          ~numerics:C.Numerics.Ordered ~m:7L ~n:29L ~k:63L));
  [%expect {|
    bf16 ordered: DIFFERS
    i8ss: DIFFERS |}]
