(* Seeded bfloat16 and integer products on the Coral RTL, called directly: C's
   own 32-bit words start the accumulators, the matrix engine's function
   (Zvt, on vme-core-mini-axi) and the vector unit's (RVV, on
   rvv-core-mini-axi) bitwise against [Contraction.eval], then each with its
   seed zeroed, which the runs must catch. *)

module A = Machine_rivet_riscv32.Rv32_asm
module C = Contraction
module L = Coral_layout
module Z = Coralnpu_kernels_zvt.Zvt_contraction
module R = Coralnpu_kernels.Rvv_contraction

let dir = Sys.getenv "CORALNPU_DATA"
let entry = "seed_entry"

let caller d =
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

(* [engine]: the matrix engine's function, else the vector unit's; [zeroed]
   injects the seed defect *)
let run ~engine ~zeroed d =
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
  let configuration =
    if engine then "vme-core-mini-axi" else "rvv-core-mini-axi"
  in
  let ( let* ) = Result.bind in
  let* profile =
    Result.map_error
      (Fmt.str "%a" Coral_profile.pp_error)
      (Err.payload (Coral_profile.load ~dir configuration))
  in
  let* helper =
    Result.map_error Coralnpu_kernels.Kernel_refusal.to_string
    @@
    if engine then
      Coralnpu_kernels_zvt.Zvt_contraction.module_
        ?mutation:(if zeroed then Some Z.Mutation.Seed_zeroed else None)
        d
    else
      R.module_
        ?mutation:(if zeroed then Some R.Mutation.Seed_zeroed else None)
        d
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
              slot L.Placement.In (size * m * k);
              slot L.Placement.In (size * n * k);
              slot L.Placement.In_out (4 * m * n);
            ]))
  in
  let* o =
    Result.map_error
      (Fmt.str "%a" Coral_run.Run_error.pp)
      (Err.payload
         (Coral_run.run ~deadline:Deadline.seconds
            ~flags:[ "--cycles=50000000" ] ~dir image ~bound:(fun slot_ ->
              match (slot_ : Core.Table_slot.t :> int) with
              | 0 -> Some a
              | 1 -> Some b
              | _ -> Some seed)))
  in
  Ok
    (List.assoc (Core.Table_slot.of_int 2) o.Coral_run.outputs
    = Bytes.to_string expected)

let show name = function
  | Ok true -> Fmt.pr "%s: bitwise@." name
  | Ok false -> Fmt.pr "%s: DIFFERS@." name
  | Error e -> Fmt.pr "%s: %s@." name e

let signs =
  C.Sign.
    [
      (Signed, Signed);
      (Signed, Unsigned);
      (Unsigned, Signed);
      (Unsigned, Unsigned);
    ]

let%expect_test "seeded bfloat16 and integer products on the RTL" =
  List.iter
    (fun (m, n, k) ->
      List.iter
        (fun (unit, engine, numerics) ->
          let d = descriptor ~format:C.Element.Bf16 ~numerics ~m ~n ~k in
          show
            (Fmt.str "%Ldx%Ld k%Ld bf16 %s, %s" m n k (C.Numerics.name numerics)
               unit)
            (run ~engine ~zeroed:false d))
        C.Numerics.
          [ ("vector unit", false, Ordered); ("matrix engine", true, Paired) ];
      List.iter
        (fun (a, b) ->
          let d =
            descriptor
              ~format:(C.Element.I8 { a; b })
              ~numerics:C.Numerics.Ordered ~m ~n ~k
          in
          List.iter
            (fun (unit, engine) ->
              show
                (Fmt.str "%Ldx%Ld k%Ld %s, %s" m n k
                   (C.Element.name d.C.format)
                   unit)
                (run ~engine ~zeroed:false d))
            [ ("vector unit", false); ("matrix engine", true) ])
        signs)
    [ (32L, 32L, 64L); (19L, 29L, 61L) ];
  [%expect
    {|
    32x32 k64 bf16 ordered, vector unit: bitwise
    32x32 k64 bf16 paired, matrix engine: bitwise
    32x32 k64 i8ss, vector unit: bitwise
    32x32 k64 i8ss, matrix engine: bitwise
    32x32 k64 i8su, vector unit: bitwise
    32x32 k64 i8su, matrix engine: bitwise
    32x32 k64 i8us, vector unit: bitwise
    32x32 k64 i8us, matrix engine: bitwise
    32x32 k64 i8uu, vector unit: bitwise
    32x32 k64 i8uu, matrix engine: bitwise
    19x29 k61 bf16 ordered, vector unit: bitwise
    19x29 k61 bf16 paired, matrix engine: bitwise
    19x29 k61 i8ss, vector unit: bitwise
    19x29 k61 i8ss, matrix engine: bitwise
    19x29 k61 i8su, vector unit: bitwise
    19x29 k61 i8su, matrix engine: bitwise
    19x29 k61 i8us, vector unit: bitwise
    19x29 k61 i8us, matrix engine: bitwise
    19x29 k61 i8uu, vector unit: bitwise
    19x29 k61 i8uu, matrix engine: bitwise |}]

let%expect_test "a zeroed seed is red on the RTL" =
  List.iter
    (fun (unit, engine) ->
      show ("bf16, " ^ unit)
        (run ~engine ~zeroed:true
           (descriptor ~format:C.Element.Bf16
              ~numerics:
                (if engine then C.Numerics.Paired else C.Numerics.Ordered)
              ~m:19L ~n:29L ~k:61L));
      show ("i8ss, " ^ unit)
        (run ~engine ~zeroed:true
           (descriptor
              ~format:(C.Element.I8 { a = C.Sign.Signed; b = C.Sign.Signed })
              ~numerics:C.Numerics.Ordered ~m:19L ~n:29L ~k:61L)))
    [ ("vector unit", false); ("matrix engine", true) ];
  [%expect
    {|
    bf16, vector unit: DIFFERS
    i8ss, vector unit: DIFFERS
    bf16, matrix engine: DIFFERS
    i8ss, matrix engine: DIFFERS |}]
