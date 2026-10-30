(* The matrix engine's bfloat16 functions, called directly on the VME RTL: C's
   binary32 results bitwise against the descriptor's [Paired] reference (each
   two k steps summed and rounded to odd, then added to the accumulator to
   nearest), over operands whose exponents span enough to make that differ from
   rounding every step, with the cycles each takes. The plain function loads
   its slices straight from the operands; the packed one packs them into
   panels of its own for the suite's double-buffered blocks, or reads an
   operand in place where its k rows are contiguous. *)

module A = Machine_rivet_riscv32.Rv32_asm
module C = Contraction
module L = Coral_layout
module Z = Coralnpu_kernels_zvt

let dir = Sys.getenv "CORALNPU_DATA"
let configuration = "vme-core-mini-axi"
let entry = "bf16_entry"

let caller d =
  A.module_ "bf16_caller" (fun () ->
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

(* SplitMix64's finalizer: every bit of a seed and an index reaches every bit
   of the result *)
let mix seed e =
  let open Int64 in
  let z =
    add (shift_left (of_int seed) 32) (mul (of_int (e + 1)) 0x9E3779B97F4A7C15L)
  in
  let z = mul (logxor z (shift_right_logical z 30)) 0xBF58476D1CE4E5B9L in
  let z = mul (logxor z (shift_right_logical z 27)) 0x94D049BB133111EBL in
  to_int (shift_right_logical (logxor z (shift_right_logical z 31)) 8)

(* a bfloat16 of random sign, mantissa and an exponent within [spread] of 1:
   products and sums then overlap enough for the rounding to matter *)
let element seed e ~spread =
  let h = mix seed e in
  let sign = h land 1 and man = (h lsr 1) land 0x7F in
  let ex = 127 - spread + ((h lsr 8) mod ((2 * spread) + 1)) in
  (sign lsl 15) lor (ex lsl 7) lor man

let bytes_of seed count ~spread =
  let b = Bytes.create (2 * count) in
  for e = 0 to count - 1 do
    Bytes.set_uint16_le b (2 * e) (element seed e ~spread)
  done;
  Bytes.to_string b

(* A as [m, k] rows, or transposed ([k, m] rows, a row's indices contiguous);
   B as [k, n] rows *)
let descriptor ?(numerics = C.Numerics.Paired) ?(a_transposed = false)
    ?(c_transposed = false) ?(seeded = false) ~m ~n ~k () =
  Result.get_ok
    (Err.payload
       (C.make ~batch:1L ~m ~n ~k
          ~a:
            (if a_transposed then { C.Operand.batch = 0L; outer = 1L; k = m }
             else { C.Operand.batch = 0L; outer = k; k = 1L })
          ~b:{ C.Operand.batch = 0L; outer = 1L; k = n }
          ~c:
            (if c_transposed then { C.Output.batch = 0L; row = 1L; col = m }
             else { C.Output.batch = 0L; row = n; col = 1L })
          ~format:C.Element.Bf16
          ~seed:(if seeded then C.Seed.Tensor else C.Seed.Zero)
          ~numerics))

let reference ?(seed = "") d a b =
  let m = Int64.to_int d.C.m and n = Int64.to_int d.C.n in
  let read s off =
    Int32.float_of_bits
      (Int32.shift_left
         (Int32.of_int (String.get_uint16_le s (2 * Int64.to_int off)))
         16)
  in
  let out = Bytes.make (4 * m * n) '\000' in
  let seed_at off =
    if seed = "" then 0.
    else Int32.float_of_bits (String.get_int32_le seed (4 * Int64.to_int off))
  in
  C.eval d ~a:(read a) ~b:(read b) ~c:seed_at ~write:(fun off x ->
      Bytes.set_int32_le out (4 * Int64.to_int off) (Int32.bits_of_float x));
  Bytes.to_string out

(* Which function, and the defect injected into it, if any *)
module Engine = struct
  type t =
    | Packed of Z.Zvt_packed_contraction.Mutation.t option
    | Plain of Z.Zvt_contraction.Mutation.t option
    | Outside
        (** the same, unless B already lies in the DTCM: checked at run time *)
    | Staged of Z.Zvt_packed_contraction.Mutation.t option
        (** the packed function copying a B read in place into DTCM *)

  let plain = Plain None
  let packed = Packed None
  let staged = Staged None
end

type result = {
  actual : string;  (** C's bytes *)
  wrong : int;  (** elements not bitwise equal to the reference *)
  nearest : int;  (** elements the reference gives differently to nearest *)
  cycles : int64;
}

let run ?(sram = false) ?(engine = Engine.plain) ?operands ?seed ~spread d =
  let m = Int64.to_int d.C.m and n = Int64.to_int d.C.n in
  let k = Int64.to_int d.C.k in
  let a, b =
    match operands with
    | Some ab -> ab
    | None -> (bytes_of 1 (m * k) ~spread, bytes_of 2 (n * k) ~spread)
  in
  let expected = reference ?seed d a b in
  (* an operand read in place is read a few k rows past its end *)
  let pad =
    match engine with
    | Engine.Packed _ | Staged _ | Outside -> 16 * max m n
    | Plain _ -> 0
  in
  let ordered =
    reference
      (Result.get_ok
         (Err.payload
            (C.make ~batch:d.C.batch ~m:d.C.m ~n:d.C.n ~k:d.C.k ~a:d.C.a
               ~b:d.C.b ~c:d.C.c ~format:d.C.format ~seed:d.C.seed
               ~numerics:C.Numerics.Ordered)))
      ?seed a b
  in
  let differing xs ys =
    List.length
      (List.filter
         (fun e -> String.sub xs (4 * e) 4 <> String.sub ys (4 * e) 4)
         (List.init (m * n) Fun.id))
  in
  let ( let* ) = Result.bind in
  let* profile =
    Result.map_error
      (Fmt.str "%a" Coral_profile.pp_error)
      (Err.payload (Coral_profile.load ~dir configuration))
  in
  let* helper =
    match engine with
    | Engine.Packed mutation -> Z.Zvt_packed_contraction.module_ ?mutation d
    | Plain mutation -> Z.Zvt_contraction.module_ ?mutation d
    | Outside ->
        let dtcm =
          List.find
            (fun (r : Coral_profile.Region.t) ->
              r.Coral_profile.Region.role = Coral_profile.Role.Dtcm)
            profile.Coral_profile.regions
        in
        Z.Zvt_packed_contraction.module_
          ~stage_b:
            (Z.Zvt_packed_contraction.Stage.Outside
               {
                 base = Int64.to_int dtcm.Coral_profile.Region.base;
                 bytes = Int64.to_int dtcm.Coral_profile.Region.size;
               })
          d
    | Staged mutation ->
        Z.Zvt_packed_contraction.module_ ?mutation
          ~stage_b:Z.Zvt_packed_contraction.Stage.Always d
  in
  let* call = caller d in
  let slot placement size =
    {
      L.Slot.align = 16L;
      home = (if sram then L.Home.Ram else L.Home.Any);
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
            ~matrix:true profile ~entry ~kernel:[ call; helper ]
            [
              slot L.Placement.In ((2 * m * k) + pad);
              slot L.Placement.In ((2 * n * k) + pad);
              slot
                (if seed = None then L.Placement.Out else L.Placement.In_out)
                (4 * m * n);
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
           | 2 -> seed
           | _ -> None)))
  in
  let actual = List.assoc 2 o.Coral_run.outputs in
  Ok
    {
      actual;
      wrong = differing actual expected;
      nearest = differing expected ordered;
      cycles = o.Coral_run.mcycle;
    }

let show ?sram ?engine ?seed ~what d =
  let m = d.C.m and n = d.C.n and k = d.C.k in
  match run ?sram ?engine ?seed ~spread:12 d with
  | Ok r ->
      Fmt.pr
        "%Ldx%Ld k%Ld %s: %s (%d of %Ld differ from every step rounded to \
         nearest), %Ld cycles, %.1f ops/cycle@."
        m n k what
        (if r.wrong = 0 then "bitwise" else Fmt.str "DIFFERS in %d" r.wrong)
        r.nearest (Int64.mul m n) r.cycles
        (2. *. Int64.to_float (C.macs d) /. Int64.to_float r.cycles)
  | Error e -> Fmt.pr "%Ldx%Ld k%Ld %s: %s@." m n k what e

let%expect_test "bfloat16 pairs on the matrix engine" =
  List.iter
    (fun (m, n, k) -> show ~what:"plain" (descriptor ~m ~n ~k ()))
    [ (32L, 32L, 64L); (19L, 29L, 61L); (16L, 16L, 1L) ];
  [%expect
    {|
    32x32 k64 plain: bitwise (734 of 1024 differ from every step rounded to nearest), 6169 cycles, 21.2 ops/cycle
    19x29 k61 plain: bitwise (398 of 551 differ from every step rounded to nearest), 4451 cycles, 15.1 ops/cycle
    16x16 k1 plain: bitwise (0 of 256 differ from every step rounded to nearest), 196 cycles, 2.6 ops/cycle |}]

(* The packed function over A row-major (packed by strided loads, the slow
   case) and transposed (read in place, with k a multiple of four), partial
   tiles and a 6-row tail included. *)
let%expect_test "the packed function" =
  List.iter
    (fun (m, n, k) ->
      show ~engine:Engine.packed ~what:"packed, A row-major"
        (descriptor ~m ~n ~k ()))
    [ (32L, 32L, 64L); (19L, 29L, 61L); (70L, 40L, 33L); (40L, 32L, 62L) ];
  List.iter
    (fun (m, n, k) ->
      show ~engine:Engine.packed ~what:"packed, A in place"
        (descriptor ~a_transposed:true ~m ~n ~k ()))
    [ (32L, 32L, 64L); (32L, 64L, 64L); (70L, 40L, 32L) ];
  (* C stored transposed: the product with its operands exchanged, which
     turns the next layer's A into a k-major operand *)
  List.iter
    (fun (m, n, k, a_transposed) ->
      show ~engine:Engine.packed
        ~what:
          (if a_transposed then "packed, C transposed, A in place"
           else "packed, C transposed, A row-major")
        (descriptor ~a_transposed ~c_transposed:true ~m ~n ~k ()))
    [
      (32L, 32L, 64L, true);
      (32L, 64L, 64L, true);
      (70L, 40L, 32L, true);
      (32L, 32L, 64L, false);
      (19L, 29L, 61L, true);
    ];
  [%expect
    {|
    32x32 k64 packed, A row-major: bitwise (734 of 1024 differ from every step rounded to nearest), 3008 cycles, 43.6 ops/cycle
    19x29 k61 packed, A row-major: bitwise (398 of 551 differ from every step rounded to nearest), 4686 cycles, 14.3 ops/cycle
    70x40 k33 packed, A row-major: bitwise (1853 of 2800 differ from every step rounded to nearest), 10376 cycles, 17.8 ops/cycle
    40x32 k62 packed, A row-major: bitwise (925 of 1280 differ from every step rounded to nearest), 5243 cycles, 30.3 ops/cycle
    32x32 k64 packed, A in place: bitwise (747 of 1024 differ from every step rounded to nearest), 1318 cycles, 99.4 ops/cycle
    32x64 k64 packed, A in place: bitwise (1481 of 2048 differ from every step rounded to nearest), 2514 cycles, 104.3 ops/cycle
    70x40 k32 packed, A in place: bitwise (1889 of 2800 differ from every step rounded to nearest), 3603 cycles, 49.7 ops/cycle
    32x32 k64 packed, C transposed, A in place: bitwise (747 of 1024 differ from every step rounded to nearest), 1318 cycles, 99.4 ops/cycle
    32x64 k64 packed, C transposed, A in place: bitwise (1481 of 2048 differ from every step rounded to nearest), 2514 cycles, 104.3 ops/cycle
    70x40 k32 packed, C transposed, A in place: bitwise (1889 of 2800 differ from every step rounded to nearest), 3674 cycles, 48.8 ops/cycle
    32x32 k64 packed, C transposed, A row-major: bitwise (734 of 1024 differ from every step rounded to nearest), 3009 cycles, 43.6 ops/cycle
    19x29 k61 packed, C transposed, A in place: bitwise (410 of 551 differ from every step rounded to nearest), 3933 cycles, 17.1 ops/cycle |}]

(* Each mutant is a defect a run must detect: a partial last column tile
   written whole, A's slices where B's go, A's rows an element too far apart
   (packed or read in place), the last k step not packed. *)
let%expect_test "the evidence catches each defect" =
  let rows = descriptor ~m:19L ~n:29L ~k:61L () in
  let in_place = descriptor ~a_transposed:true ~m:32L ~n:32L ~k:64L () in
  let verdict what d engine =
    match run ~engine ~spread:12 d with
    | Ok r ->
        Fmt.pr "%s: %s@." what (if r.wrong = 0 then "bitwise" else "DIFFERS")
    | Error e -> Fmt.pr "%s: %s@." what e
  in
  verdict "a full last tile" rows
    (Engine.Plain (Some Z.Zvt_contraction.Mutation.Full_tn));
  verdict "swapped operands" rows
    (Engine.Plain (Some Z.Zvt_contraction.Mutation.Swapped_operands));
  verdict "packed, A's pitch" rows
    (Engine.Packed (Some Z.Zvt_packed_contraction.Mutation.A_pitch));
  verdict "packed, a short k" rows
    (Engine.Packed (Some Z.Zvt_packed_contraction.Mutation.Short_k));
  verdict "in place, A's pitch" in_place
    (Engine.Packed (Some Z.Zvt_packed_contraction.Mutation.A_pitch));
  verdict "staged B, a short copy" in_place
    (Engine.Staged (Some Z.Zvt_packed_contraction.Mutation.Short_copy));
  [%expect
    {|
    a full last tile: DIFFERS
    swapped operands: DIFFERS
    packed, A's pitch: DIFFERS
    packed, a short k: DIFFERS
    in place, A's pitch: DIFFERS
    staged B, a short copy: DIFFERS |}]

(* Operands in SRAM: the core reads them four bytes a cycle and the DMA moves
   a byte and a seventh, so a B reused by every block of A is worth copying
   into DTCM, once, by the core. *)
let%expect_test "B staged into DTCM" =
  List.iter
    (fun (m, n, k) ->
      let d = descriptor ~a_transposed:true ~m ~n ~k () in
      show ~sram:true ~engine:Engine.packed ~what:"in SRAM, in place" d;
      show ~sram:true ~engine:(Engine.Staged None) ~what:"in SRAM, B staged" d)
    [ (32L, 32L, 64L); (192L, 32L, 128L); (192L, 64L, 128L) ];
  [%expect
    {|
    32x32 k64 in SRAM, in place: bitwise (747 of 1024 differ from every step rounded to nearest), 3917 cycles, 33.5 ops/cycle
    32x32 k64 in SRAM, B staged: bitwise (747 of 1024 differ from every step rounded to nearest), 4567 cycles, 28.7 ops/cycle
    192x32 k128 in SRAM, in place: bitwise (4905 of 6144 differ from every step rounded to nearest), 36716 cycles, 42.8 ops/cycle
    192x32 k128 in SRAM, B staged: bitwise (4905 of 6144 differ from every step rounded to nearest), 30273 cycles, 52.0 ops/cycle
    192x64 k128 in SRAM, in place: bitwise (9757 of 12288 differ from every step rounded to nearest), 73310 cycles, 42.9 ops/cycle
    192x64 k128 in SRAM, B staged: bitwise (9757 of 12288 differ from every step rounded to nearest), 60429 cycles, 52.1 ops/cycle |}]

(* Staged only where B is outside the DTCM: in SRAM it is copied (the rate of
   the staged run), in the DTCM it is read where it is (the rate of the plain
   one), both bitwise. *)
let%expect_test "B staged where it is outside the DTCM" =
  let d = descriptor ~a_transposed:true ~m:96L ~n:32L ~k:64L () in
  show ~sram:true ~engine:Engine.packed ~what:"in SRAM, in place" d;
  show ~sram:true ~engine:Engine.Outside ~what:"in SRAM, staged outside" d;
  show ~engine:Engine.packed ~what:"in DTCM, in place" d;
  show ~engine:Engine.Outside ~what:"in DTCM, staged outside" d;
  [%expect
    {|
    96x32 k64 in SRAM, in place: bitwise (2261 of 3072 differ from every step rounded to nearest), 11507 cycles, 34.2 ops/cycle
    96x32 k64 in SRAM, staged outside: bitwise (2261 of 3072 differ from every step rounded to nearest), 10588 cycles, 37.1 ops/cycle
    96x32 k64 in DTCM, in place: bitwise (2261 of 3072 differ from every step rounded to nearest), 6776 cycles, 58.0 ops/cycle
    96x32 k64 in DTCM, staged outside: bitwise (2261 of 3072 differ from every step rounded to nearest), 6780 cycles, 58.0 ops/cycle |}]

(* binary32 words of random sign, mantissa and an exponent within [spread] of 1 *)
let f32_bytes seed count ~spread =
  let b = Bytes.create (4 * count) in
  for e = 0 to count - 1 do
    let h = mix seed e in
    let sign = h land 1 and man = (h lsr 1) land 0x7FFFFF in
    let ex = 127 - spread + ((h lsr 24) mod ((2 * spread) + 1)) in
    Bytes.set_int32_le b (4 * e)
      (Int32.of_int ((sign lsl 31) lor (ex lsl 23) lor man))
  done;
  Bytes.to_string b

let f32_word x =
  let b = Bytes.create 4 in
  Bytes.set_int32_le b 0 (Int32.bits_of_float x);
  Bytes.to_string b

(* C is the seed: each block's accumulators start from C's tiles and the sums
   continue from there, bitwise against the descriptor's [Paired] reference
   from a random seed. A k the depth padding changes is refused. *)
let%expect_test "the packed function seeded from C" =
  let seeded ?a_transposed ?c_transposed ~m ~n ~k () =
    let d = descriptor ?a_transposed ?c_transposed ~seeded:true ~m ~n ~k () in
    let seed = f32_bytes 3 (Int64.to_int (Int64.mul d.C.m d.C.n)) ~spread:12 in
    (d, seed)
  in
  List.iter
    (fun (what, a_transposed, c_transposed, (m, n, k)) ->
      let d, seed = seeded ~a_transposed ~c_transposed ~m ~n ~k () in
      show ~engine:Engine.packed ~seed ~what d)
    [
      ("seeded, A row-major", false, false, (32L, 32L, 64L));
      ("seeded, the context shape, A in place", true, false, (16L, 64L, 32L));
      ("seeded, partial tiles", false, false, (19L, 29L, 60L));
      ("seeded, a three-step tail", false, false, (70L, 40L, 63L));
      ("seeded, C transposed, A in place", true, true, (32L, 64L, 64L));
      ("seeded, k = 61", false, false, (19L, 29L, 61L));
      ("seeded, k = 62", false, false, (19L, 29L, 62L));
    ];
  [%expect
    {|
    32x32 k64 seeded, A row-major: bitwise (742 of 1024 differ from every step rounded to nearest), 3393 cycles, 38.6 ops/cycle
    16x64 k32 seeded, the context shape, A in place: bitwise (677 of 1024 differ from every step rounded to nearest), 1599 cycles, 41.0 ops/cycle
    19x29 k60 seeded, partial tiles: bitwise (411 of 551 differ from every step rounded to nearest), 3496 cycles, 18.9 ops/cycle
    70x40 k63 seeded, a three-step tail: bitwise (2006 of 2800 differ from every step rounded to nearest), 19845 cycles, 17.8 ops/cycle
    32x64 k64 seeded, C transposed, A in place: bitwise (1471 of 2048 differ from every step rounded to nearest), 3283 cycles, 79.8 ops/cycle
    19x29 k61 seeded, k = 61: a seeded k that the depth padding changes
    19x29 k62 seeded, k = 62: a seeded k that the depth padding changes |}]

(* The seed is not "compute the product, then add it": 2^25 plus two pairs of
   two ones is 2^25 when each pair is added to the running sum (2^25 + 2 ties
   to even), and 2^25 + 4 when the four ones are summed first. *)
let%expect_test "a seed is continued, not added after" =
  let ones count = String.concat "" (List.init count (fun _ -> "\x80\x3f")) in
  let d = descriptor ~a_transposed:true ~seeded:true ~m:16L ~n:16L ~k:4L () in
  let seed = String.concat "" (List.init 256 (fun _ -> f32_word 33554432.)) in
  (match
     run ~engine:Engine.packed ~operands:(ones 64, ones 64) ~seed ~spread:0 d
   with
  | Ok r ->
      Fmt.pr "%s: first word %ld, %d differ@."
        (if r.wrong = 0 then "bitwise" else "DIFFERS")
        (String.get_int32_le r.actual 0)
        r.wrong
  | Error e -> Fmt.pr "%s@." e);
  Fmt.pr "2^25 = %ld, 2^25 + 4 = %ld@."
    (Int32.bits_of_float 33554432.)
    (Int32.bits_of_float 33554436.);
  (match
     run
       ~engine:
         (Engine.Packed (Some Z.Zvt_packed_contraction.Mutation.Seed_ignored))
       ~operands:(ones 64, ones 64)
       ~seed ~spread:0 d
   with
  | Ok r ->
      Fmt.pr "seed ignored: %s@." (if r.wrong = 0 then "bitwise" else "DIFFERS")
  | Error e -> Fmt.pr "%s@." e);
  [%expect
    {|
    bitwise: first word 1275068416, 0 differ
    2^25 = 1275068416, 2^25 + 4 = 1275068417
    seed ignored: DIFFERS |}]

(* The pair rule was fitted on random data, where a sum is never zero. Every
   combination of signed zeros as the four products of a column's two pairs,
   against seeds of negative zero (even rows) and positive zero (odd rows):
   bitwise against [Paired], which follows the IEEE rules for a zero sum. *)
let%expect_test "signed zeros through the engine's pair" =
  let m = 16 and n = 16 and k = 4 in
  let d = descriptor ~a_transposed:true ~seeded:true ~m:16L ~n:16L ~k:4L () in
  let a = String.concat "" (List.init (m * k) (fun _ -> "\x80\x3f")) in
  let b =
    let buf = Bytes.create (2 * k * n) in
    for kk = 0 to k - 1 do
      for j = 0 to n - 1 do
        Bytes.set_uint16_le buf
          (2 * ((kk * n) + j))
          (if (j lsr kk) land 1 = 1 then 0x8000 else 0)
      done
    done;
    Bytes.to_string buf
  in
  let seed =
    String.concat ""
      (List.init (m * n) (fun e ->
           f32_word (if e / n land 1 = 0 then -0. else 0.)))
  in
  (match run ~engine:Engine.packed ~operands:(a, b) ~seed ~spread:0 d with
  | Ok r ->
      let negative = ref 0 in
      for e = 0 to (m * n) - 1 do
        if String.get_int32_le r.actual (4 * e) = 0x80000000l then incr negative
      done;
      Fmt.pr "%s, %d of %d results negative zero@."
        (if r.wrong = 0 then "bitwise" else Fmt.str "DIFFERS in %d" r.wrong)
        !negative (m * n)
  | Error e -> Fmt.pr "%s@." e);
  [%expect {| bitwise, 0 of 256 results negative zero |}]
