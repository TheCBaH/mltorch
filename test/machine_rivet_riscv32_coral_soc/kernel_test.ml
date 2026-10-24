(* The generated RVV function alone on the subsystem, operands in SRAM, at
   the DMA suite's resident-B shapes (gemm_rvv_float.t): a caller hands it
   A, B and C from the slot table, under the plan the transfer model
   chooses and under each candidate, C bitwise against the descriptor's
   reference, with the cycles of the whole call. *)

module A = Machine_rivet_riscv32.Rv32_asm
module C = Contraction
module L = Coral_layout
module P = Coral_transfer.Transfer_plan
module Tm = Coral_transfer.Transfer_model

let dir = Sys.getenv "CORALNPU_SUBSYSTEM_DATA"
let entry = "kernel_entry"

let caller d =
  A.module_ "kernel_caller" (fun () ->
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

(* small integers over 16: every product and sum exact, so C is the same
   bits whatever the requirement *)
let value seed e =
  let h = ((seed * 7919) + (e * 104729)) * 2654435761 land 0x3fffffff in
  float_of_int ((h mod 33) - 16) /. 16.

let bytes_of seed count =
  let b = Bytes.create (4 * count) in
  for e = 0 to count - 1 do
    Bytes.set_int32_le b (4 * e) (Int32.bits_of_float (value seed e))
  done;
  Bytes.to_string b

let descriptor ~m ~n ~k =
  Result.get_ok
    (Err.payload
       (C.make ~batch:1L ~m ~n ~k
          ~a:{ C.Operand.batch = 0L; outer = k; k = 1L }
          ~b:{ C.Operand.batch = 0L; outer = 1L; k = n }
          ~c:{ C.Output.batch = 0L; row = n; col = 1L }
          ~format:C.Element.F32 ~seed:C.Seed.Zero
          ~numerics:C.Numerics.Fused_sequential))

let run ?mutation ~configuration d plan =
  let m = Int64.to_int d.C.m and n = Int64.to_int d.C.n in
  let k = Int64.to_int d.C.k in
  let a = bytes_of 1 (m * k) and b = bytes_of 2 (k * n) in
  let read s off =
    Int32.float_of_bits (String.get_int32_le s (4 * Int64.to_int off))
  in
  let expected = Bytes.make (4 * m * n) '\000' in
  C.eval d ~a:(read a) ~b:(read b)
    ~c:(fun _ -> 0.)
    ~write:(fun off x ->
      Bytes.set_int32_le expected (4 * Int64.to_int off) (Int32.bits_of_float x));
  let ( let* ) = Result.bind in
  let* profile =
    Result.map_error
      (Fmt.str "%a" Coral_profile.pp_error)
      (Err.payload (Coral_profile.load ~dir configuration))
  in
  let budget =
    List.fold_left
      (fun acc (reg : Coral_profile.Region.t) ->
        match reg.Coral_profile.Region.role with
        | Coral_profile.Role.Dtcm ->
            acc
            + Int64.to_int
                (Int64.div (Int64.mul reg.Coral_profile.Region.size 5L) 8L)
        | _ -> acc)
      0 profile.Coral_profile.regions
  in
  let plan = match plan with Some p -> p | None -> Tm.choose ~budget d in
  let* w =
    Result.map_error (Fmt.str "%a" P.Refusal.pp) (P.check ~budget d plan)
  in
  let* helper = Coralnpu_kernels.Rvv_contraction.module_ ?mutation ~plan:w d in
  let* call = caller d in
  let slot placement size =
    {
      L.Slot.align = 16L;
      home = L.Home.Ram;
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
            ~matrix:false profile ~entry ~kernel:[ call; helper ]
            [
              slot L.Placement.In (4 * m * k);
              slot L.Placement.In (4 * k * n);
              slot L.Placement.Out (4 * m * n);
            ]))
  in
  let* o =
    Result.map_error
      (Fmt.str "%a" Coral_soc_run.Run_error.pp)
      (Err.payload
         (Coral_soc_run.run ~deadline:1800. ~read_back:[ 2 ] ~budget:65536L ~dir
            image ~bound:(function
           | 0 -> Some a
           | 1 -> Some b
           | _ -> None)))
  in
  Ok
    ( plan,
      List.assoc 2 o.Coral_soc_run.read_back = Bytes.to_string expected,
      o.Coral_soc_run.mcycle )

let%expect_test "the generated function at the DMA suite's shapes" =
  List.iter
    (fun (m, n, k) ->
      let d = descriptor ~m ~n ~k in
      let show what plan =
        match run ~configuration:"subsystem-vector-normal" d plan with
        | Ok (p, exact, cycles) ->
            Fmt.pr "%Ldx%Ld k%Ld, %s (%a): %s, %Ld cycles, %.2f ops/cycle@." m n
              k what P.pp p
              (if exact then "bitwise" else "DIFFERS")
              cycles
              (2. *. Int64.to_float (C.macs d) /. Int64.to_float cycles)
        | Error e -> Fmt.pr "%Ldx%Ld k%Ld, %s: %s@." m n k what e
      in
      show "chosen" None;
      List.iter
        (fun p -> show "candidate" (Some p))
        (P.candidates ~budget:20480 d))
    [ (192L, 32L, 128L) ];
  (* the shared copy's and C's ring obligations, each defect caught *)
  let d = descriptor ~m:192L ~n:32L ~k:128L in
  List.iter
    (fun (what, mutation) ->
      match run ~mutation ~configuration:"subsystem-vector-normal" d None with
      | Ok (_, exact, _) ->
          Fmt.pr "%s: %s@." what (if exact then "bitwise" else "DIFFERS")
      | Error e -> Fmt.pr "%s: %s@." what e)
    Coralnpu_kernels.Rvv_contraction.Mutation.
      [
        ("B's core share not copied", No_share);
        ("C out of the current half", C_current_half);
      ];
  [%expect
    {|
    192x32 k128, chosen (A streamed by 3 rows, B shared, C by the DMA): bitwise, 170634 cycles, 9.22 ops/cycle
    192x32 k128, candidate (A in place, B in place): bitwise, 526546 cycles, 2.99 ops/cycle
    192x32 k128, candidate (A streamed by 18 rows, B in place): bitwise, 437069 cycles, 3.60 ops/cycle
    192x32 k128, candidate (A streamed by 15 rows, B in place, C by the DMA): bitwise, 431444 cycles, 3.65 ops/cycle
    192x32 k128, candidate (A in place, B copied): bitwise, 279123 cycles, 5.64 ops/cycle
    192x32 k128, candidate (A streamed by 3 rows, B copied): bitwise, 186304 cycles, 8.44 ops/cycle
    192x32 k128, candidate (A streamed by 3 rows, B copied, C by the DMA): bitwise, 179896 cycles, 8.74 ops/cycle
    192x32 k128, candidate (A in place, B shared): bitwise, 269866 cycles, 5.83 ops/cycle
    192x32 k128, candidate (A streamed by 3 rows, B shared): bitwise, 177041 cycles, 8.88 ops/cycle
    192x32 k128, candidate (A streamed by 3 rows, B shared, C by the DMA): bitwise, 170634 cycles, 9.22 ops/cycle
    B's core share not copied: DIFFERS
    C out of the current half: DIFFERS |}]
