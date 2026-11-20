(* One invocation's program as each target's stage prints it: the generic
   program, the target's selection, its reference allocation, and the realized
   and scanned artifacts with their summaries. *)

module F = Native_test.Graph_fixtures
module M = Machine_model.Mir_model
module Key = Lowering_key

let relu () =
  F.build "relu"
    Graph_builder.(
      let* x = input ~shape:(F.s 1 1 1 1 1 4) () in
      relu x)

(* binary32 scalar, what every target selects *)
let fp32_scalar =
  Ssa_backends.Pipeline.Planned
    {
      numerics = Ssa_ir.Ssa_numerics.Fp32_ordered;
      target = Ssa_ir.Ssa_target.scalar;
    }

let lowered () =
  let b =
    Err.or_raise ~pp_error:Loop_ir.Loop_bundle.pp_error
      (Loop_ir.Loop_bundle.build (relu ()))
  in
  let inv = List.hd b.Loop_ir.Loop_bundle.invocations in
  Result.get_ok (M.lower_one ~pipeline:fp32_scalar ~invocation:0l inv)

(* The first lines and the length stand for the text: the whole of one stage is
   pinned below. *)
let show target stage =
  Fmt.pr "--- %s %s: " (Key.Target.name target) (Key.Mir_stage.name stage);
  match M.Stage_text.of_lowered target stage (lowered ()) with
  | Ok { M.Stage_text.text; note } ->
      let lines = String.split_on_char '\n' text in
      Fmt.pr "%d lines@." (List.length lines);
      List.iteri (fun i l -> if i < 3 then Fmt.pr "  %s@." l) lines;
      Option.iter (Fmt.pr "  note: %s@.") note
  | Error r -> Fmt.pr "refused: %a@." M.Reason.pp r

let%expect_test "every target and stage of one invocation" =
  List.iter
    (fun target -> List.iter (fun stage -> show target stage) Key.Mir_stage.all)
    Key.Target.all;
  [%expect
    {|
    --- aarch64 allocated: 90 lines
      target aarch64 [fp]
      fn0 main func kernel() -> (w0) {
        slot slot6: 4 bytes align 4
    --- aarch64 generic: 43 lines
      data_model lp64-le-elf
      planning subject=47a6877330019e499243d7b5b8317747 policy=numerics=fp32_ordered vectorized=f32 other=f32 fma=none reassociation=none schedule=scalar/row_block=1 precision=f32 lanes=1 fma=forbidden capabilities=vector_bits:64
      region0: 16 bytes align 16 bound
    --- aarch64 realized: 81 lines
      target aarch64 [fp]
      fn0 main func kernel() -> (w0) frame 112 {
      bb0():
      note: aarch64 (Arm ARM DDI 0487 K.a) features [fp], numerics=fp32_ordered vectorized=f32 other=f32 fma=none reassociation=none/scalar/row_block=1/forbidden; symbols: 2 bound, 1 bss, 0 rodata, 0 external [], 1 function; 4 relocations
    --- aarch64 scanned: 39 lines
      target aarch64 [fp]
      fn0 main func kernel() -> (w0) frame 16 {
      bb0():
      note: aarch64 (Arm ARM DDI 0487 K.a) features [fp], numerics=fp32_ordered vectorized=f32 other=f32 fma=none reassociation=none/scalar/row_block=1/forbidden; symbols: 2 bound, 1 bss, 0 rodata, 0 external [], 1 function; 4 relocations
    --- aarch64 selected: 39 lines
      target aarch64 [fp]
      data_model lp64-le-elf
      planning subject=47a6877330019e499243d7b5b8317747 policy=numerics=fp32_ordered vectorized=f32 other=f32 fma=none reassociation=none schedule=scalar/row_block=1 precision=f32 lanes=1 fma=forbidden capabilities=vector_bits:64
    --- riscv32 allocated: 88 lines
      target riscv32 []
      fn0 main func kernel() -> (a0) {
        slot slot6: 4 bytes align 4
    --- riscv32 generic: 43 lines
      data_model lp64-le-elf
      planning subject=47a6877330019e499243d7b5b8317747 policy=numerics=fp32_ordered vectorized=f32 other=f32 fma=none reassociation=none schedule=scalar/row_block=1 precision=f32 lanes=1 fma=forbidden capabilities=vector_bits:64
      region0: 16 bytes align 16 bound
    --- riscv32 realized: 73 lines
      target riscv32 []
      fn0 main func kernel() -> (a0) frame 80 {
      bb0():
      note: riscv32 (The RISC-V Instruction Set Manual, Volume I 20240411) features [], numerics=fp32_ordered vectorized=f32 other=f32 fma=none reassociation=none/scalar/row_block=1/forbidden; symbols: 2 bound, 1 bss, 0 rodata, 0 external [], 1 function; 2 relocations
    --- riscv32 scanned: 28 lines
      target riscv32 []
      fn0 main func kernel() -> (a0) frame 0 {
      bb0():
      note: riscv32 (The RISC-V Instruction Set Manual, Volume I 20240411) features [], numerics=fp32_ordered vectorized=f32 other=f32 fma=none reassociation=none/scalar/row_block=1/forbidden; symbols: 2 bound, 1 bss, 0 rodata, 0 external [], 1 function; 2 relocations
    --- riscv32 selected: 36 lines
      target riscv32 []
      data_model ilp32-le-elf
      planning subject=47a6877330019e499243d7b5b8317747 policy=numerics=fp32_ordered vectorized=f32 other=f32 fma=none reassociation=none schedule=scalar/row_block=1 precision=f32 lanes=1 fma=forbidden capabilities=vector_bits:64
    --- riscv32_grouped allocated: refused: the reference allocator does not allocate register groups (allocated)
    --- riscv32_grouped generic: 43 lines
      data_model lp64-le-elf
      planning subject=47a6877330019e499243d7b5b8317747 policy=numerics=fp32_ordered vectorized=f32 other=f32 fma=none reassociation=none schedule=scalar/row_block=1 precision=f32 lanes=1 fma=forbidden capabilities=vector_bits:64
      region0: 16 bytes align 16 bound
    --- riscv32_grouped realized: refused: the reference allocator does not allocate register groups (realized)
    --- riscv32_grouped scanned: 28 lines
      target riscv32 []
      fn0 main func kernel() -> (a0) frame 0 {
      bb0():
      note: riscv32 (The RISC-V Instruction Set Manual, Volume I 20240411) features [], numerics=fp32_ordered vectorized=f32 other=f32 fma=none reassociation=none/scalar/row_block=1/forbidden; symbols: 2 bound, 1 bss, 0 rodata, 0 external [], 1 function; 2 relocations
    --- riscv32_grouped selected: 36 lines
      target riscv32 []
      data_model ilp32-le-elf
      planning subject=47a6877330019e499243d7b5b8317747 policy=numerics=fp32_ordered vectorized=f32 other=f32 fma=none reassociation=none schedule=scalar/row_block=1 precision=f32 lanes=1 fma=forbidden capabilities=vector_bits:64
    --- x86_64 allocated: 115 lines
      target x86_64 [sse2]
      fn0 main func kernel() -> (eax) {
        slot slot6: 4 bytes align 4
    --- x86_64 generic: 43 lines
      data_model lp64-le-elf
      planning subject=47a6877330019e499243d7b5b8317747 policy=numerics=fp32_ordered vectorized=f32 other=f32 fma=none reassociation=none schedule=scalar/row_block=1 precision=f32 lanes=1 fma=forbidden capabilities=vector_bits:64
      region0: 16 bytes align 16 bound
    --- x86_64 realized: 94 lines
      target x86_64 [sse2]
      fn0 main func kernel() -> (eax) frame 120 {
      bb0():
      note: x86_64 (Intel SDM 325462 084) features [sse2], numerics=fp32_ordered vectorized=f32 other=f32 fma=none reassociation=none/scalar/row_block=1/forbidden; symbols: 2 bound, 1 bss, 0 rodata, 0 external [], 1 function; 2 relocations
    --- x86_64 scanned: 40 lines
      target x86_64 [sse2]
      fn0 main func kernel() -> (eax) frame 8 {
      bb0():
      note: x86_64 (Intel SDM 325462 084) features [sse2], numerics=fp32_ordered vectorized=f32 other=f32 fma=none reassociation=none/scalar/row_block=1/forbidden; symbols: 2 bound, 1 bss, 0 rodata, 0 external [], 1 function; 2 relocations
    --- x86_64 selected: 45 lines
      target x86_64 [sse2]
      data_model lp64-le-elf
      planning subject=47a6877330019e499243d7b5b8317747 policy=numerics=fp32_ordered vectorized=f32 other=f32 fma=none reassociation=none schedule=scalar/row_block=1 precision=f32 lanes=1 fma=forbidden capabilities=vector_bits:64 |}]

let whole target stage =
  match M.Stage_text.of_lowered target stage (lowered ()) with
  | Ok { M.Stage_text.text; _ } -> Fmt.pr "%s@." text
  | Error r -> Fmt.pr "refused: %a@." M.Reason.pp r

let%expect_test
    "the scanned program of one target, whole: operands are locations" =
  whole Key.Target.Riscv32 Key.Mir_stage.Scanned;
  [%expect
    {|
    target riscv32 []
    fn0 main func kernel() -> (a0) frame 0 {
    bb0():
      jump bb1
    bb1():
      a0 = li.i32 0
      a1 = li.i32 4
      a2 = li.i32 0
      fa0 = fmv.w.x a2
      a2 = la view0
      a3 = la view1
      jump bb2
    bb2(a0, a2, a3):
      branch blt a0, a1, bb4, bb3
    bb3():
      a0 = li.i32 0
      return a0
    bb4():
      fa1 = flw 0(a2)
      a4 = flt.s fa1, fa0
      fa1 = fsel a4, fa0, fa1
      fsw fa1, 0(a3)
      a0 = addi a0, 1
      a4 = li.i32 4
      a2 = add a2, a4
      a3 = add a3, a4
      jump bb2
    } |}]

let%expect_test "the selected program of one target, whole" =
  (match
     M.Stage_text.of_lowered Key.Target.Riscv32 Key.Mir_stage.Selected
       (lowered ())
   with
  | Ok { M.Stage_text.text; _ } -> Fmt.pr "%s@." text
  | Error r -> Fmt.pr "refused: %a@." M.Reason.pp r);
  [%expect
    {|
    target riscv32 []
    data_model ilp32-le-elf
    planning subject=47a6877330019e499243d7b5b8317747 policy=numerics=fp32_ordered vectorized=f32 other=f32 fma=none reassociation=none schedule=scalar/row_block=1 precision=f32 lanes=1 fma=forbidden capabilities=vector_bits:64
    region0: 16 bytes align 16 bound
    region1: 16 bytes align 16 bound
    region1000000: 104 bytes align 16 uninit
    view0: region0[0, +16) r in t0
    view1: region1[0, +16) rw out t1
    view1000000: region1000000[0, +104) rw runtime
    fn0 main func kernel -> (i32) {
    bb0(; !0):
      jump bb1(; !0)
    bb1(; !1):
      %2 = li.i32 0
      %3 = li.i32 4
      %4 = li.i32 0
      %5 = fmv.w.x %4
      %6 = la view0
      %7 = la view1
      jump bb2(%2, %6, %7; !1)
    bb2(%8: i32, %9: ptr64, %10: ptr64; !11):
      branch blt %8, %3, bb4(; !11), bb3(; !11)
    bb3(; !12):
      %13 = li.i32 0
      return %13; !12
    bb4(; !14):
      %15, !16 = flw 0(%9), !14
      %17 = flt.s %15, %5
      %18 = fsel %17, %5, %15
      !19 = fsw %18, 0(%10), !16
      %20 = addi %8, 1
      %21 = li.i32 4
      %22 = add %9, %21
      %23 = add %10, %21
      jump bb2(%20, %22, %23; !19)
    } |}]
