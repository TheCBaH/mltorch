(* A kernel that does not fit the instruction memory runs from overlays on the
   subsystem, which can load code. The generated bfloat16 [Sdpa] (masked, 9x29,
   d64) takes 6.5 KiB of it here; on a core with [itcm] bytes of instruction
   memory the route moves two unrolled loop bodies of the kernel out of the
   image and the result is bitwise against the binary32 oracle all the same.
   The overlays cost a reload each time the loop alternates between them. *)

module T = Machine_model_test.Model_test
module C = Machine_rivet_riscv32_coral.Rv32_mir_coral

let config =
  {
    C.Config.dir = Sys.getenv "CORALNPU_SUBSYSTEM_DATA";
    configuration = "subsystem-matrix-normal";
    deadline = 3600.;
    flags = [];
  }

let fixture = List.nth T.sdpa_bf16_fixtures 2

let run ?itcm label =
  let name, g = fixture in
  let cycles = ref 0L in
  let on_run (o : C.Run.t) = cycles := Int64.add !cycles o.C.Run.mcycle in
  T.check
    ~route:(C.route ?itcm ~on_run config)
    ~contractions:true
    ~pipeline:
      (Ssa_backends.Pipeline.Planned
         {
           numerics = Ssa_ir.Ssa_numerics.Fp32_ordered;
           target = Ssa_ir.Ssa_target.zve32f128;
         })
    ~oracle:T.Fp32 ~same:T.same_portable
    ~input:(fun ~salt id sg -> T.in_format sg (T.adversarial_input ~salt id sg))
    ~calls:1
    (name ^ ", " ^ label)
    g;
  Fmt.pr "  cycles %Ld@." !cycles

let%expect_test "the kernel runs from overlays on a smaller instruction memory"
    =
  run "all of the instruction memory";
  run ~itcm:6656L "6656 bytes of it";
  [%expect
    {|
    sdpa bf16 9x29 d64 masked, all of the instruction memory (1 invocations): bitwise
      cycles 1780310
    sdpa bf16 9x29 d64 masked, 6656 bytes of it (1 invocations): bitwise
      cycles 1830995 |}]

(* Past what two regions can free, the image is refused as before. *)
let%expect_test "an instruction memory too small even for the overlays" =
  run ~itcm:4096L "4096 bytes of it";
  [%expect
    {|
    sdpa bf16 9x29 d64 masked, 4096 bytes of it: refused: invocation 0 (n0): itcm: 6576 bytes needed, 4096 available
      cycles 0 |}]
