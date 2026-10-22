(* The same bundles on Coral's RTL vector core (rvv-core-mini-highmem-axi):
   bitwise against the binary32 oracle, scalar and Zve32f, with the cycles and
   instructions each plan takes over its every invocation. The RTL testbench
   is cycle-accurate for the core, so these are the first measured costs of
   the Zve32f plan; qemu's timing says nothing about an RV32 core. *)

module T = Machine_model_test.Model_test
module C = Machine_rivet_riscv32_coral.Rv32_mir_coral

let config =
  {
    C.Config.dir = Sys.getenv "CORALNPU_DATA";
    configuration = "rvv-core-mini-highmem-axi";
    deadline = 1200.;
    flags = [ "--cycles=500000000" ];
  }

let planned target =
  Ssa_backends.Pipeline.Planned
    { numerics = Ssa_ir.Ssa_numerics.Fp32_ordered; target }

let measure target name g =
  let cycles = ref 0L and retired = ref 0L in
  let on_run (o : Coral_run.outcome) =
    cycles := Int64.add !cycles o.Coral_run.mcycle;
    retired := Int64.add !retired o.Coral_run.minstret
  in
  T.check ~route:(C.route ~on_run config) ~pipeline:(planned target)
    ~oracle:T.Fp32 ~constant:T.positive ~calls:1 name g;
  (!cycles, !retired)

let%expect_test "scalar and Zve32f on the RTL vector core" =
  List.iter
    (fun (name, g) ->
      let sc, si = measure Ssa_ir.Ssa_target.scalar (name ^ ", scalar") g in
      let vc, vi =
        measure Ssa_ir.Ssa_target.zve32f128 (name ^ ", zve32f128") g
      in
      Fmt.pr "  cycles %Ld -> %Ld (%.2fx), instructions %Ld -> %Ld (%.2fx)@." sc
        vc
        (Int64.to_float sc /. Int64.to_float vc)
        si vi
        (Int64.to_float si /. Int64.to_float vi))
    T.vector_fixtures;
  [%expect
    {|
    relu then add over C=32, scalar (2 invocations): bitwise
    relu then add over C=32, zve32f128 (2 invocations): bitwise
      cycles 7572 -> 1909 (3.97x), instructions 4627 -> 1432 (3.23x)
    softmax over H, W=32, scalar (1 invocations): bitwise
    softmax over H, W=32, zve32f128 (1 invocations): bitwise
      cycles 495736 -> 495736 (1.00x), instructions 237748 -> 237748 (1.00x)
    layer_norm over H, W=32, scalar (1 invocations): bitwise
    layer_norm over H, W=32, zve32f128 (1 invocations): bitwise
      cycles 84070 -> 84070 (1.00x), instructions 46208 -> 46208 (1.00x)
    conv, batch norm, relu to C=32, scalar (3 invocations): bitwise
    conv, batch norm, relu to C=32, zve32f128 (3 invocations): bitwise
      cycles 192313 -> 23807 (8.08x), instructions 139487 -> 20566 (6.78x) |}]
