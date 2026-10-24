(* The binary32 bundles on the subsystem's normal-TCM vector core and on the
   standalone core of the same configuration: bitwise against the binary32
   oracle on both, and the cycles each run takes. Every slot is first fit, in
   DTCM, so the two should differ by code placement alone. *)

module T = Machine_model_test.Model_test
module C = Machine_rivet_riscv32_coral.Rv32_mir_coral

let subsystem =
  {
    C.Config.dir = Sys.getenv "CORALNPU_SUBSYSTEM_DATA";
    configuration = "subsystem-vector-normal";
    deadline = 1200.;
    flags = [];
  }

let standalone =
  {
    C.Config.dir = Sys.getenv "CORALNPU_DATA";
    configuration = "rvv-core-mini-axi";
    deadline = 1200.;
    flags = [ "--cycles=500000000" ];
  }

let zve32f =
  Ssa_backends.Pipeline.Planned
    {
      numerics = Ssa_ir.Ssa_numerics.Fp32_ordered;
      target = Ssa_ir.Ssa_target.zve32f128;
    }

let measure ?route_mutation ?home ?budget config name g =
  let cycles = ref 0L in
  let on_run (o : C.Run.t) = cycles := Int64.add !cycles o.C.Run.mcycle in
  T.check
    ~route:(C.route ?route_mutation ?home ?budget ~on_run config)
    ~pipeline:zve32f ~oracle:T.Fp32 ~constant:T.positive ~calls:1 name g;
  !cycles

let%expect_test "Zve32f bundles on the subsystem and standalone" =
  List.iter
    (fun (name, g) ->
      let s = measure subsystem (name ^ ", subsystem") g in
      let a = measure standalone (name ^ ", standalone") g in
      Fmt.pr "  cycles %Ld on the subsystem, %Ld standalone (%+Ld)@." s a
        (Int64.sub s a))
    T.vector_fixtures;
  [%expect
    {|
    relu then add over C=32, subsystem (2 invocations): bitwise
    relu then add over C=32, standalone (2 invocations): bitwise
      cycles 1910 on the subsystem, 1909 standalone (+1)
    softmax over H, W=32, subsystem (1 invocations): bitwise
    softmax over H, W=32, standalone (1 invocations): bitwise
      cycles 495832 on the subsystem, 495736 standalone (+96)
    layer_norm over H, W=32, subsystem (1 invocations): bitwise
    layer_norm over H, W=32, standalone (1 invocations): bitwise
      cycles 84166 on the subsystem, 84070 standalone (+96)
    conv, batch norm, relu to C=32, subsystem (3 invocations): bitwise
    conv, batch norm, relu to C=32, standalone (3 invocations): bitwise
      cycles 23831 on the subsystem, 23807 standalone (+24) |}]

(* Outputs stored under the wrong slot are caught by the oracle; a read-back
   larger than the budget is refused before any byte crosses the bridge. *)
let%expect_test "route defects on the subsystem" =
  let pointwise = List.assoc "relu then add over C=32" T.vector_fixtures in
  ignore
    (measure ~route_mutation:C.Mutation.Misplaced_output subsystem
       "misplaced output" pointwise);
  ignore (measure ~budget:16L subsystem "a 16-byte budget" pointwise);
  [%expect
    {|
    misplaced output (2 invocations): DIFFERS
    a 16-byte budget (2 invocations): invocation 0 (n0): unsupported(768 bytes to read back over the bridge, more than the budget of 16) |}]

(* The tensors pinned to SRAM, the rest first fit: the same bits, and the
   cost of reaching the tensors over the bus. *)
let%expect_test "tensors in SRAM" =
  let home (s : Machine_rivet_common.Rivet_table.slot) =
    match s.Machine_rivet_common.Rivet_table.section with
    | Machine_model.Mir_artifact.Section.Bound -> Coral_layout.Home.Ram
    | _ -> Coral_layout.Home.Any
  in
  List.iter
    (fun (name, g) ->
      let dtcm = measure subsystem (name ^ ", DTCM") g in
      let sram = measure ~home subsystem (name ^ ", SRAM") g in
      Fmt.pr "  cycles %Ld in DTCM, %Ld in SRAM@." dtcm sram)
    [ List.hd T.vector_fixtures; List.nth T.vector_fixtures 3 ];
  [%expect
    {|
    relu then add over C=32, DTCM (2 invocations): bitwise
    relu then add over C=32, SRAM (2 invocations): bitwise
      cycles 1910 in DTCM, 2778 in SRAM
    conv, batch norm, relu to C=32, DTCM (3 invocations): bitwise
    conv, batch norm, relu to C=32, SRAM (3 invocations): bitwise
      cycles 23831 in DTCM, 45698 in SRAM |}]
