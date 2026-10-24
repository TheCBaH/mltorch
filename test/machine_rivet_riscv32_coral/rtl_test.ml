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

let measure ?(numerics = Ssa_ir.Ssa_numerics.Fp32_ordered) ?(oracle = T.Fp32)
    ?registers target name g =
  let cycles = ref 0L and retired = ref 0L in
  let on_run (o : C.Run.t) =
    cycles := Int64.add !cycles o.C.Run.mcycle;
    retired := Int64.add !retired o.C.Run.minstret
  in
  T.check
    ~route:(C.route ?registers ~on_run config)
    ~pipeline:(Ssa_backends.Pipeline.Planned { numerics; target })
    ~oracle ~constant:T.positive ~calls:1 name g;
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
      cycles 7572 -> 1915 (3.95x), instructions 4627 -> 1432 (3.23x)
    softmax over H, W=32, scalar (1 invocations): bitwise
    softmax over H, W=32, zve32f128 (1 invocations): bitwise
      cycles 495736 -> 495736 (1.00x), instructions 237748 -> 237748 (1.00x)
    layer_norm over H, W=32, scalar (1 invocations): bitwise
    layer_norm over H, W=32, zve32f128 (1 invocations): bitwise
      cycles 84070 -> 84070 (1.00x), instructions 46208 -> 46208 (1.00x)
    conv, batch norm, relu to C=32, scalar (3 invocations): bitwise
    conv, batch norm, relu to C=32, zve32f128 (3 invocations): bitwise
      cycles 192313 -> 23903 (8.05x), instructions 139487 -> 20566 (6.78x) |}]

(* The matrix products on the same core: whole-invocation cycles, and the
   rate in operations a cycle (a multiply-add is two) against the peak
   suite's RVV FP32 GEMM burst of 9.9 at C[9,32], K=64. *)
let%expect_test "matrix products on the RTL vector core" =
  List.iter
    (fun (name, g, macs) ->
      let sc, si = measure Ssa_ir.Ssa_target.scalar (name ^ ", scalar") g in
      let vc, vi =
        measure Ssa_ir.Ssa_target.zve32f128 (name ^ ", zve32f128") g
      in
      let rate c = 2. *. float_of_int macs /. Int64.to_float c in
      Fmt.pr
        "  cycles %Ld -> %Ld (%.2fx), instructions %Ld -> %Ld (%.2fx); %.3f -> \
         %.3f ops/cycle@."
        sc vc
        (Int64.to_float sc /. Int64.to_float vc)
        si vi
        (Int64.to_float si /. Int64.to_float vi)
        (rate sc) (rate vc))
    (List.map
       (fun (name, g) -> (name, g, T.contraction_macs g))
       T.contraction_fixtures);
  [%expect
    {|
    linear 9x32 k64, scalar (1 invocations): bitwise
    linear 9x32 k64, zve32f128 (1 invocations): bitwise
      cycles 196231 -> 38509 (5.10x), instructions 89873 -> 31021 (2.90x); 0.188 -> 0.957 ops/cycle
    linear 9x32 k64 bias, scalar (1 invocations): bitwise
    linear 9x32 k64 bias, zve32f128 (1 invocations): bitwise
      cycles 199053 -> 38773 (5.13x), instructions 91103 -> 31162 (2.92x); 0.185 -> 0.951 ops/cycle
    linear 7x29 k63 bias, scalar (1 invocations): bitwise
    linear 7x29 k63 bias, zve32f128 (1 invocations): bitwise
      cycles 141784 -> 81254 (1.74x), instructions 65773 -> 45457 (1.45x); 0.180 -> 0.315 ops/cycle
    bmm 1x9x32 k64, scalar (1 invocations): bitwise
    bmm 1x9x32 k64, zve32f128 (1 invocations): bitwise
      cycles 200460 -> 27045 (7.41x), instructions 94319 -> 25723 (3.67x); 0.184 -> 1.363 ops/cycle
    bmm 3x7x29 k63, scalar (1 invocations): bitwise
    bmm 3x7x29 k63, zve32f128 (1 invocations): bitwise
      cycles 428682 -> 239324 (1.79x), instructions 204858 -> 134687 (1.52x); 0.179 -> 0.321 ops/cycle |}]

(* The contracted policy's fused accumulation ([vfmacc.vv]) against the
   ordered Zve32f plan on the same core: bitwise against each policy's own
   oracle, and the cycles each takes. *)
let%expect_test "matrix products: contracted against ordered Zve32f" =
  List.iter
    (fun (name, g) ->
      let oc, _ = measure Ssa_ir.Ssa_target.zve32f128 (name ^ ", ordered") g in
      let cc, _ =
        measure ~numerics:Ssa_ir.Ssa_numerics.Fp32_contracted
          ~oracle:T.Fp32_contracted Ssa_ir.Ssa_target.zve32f128
          (name ^ ", contracted") g
      in
      let rate c =
        2. *. float_of_int (T.contraction_macs g) /. Int64.to_float c
      in
      Fmt.pr "  cycles %Ld -> %Ld (%.2fx); %.3f -> %.3f ops/cycle@." oc cc
        (Int64.to_float oc /. Int64.to_float cc)
        (rate oc) (rate cc))
    T.contraction_fixtures;
  [%expect
    {|
    linear 9x32 k64, ordered (1 invocations): bitwise
    linear 9x32 k64, contracted (1 invocations): bitwise
      cycles 38509 -> 34910 (1.10x); 0.957 -> 1.056 ops/cycle
    linear 9x32 k64 bias, ordered (1 invocations): bitwise
    linear 9x32 k64 bias, contracted (1 invocations): bitwise
      cycles 38773 -> 35185 (1.10x); 0.951 -> 1.048 ops/cycle
    linear 7x29 k63 bias, ordered (1 invocations): bitwise
    linear 7x29 k63 bias, contracted (1 invocations): bitwise
      cycles 81254 -> 56355 (1.44x); 0.315 -> 0.454 ops/cycle
    bmm 1x9x32 k64, ordered (1 invocations): bitwise
    bmm 1x9x32 k64, contracted (1 invocations): bitwise
      cycles 27045 -> 24211 (1.12x); 1.363 -> 1.523 ops/cycle
    bmm 3x7x29 k63, ordered (1 invocations): bitwise
    bmm 3x7x29 k63, contracted (1 invocations): bitwise
      cycles 239324 -> 159944 (1.50x); 0.321 -> 0.480 ops/cycle |}]

(* Sixteen-lane vectors in groups of four registers (LMUL 4) against one
   register of four lanes, the same Zve32f plan: bitwise against the oracle,
   and the cycles each takes. *)
let%expect_test "register groups against single registers" =
  let grouped = Machine_rivet_riscv32_mir.Rv32_mir_route.Registers.Grouped in
  List.iter
    (fun (numerics, oracle, pname) ->
      List.iter
        (fun (name, g) ->
          let name = Fmt.str "%s, %s" name pname in
          let sc, _ =
            measure ~numerics ~oracle Ssa_ir.Ssa_target.zve32f128
              (name ^ ", m1") g
          in
          let gc, _ =
            measure ~numerics ~oracle ~registers:grouped
              Ssa_ir.Ssa_target.zve32f128 (name ^ ", m4") g
          in
          Fmt.pr "  cycles %Ld -> %Ld (%.2fx)@." sc gc
            (Int64.to_float sc /. Int64.to_float gc))
        (List.hd T.vector_fixtures
        :: List.nth T.vector_fixtures 3
        :: T.contraction_fixtures))
    [
      (Ssa_ir.Ssa_numerics.Fp32_ordered, T.Fp32, "ordered");
      (Ssa_ir.Ssa_numerics.Fp32_contracted, T.Fp32_contracted, "contracted");
    ];
  [%expect
    {|
    relu then add over C=32, ordered, m1 (2 invocations): bitwise
    relu then add over C=32, ordered, m4 (2 invocations): bitwise
      cycles 1915 -> 1152 (1.66x)
    conv, batch norm, relu to C=32, ordered, m1 (3 invocations): bitwise
    conv, batch norm, relu to C=32, ordered, m4 (3 invocations): bitwise
      cycles 23903 -> 20966 (1.14x)
    linear 9x32 k64, ordered, m1 (1 invocations): bitwise
    linear 9x32 k64, ordered, m4 (1 invocations): bitwise
      cycles 38509 -> 31685 (1.22x)
    linear 9x32 k64 bias, ordered, m1 (1 invocations): bitwise
    linear 9x32 k64 bias, ordered, m4 (1 invocations): bitwise
      cycles 38773 -> 25290 (1.53x)
    linear 7x29 k63 bias, ordered, m1 (1 invocations): bitwise
    linear 7x29 k63 bias, ordered, m4 (1 invocations): bitwise
      cycles 81254 -> 76071 (1.07x)
    bmm 1x9x32 k64, ordered, m1 (1 invocations): bitwise
    bmm 1x9x32 k64, ordered, m4 (1 invocations): bitwise
      cycles 27045 -> 16285 (1.66x)
    bmm 3x7x29 k63, ordered, m1 (1 invocations): bitwise
    bmm 3x7x29 k63, ordered, m4 (1 invocations): bitwise
      cycles 239324 -> 223919 (1.07x)
    relu then add over C=32, contracted, m1 (2 invocations): bitwise
    relu then add over C=32, contracted, m4 (2 invocations): bitwise
      cycles 1915 -> 1152 (1.66x)
    conv, batch norm, relu to C=32, contracted, m1 (3 invocations): bitwise
    conv, batch norm, relu to C=32, contracted, m4 (3 invocations): bitwise
      cycles 22055 -> 19865 (1.11x)
    linear 9x32 k64, contracted, m1 (1 invocations): bitwise
    linear 9x32 k64, contracted, m4 (1 invocations): bitwise
      cycles 34910 -> 38079 (0.92x)
    linear 9x32 k64 bias, contracted, m1 (1 invocations): bitwise
    linear 9x32 k64 bias, contracted, m4 (1 invocations): bitwise
      cycles 35185 -> 31644 (1.11x)
    linear 7x29 k63 bias, contracted, m1 (1 invocations): bitwise
    linear 7x29 k63 bias, contracted, m4 (1 invocations): bitwise
      cycles 56355 -> 51774 (1.09x)
    bmm 1x9x32 k64, contracted, m1 (1 invocations): bitwise
    bmm 1x9x32 k64, contracted, m4 (1 invocations): bitwise
      cycles 24211 -> 18333 (1.32x)
    bmm 3x7x29 k63, contracted, m1 (1 invocations): bitwise
    bmm 3x7x29 k63, contracted, m4 (1 invocations): bitwise
      cycles 159944 -> 148382 (1.08x) |}]
