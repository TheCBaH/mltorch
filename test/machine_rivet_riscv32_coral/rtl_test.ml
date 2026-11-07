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
    deadline = Deadline.seconds;
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
    (T.vector_fixtures @ T.gemma_fixtures @ T.gemma_bf16_fixtures);
  [%expect
    {|
    relu then add over C=32, scalar (2 invocations): bitwise
    relu then add over C=32, zve32f128 (2 invocations): bitwise
      cycles 7569 -> 1916 (3.95x), instructions 4627 -> 1432 (3.23x)
    softmax over H, W=32, scalar (1 invocations): bitwise
    softmax over H, W=32, zve32f128 (1 invocations): bitwise
      cycles 495736 -> 495736 (1.00x), instructions 237748 -> 237748 (1.00x)
    layer_norm over H, W=32, scalar (1 invocations): bitwise
    layer_norm over H, W=32, zve32f128 (1 invocations): bitwise
      cycles 84166 -> 84166 (1.00x), instructions 46208 -> 46208 (1.00x)
    conv, batch norm, relu to C=32, scalar (3 invocations): bitwise
    conv, batch norm, relu to C=32, zve32f128 (3 invocations): bitwise
      cycles 192021 -> 23801 (8.07x), instructions 139487 -> 20566 (6.78x)
    residual add 2x160, scalar (1 invocations): bitwise
    residual add 2x160, zve32f128 (1 invocations): bitwise
      cycles 5869 -> 971 (6.04x), instructions 3609 -> 817 (4.42x)
    residual add 1x67, scalar (1 invocations): bitwise
    residual add 1x67, zve32f128 (1 invocations): bitwise
      cycles 2407 -> 417 (5.77x), instructions 1041 -> 244 (4.27x)
    residual add 1x33, scalar (1 invocations): bitwise
    residual add 1x33, zve32f128 (1 invocations): bitwise
      cycles 1217 -> 223 (5.46x), instructions 531 -> 134 (3.96x)
    rms norm 2x160, scalar (1 invocations): bitwise
    rms norm 2x160, zve32f128 (1 invocations): bitwise
      cycles 18846 -> 18846 (1.00x), instructions 10714 -> 10714 (1.00x)
    rms norm 1x67, scalar (1 invocations): bitwise
    rms norm 1x67, zve32f128 (1 invocations): bitwise
      cycles 5246 -> 5246 (1.00x), instructions 2686 -> 2686 (1.00x)
    rms norm 1x33, scalar (1 invocations): bitwise
    rms norm 1x33, zve32f128 (1 invocations): bitwise
      cycles 2662 -> 2662 (1.00x), instructions 1360 -> 1360 (1.00x)
    tanh gelu mul 2x160, scalar (2 invocations): bitwise
    tanh gelu mul 2x160, zve32f128 (2 invocations): bitwise
      cycles 167914 -> 54253 (3.10x), instructions 78164 -> 26216 (2.98x)
    tanh gelu mul 1x67, scalar (2 invocations): bitwise
    tanh gelu mul 1x67, zve32f128 (2 invocations): bitwise
      cycles 37132 -> 12782 (2.91x), instructions 16894 -> 6117 (2.76x)
    tanh gelu mul 1x33, scalar (2 invocations): bitwise
    tanh gelu mul 1x33, zve32f128 (2 invocations): bitwise
      cycles 18364 -> 6225 (2.95x), instructions 8368 -> 3005 (2.78x)
    residual add bf16 2x160, scalar (1 invocations): bitwise
    residual add bf16 2x160, zve32f128 (1 invocations): bitwise
      cycles 7798 -> 1513 (5.15x), instructions 5219 -> 1402 (3.72x)
    residual add bf16 1x67, scalar (1 invocations): bitwise
    residual add bf16 1x67, zve32f128 (1 invocations): bitwise
      cycles 2809 -> 504 (5.57x), instructions 1443 -> 386 (3.74x)
    residual add bf16 1x33, scalar (1 invocations): bitwise
    residual add bf16 1x33, zve32f128 (1 invocations): bitwise
      cycles 1415 -> 264 (5.36x), instructions 729 -> 202 (3.61x)
    rms norm bf16 2x160, scalar (1 invocations): bitwise
    rms norm bf16 2x160, zve32f128 (1 invocations): bitwise
      cycles 22058 -> 22058 (1.00x), instructions 12967 -> 12967 (1.00x)
    rms norm bf16 1x67, scalar (1 invocations): bitwise
    rms norm bf16 1x67, zve32f128 (1 invocations): bitwise
      cycles 5916 -> 5916 (1.00x), instructions 3222 -> 3222 (1.00x)
    rms norm bf16 1x33, scalar (1 invocations): bitwise
    rms norm bf16 1x33, zve32f128 (1 invocations): bitwise
      cycles 2992 -> 2992 (1.00x), instructions 1624 -> 1624 (1.00x)
    tanh gelu mul bf16 2x160, scalar (2 invocations): bitwise
    tanh gelu mul bf16 2x160, zve32f128 (2 invocations): bitwise
      cycles 169852 -> 56196 (3.02x), instructions 80105 -> 28572 (2.80x)
    tanh gelu mul bf16 1x67, scalar (2 invocations): bitwise
    tanh gelu mul bf16 1x67, zve32f128 (2 invocations): bitwise
      cycles 37467 -> 13173 (2.84x), instructions 17430 -> 6625 (2.63x)
    tanh gelu mul bf16 1x33, scalar (2 invocations): bitwise
    tanh gelu mul bf16 1x33, zve32f128 (2 invocations): bitwise
      cycles 18529 -> 6422 (2.89x), instructions 8632 -> 3257 (2.65x) |}]

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
      cycles 196311 -> 38508 (5.10x), instructions 89873 -> 31021 (2.90x); 0.188 -> 0.957 ops/cycle
    linear 9x32 k64 bias, scalar (1 invocations): bitwise
    linear 9x32 k64 bias, zve32f128 (1 invocations): bitwise
      cycles 199197 -> 38774 (5.14x), instructions 91103 -> 31162 (2.92x); 0.185 -> 0.951 ops/cycle
    linear 7x29 k63 bias, scalar (1 invocations): bitwise
    linear 7x29 k63 bias, zve32f128 (1 invocations): bitwise
      cycles 141777 -> 81303 (1.74x), instructions 65773 -> 45457 (1.45x); 0.180 -> 0.315 ops/cycle
    bmm 1x9x32 k64, scalar (1 invocations): bitwise
    bmm 1x9x32 k64, zve32f128 (1 invocations): bitwise
      cycles 200460 -> 27043 (7.41x), instructions 94319 -> 25723 (3.67x); 0.184 -> 1.363 ops/cycle
    bmm 3x7x29 k63, scalar (1 invocations): bitwise
    bmm 3x7x29 k63, zve32f128 (1 invocations): bitwise
      cycles 428490 -> 239216 (1.79x), instructions 204858 -> 134687 (1.52x); 0.179 -> 0.321 ops/cycle |}]

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
      cycles 38508 -> 34909 (1.10x); 0.957 -> 1.056 ops/cycle
    linear 9x32 k64 bias, ordered (1 invocations): bitwise
    linear 9x32 k64 bias, contracted (1 invocations): bitwise
      cycles 38774 -> 35186 (1.10x); 0.951 -> 1.048 ops/cycle
    linear 7x29 k63 bias, ordered (1 invocations): bitwise
    linear 7x29 k63 bias, contracted (1 invocations): bitwise
      cycles 81303 -> 56362 (1.44x); 0.315 -> 0.454 ops/cycle
    bmm 1x9x32 k64, ordered (1 invocations): bitwise
    bmm 1x9x32 k64, contracted (1 invocations): bitwise
      cycles 27043 -> 24209 (1.12x); 1.363 -> 1.523 ops/cycle
    bmm 3x7x29 k63, ordered (1 invocations): bitwise
    bmm 3x7x29 k63, contracted (1 invocations): bitwise
      cycles 239216 -> 161201 (1.48x); 0.321 -> 0.476 ops/cycle |}]

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
      cycles 1916 -> 1150 (1.67x)
    conv, batch norm, relu to C=32, ordered, m1 (3 invocations): bitwise
    conv, batch norm, relu to C=32, ordered, m4 (3 invocations): bitwise
      cycles 23801 -> 20993 (1.13x)
    linear 9x32 k64, ordered, m1 (1 invocations): bitwise
    linear 9x32 k64, ordered, m4 (1 invocations): bitwise
      cycles 38508 -> 31684 (1.22x)
    linear 9x32 k64 bias, ordered, m1 (1 invocations): bitwise
    linear 9x32 k64 bias, ordered, m4 (1 invocations): bitwise
      cycles 38774 -> 25290 (1.53x)
    linear 7x29 k63 bias, ordered, m1 (1 invocations): bitwise
    linear 7x29 k63 bias, ordered, m4 (1 invocations): bitwise
      cycles 81303 -> 76078 (1.07x)
    bmm 1x9x32 k64, ordered, m1 (1 invocations): bitwise
    bmm 1x9x32 k64, ordered, m4 (1 invocations): bitwise
      cycles 27043 -> 16285 (1.66x)
    bmm 3x7x29 k63, ordered, m1 (1 invocations): bitwise
    bmm 3x7x29 k63, ordered, m4 (1 invocations): bitwise
      cycles 239216 -> 223916 (1.07x)
    relu then add over C=32, contracted, m1 (2 invocations): bitwise
    relu then add over C=32, contracted, m4 (2 invocations): bitwise
      cycles 1916 -> 1150 (1.67x)
    conv, batch norm, relu to C=32, contracted, m1 (3 invocations): bitwise
    conv, batch norm, relu to C=32, contracted, m4 (3 invocations): bitwise
      cycles 21977 -> 19886 (1.11x)
    linear 9x32 k64, contracted, m1 (1 invocations): bitwise
    linear 9x32 k64, contracted, m4 (1 invocations): bitwise
      cycles 34909 -> 38078 (0.92x)
    linear 9x32 k64 bias, contracted, m1 (1 invocations): bitwise
    linear 9x32 k64 bias, contracted, m4 (1 invocations): bitwise
      cycles 35186 -> 31636 (1.11x)
    linear 7x29 k63 bias, contracted, m1 (1 invocations): bitwise
    linear 7x29 k63 bias, contracted, m4 (1 invocations): bitwise
      cycles 56362 -> 51823 (1.09x)
    bmm 1x9x32 k64, contracted, m1 (1 invocations): bitwise
    bmm 1x9x32 k64, contracted, m4 (1 invocations): bitwise
      cycles 24209 -> 18333 (1.32x)
    bmm 3x7x29 k63, contracted, m1 (1 invocations): bitwise
    bmm 3x7x29 k63, contracted, m4 (1 invocations): bitwise
      cycles 161201 -> 148442 (1.09x) |}]

(* The matrix products as generated RVV functions against the planned
   Zve32f loops, both under the contracted policy: bitwise against its
   oracle, and the cycles each takes. *)
let%expect_test "generated contractions against planned loops" =
  List.iter
    (fun (name, g) ->
      let numerics = Ssa_ir.Ssa_numerics.Fp32_contracted
      and oracle = T.Fp32_contracted in
      let lc, _ =
        measure ~numerics ~oracle Ssa_ir.Ssa_target.zve32f128 (name ^ ", loops")
          g
      in
      let cycles = ref 0L in
      let on_run (o : C.Run.t) = cycles := Int64.add !cycles o.C.Run.mcycle in
      T.check ~route:(C.route ~on_run config) ~contractions:true
        ~pipeline:
          (Ssa_backends.Pipeline.Planned
             { numerics; target = Ssa_ir.Ssa_target.zve32f128 })
        ~oracle ~constant:T.positive ~calls:1 (name ^ ", generated") g;
      let rate c =
        2. *. float_of_int (T.contraction_macs g) /. Int64.to_float c
      in
      Fmt.pr "  cycles %Ld -> %Ld (%.2fx); %.3f -> %.3f ops/cycle@." lc !cycles
        (Int64.to_float lc /. Int64.to_float !cycles)
        (rate lc) (rate !cycles))
    T.contraction_fixtures;
  [%expect
    {|
    linear 9x32 k64, loops (1 invocations): bitwise
    linear 9x32 k64, generated (1 invocations): bitwise
      cycles 34909 -> 9280 (3.76x); 1.056 -> 3.972 ops/cycle
    linear 9x32 k64 bias, loops (1 invocations): bitwise
    linear 9x32 k64 bias, generated (1 invocations): bitwise
      cycles 35186 -> 9605 (3.66x); 1.048 -> 3.838 ops/cycle
    linear 7x29 k63 bias, loops (1 invocations): bitwise
    linear 7x29 k63 bias, generated (1 invocations): bitwise
      cycles 56362 -> 10033 (5.62x); 0.454 -> 2.549 ops/cycle
    bmm 1x9x32 k64, loops (1 invocations): bitwise
    bmm 1x9x32 k64, generated (1 invocations): bitwise
      cycles 24209 -> 4599 (5.26x); 1.523 -> 8.016 ops/cycle
    bmm 3x7x29 k63, loops (1 invocations): bitwise
    bmm 3x7x29 k63, generated (1 invocations): bitwise
      cycles 161201 -> 15203 (10.60x); 0.476 -> 5.047 ops/cycle |}]
