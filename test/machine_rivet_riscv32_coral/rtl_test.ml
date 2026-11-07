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
    ?registers ?sequence target name g =
  let cycles = ref 0L and retired = ref 0L in
  let on_run (o : C.Run.t) =
    cycles := Int64.add !cycles o.C.Run.mcycle;
    retired := Int64.add !retired o.C.Run.minstret
  in
  T.check ?sequence
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
    (T.vector_fixtures @ T.transformer_fixtures @ T.transformer_bf16_fixtures);
  [%expect
    {|
    relu then add over C=32, scalar (2 invocations): bitwise
    relu then add over C=32, zve32f128 (2 invocations): bitwise
      cycles 7492 -> 1603 (4.67x), instructions 4564 -> 1141 (4.00x)
    softmax over H, W=32, scalar (1 invocations): bitwise
    softmax over H, W=32, zve32f128 (1 invocations): bitwise
      cycles 474689 -> 474689 (1.00x), instructions 224279 -> 224279 (1.00x)
    layer_norm over H, W=32, scalar (1 invocations): bitwise
    layer_norm over H, W=32, zve32f128 (1 invocations): bitwise
      cycles 50380 -> 50380 (1.00x), instructions 27189 -> 27189 (1.00x)
    conv, batch norm, relu to C=32, scalar (3 invocations): bitwise
    conv, batch norm, relu to C=32, zve32f128 (3 invocations): bitwise
      cycles 183545 -> 23181 (7.92x), instructions 137969 -> 20356 (6.78x)
    residual add 2x160, scalar (1 invocations): bitwise
    residual add 2x160, zve32f128 (1 invocations): bitwise
      cycles 5850 -> 869 (6.73x), instructions 3594 -> 689 (5.22x)
    residual add 1x67, scalar (1 invocations): bitwise
    residual add 1x67, zve32f128 (1 invocations): bitwise
      cycles 1271 -> 303 (4.19x), instructions 776 -> 214 (3.63x)
    residual add 1x33, scalar (1 invocations): bitwise
    residual add 1x33, zve32f128 (1 invocations): bitwise
      cycles 659 -> 187 (3.52x), instructions 402 -> 127 (3.17x)
    rms norm 2x160, scalar (1 invocations): bitwise
    rms norm 2x160, zve32f128 (1 invocations): bitwise
      cycles 11824 -> 6014 (1.97x), instructions 6564 -> 3471 (1.89x)
    rms norm 1x67, scalar (1 invocations): bitwise
    rms norm 1x67, zve32f128 (1 invocations): bitwise
      cycles 2600 -> 1467 (1.77x), instructions 1432 -> 833 (1.72x)
    rms norm 1x33, scalar (1 invocations): bitwise
    rms norm 1x33, zve32f128 (1 invocations): bitwise
      cycles 1376 -> 828 (1.66x), instructions 752 -> 464 (1.62x)
    tanh gelu mul 2x160, scalar (2 invocations): bitwise
    tanh gelu mul 2x160, zve32f128 (2 invocations): bitwise
      cycles 167876 -> 54116 (3.10x), instructions 78139 -> 26040 (3.00x)
    tanh gelu mul 1x67, scalar (2 invocations): bitwise
    tanh gelu mul 1x67, zve32f128 (2 invocations): bitwise
      cycles 35260 -> 12604 (2.80x), instructions 16430 -> 6075 (2.70x)
    tanh gelu mul 1x33, scalar (2 invocations): bitwise
    tanh gelu mul 1x33, zve32f128 (2 invocations): bitwise
      cycles 17444 -> 6171 (2.83x), instructions 8142 -> 2998 (2.72x)
    residual add bf16 2x160, scalar (1 invocations): bitwise
    residual add bf16 2x160, zve32f128 (1 invocations): bitwise
      cycles 7777 -> 1451 (5.36x), instructions 5199 -> 1339 (3.88x)
    residual add bf16 1x67, scalar (1 invocations): bitwise
    residual add bf16 1x67, zve32f128 (1 invocations): bitwise
      cycles 1673 -> 439 (3.81x), instructions 1111 -> 361 (3.08x)
    residual add bf16 1x33, scalar (1 invocations): bitwise
    residual add bf16 1x33, zve32f128 (1 invocations): bitwise
      cycles 857 -> 253 (3.39x), instructions 567 -> 199 (2.85x)
    rms norm bf16 2x160, scalar (1 invocations): bitwise
    rms norm bf16 2x160, zve32f128 (1 invocations): bitwise
      cycles 14068 -> 7261 (1.94x), instructions 8806 -> 4738 (1.86x)
    rms norm bf16 1x67, scalar (1 invocations): bitwise
    rms norm bf16 1x67, zve32f128 (1 invocations): bitwise
      cycles 3069 -> 1732 (1.77x), instructions 1901 -> 1104 (1.72x)
    rms norm bf16 1x33, scalar (1 invocations): bitwise
    rms norm bf16 1x33, zve32f128 (1 invocations): bitwise
      cycles 1607 -> 954 (1.68x), instructions 983 -> 594 (1.65x)
    tanh gelu mul bf16 2x160, scalar (2 invocations): bitwise
    tanh gelu mul bf16 2x160, zve32f128 (2 invocations): bitwise
      cycles 169809 -> 56063 (3.03x), instructions 80071 -> 28434 (2.82x)
    tanh gelu mul bf16 1x67, scalar (2 invocations): bitwise
    tanh gelu mul bf16 1x67, zve32f128 (2 invocations): bitwise
      cycles 35662 -> 13021 (2.74x), instructions 16832 -> 6577 (2.56x)
    tanh gelu mul bf16 1x33, scalar (2 invocations): bitwise
    tanh gelu mul bf16 1x33, zve32f128 (2 invocations): bitwise
      cycles 17642 -> 6380 (2.77x), instructions 8340 -> 3249 (2.57x) |}]

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
      cycles 195948 -> 38447 (5.10x), instructions 89606 -> 30965 (2.89x); 0.188 -> 0.959 ops/cycle
    linear 9x32 k64 bias, scalar (1 invocations): bitwise
    linear 9x32 k64 bias, zve32f128 (1 invocations): bitwise
      cycles 197694 -> 38670 (5.11x), instructions 90049 -> 31092 (2.90x); 0.186 -> 0.953 ops/cycle
    linear 7x29 k63 bias, scalar (1 invocations): bitwise
    linear 7x29 k63 bias, zve32f128 (1 invocations): bitwise
      cycles 140606 -> 80690 (1.74x), instructions 65021 -> 45123 (1.44x); 0.182 -> 0.317 ops/cycle
    bmm 1x9x32 k64, scalar (1 invocations): bitwise
    bmm 1x9x32 k64, zve32f128 (1 invocations): bitwise
      cycles 199836 -> 26963 (7.41x), instructions 93854 -> 25627 (3.66x); 0.184 -> 1.367 ops/cycle
    bmm 3x7x29 k63, scalar (1 invocations): bitwise
    bmm 3x7x29 k63, zve32f128 (1 invocations): bitwise
      cycles 427351 -> 238354 (1.79x), instructions 203759 -> 133880 (1.52x); 0.180 -> 0.322 ops/cycle |}]

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
      cycles 38447 -> 34845 (1.10x); 0.959 -> 1.058 ops/cycle
    linear 9x32 k64 bias, ordered (1 invocations): bitwise
    linear 9x32 k64 bias, contracted (1 invocations): bitwise
      cycles 38670 -> 35090 (1.10x); 0.953 -> 1.051 ops/cycle
    linear 7x29 k63 bias, ordered (1 invocations): bitwise
    linear 7x29 k63 bias, contracted (1 invocations): bitwise
      cycles 80690 -> 55749 (1.45x); 0.317 -> 0.459 ops/cycle
    bmm 1x9x32 k64, ordered (1 invocations): bitwise
    bmm 1x9x32 k64, contracted (1 invocations): bitwise
      cycles 26963 -> 24129 (1.12x); 1.367 -> 1.528 ops/cycle
    bmm 3x7x29 k63, ordered (1 invocations): bitwise
    bmm 3x7x29 k63, contracted (1 invocations): bitwise
      cycles 238354 -> 160213 (1.49x); 0.322 -> 0.479 ops/cycle |}]

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
      cycles 1603 -> 846 (1.89x)
    conv, batch norm, relu to C=32, ordered, m1 (3 invocations): bitwise
    conv, batch norm, relu to C=32, ordered, m4 (3 invocations): bitwise
      cycles 23181 -> 20821 (1.11x)
    linear 9x32 k64, ordered, m1 (1 invocations): bitwise
    linear 9x32 k64, ordered, m4 (1 invocations): bitwise
      cycles 38447 -> 31620 (1.22x)
    linear 9x32 k64 bias, ordered, m1 (1 invocations): bitwise
    linear 9x32 k64 bias, ordered, m4 (1 invocations): bitwise
      cycles 38670 -> 25255 (1.53x)
    linear 7x29 k63 bias, ordered, m1 (1 invocations): bitwise
    linear 7x29 k63 bias, ordered, m4 (1 invocations): bitwise
      cycles 80690 -> 75463 (1.07x)
    bmm 1x9x32 k64, ordered, m1 (1 invocations): bitwise
    bmm 1x9x32 k64, ordered, m4 (1 invocations): bitwise
      cycles 26963 -> 16209 (1.66x)
    bmm 3x7x29 k63, ordered, m1 (1 invocations): bitwise
    bmm 3x7x29 k63, ordered, m4 (1 invocations): bitwise
      cycles 238354 -> 223021 (1.07x)
    relu then add over C=32, contracted, m1 (2 invocations): bitwise
    relu then add over C=32, contracted, m4 (2 invocations): bitwise
      cycles 1603 -> 846 (1.89x)
    conv, batch norm, relu to C=32, contracted, m1 (3 invocations): bitwise
    conv, batch norm, relu to C=32, contracted, m4 (3 invocations): bitwise
      cycles 21309 -> 19768 (1.08x)
    linear 9x32 k64, contracted, m1 (1 invocations): bitwise
    linear 9x32 k64, contracted, m4 (1 invocations): bitwise
      cycles 34845 -> 38014 (0.92x)
    linear 9x32 k64 bias, contracted, m1 (1 invocations): bitwise
    linear 9x32 k64 bias, contracted, m4 (1 invocations): bitwise
      cycles 35090 -> 31601 (1.11x)
    linear 7x29 k63 bias, contracted, m1 (1 invocations): bitwise
    linear 7x29 k63 bias, contracted, m4 (1 invocations): bitwise
      cycles 55749 -> 51180 (1.09x)
    bmm 1x9x32 k64, contracted, m1 (1 invocations): bitwise
    bmm 1x9x32 k64, contracted, m4 (1 invocations): bitwise
      cycles 24129 -> 18261 (1.32x)
    bmm 3x7x29 k63, contracted, m1 (1 invocations): bitwise
    bmm 3x7x29 k63, contracted, m4 (1 invocations): bitwise
      cycles 160213 -> 147484 (1.09x) |}]

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
      cycles 34845 -> 9229 (3.78x); 1.058 -> 3.994 ops/cycle
    linear 9x32 k64 bias, loops (1 invocations): bitwise
    linear 9x32 k64 bias, generated (1 invocations): bitwise
      cycles 35090 -> 9442 (3.72x); 1.051 -> 3.904 ops/cycle
    linear 7x29 k63 bias, loops (1 invocations): bitwise
    linear 7x29 k63 bias, generated (1 invocations): bitwise
      cycles 55749 -> 9505 (5.87x); 0.459 -> 2.691 ops/cycle
    bmm 1x9x32 k64, loops (1 invocations): bitwise
    bmm 1x9x32 k64, generated (1 invocations): bitwise
      cycles 24129 -> 4552 (5.30x); 1.528 -> 8.098 ops/cycle
    bmm 3x7x29 k63, loops (1 invocations): bitwise
    bmm 3x7x29 k63, generated (1 invocations): bitwise
      cycles 160213 -> 15096 (10.61x); 0.479 -> 5.083 ops/cycle |}]

(* Transformer binary32 operations with register groups: LMUL 1 against
   LMUL 4. The bfloat16 variants are left out: the grouped route refuses them
   (a 16-lane bfloat16 vector fills no whole number of registers). *)
let%expect_test "transformer operations, register groups" =
  let grouped = Machine_rivet_riscv32_mir.Rv32_mir_route.Registers.Grouped in
  List.iter
    (fun (name, g) ->
      let sc, _ = measure Ssa_ir.Ssa_target.zve32f128 (name ^ ", m1") g in
      let gc, _ =
        measure ~registers:grouped Ssa_ir.Ssa_target.zve32f128 (name ^ ", m4") g
      in
      Fmt.pr "  cycles %Ld -> %Ld (%.2fx)@." sc gc
        (Int64.to_float sc /. Int64.to_float gc))
    T.transformer_fixtures;
  [%expect
    {|
    residual add 2x160, m1 (1 invocations): bitwise
    residual add 2x160, m4 (1 invocations): bitwise
      cycles 869 -> 448 (1.94x)
    residual add 1x67, m1 (1 invocations): bitwise
    residual add 1x67, m4 (1 invocations): bitwise
      cycles 303 -> 218 (1.39x)
    residual add 1x33, m1 (1 invocations): bitwise
    residual add 1x33, m4 (1 invocations): bitwise
      cycles 187 -> 144 (1.30x)
    rms norm 2x160, m1 (1 invocations): bitwise
    rms norm 2x160, m4 (1 invocations): bitwise
      cycles 6014 -> 5671 (1.06x)
    rms norm 1x67, m1 (1 invocations): bitwise
    rms norm 1x67, m4 (1 invocations): bitwise
      cycles 1467 -> 1391 (1.05x)
    rms norm 1x33, m1 (1 invocations): bitwise
    rms norm 1x33, m4 (1 invocations): bitwise
      cycles 828 -> 785 (1.05x)
    tanh gelu mul 2x160, m1 (2 invocations): bitwise
    tanh gelu mul 2x160, m4 (2 invocations): bitwise
      cycles 54116 -> 35209 (1.54x)
    tanh gelu mul 1x67, m1 (2 invocations): bitwise
    tanh gelu mul 1x67, m4 (2 invocations): bitwise
      cycles 12604 -> 8881 (1.42x)
    tanh gelu mul 1x33, m1 (2 invocations): bitwise
    tanh gelu mul 1x33, m4 (2 invocations): bitwise
      cycles 6171 -> 4306 (1.43x) |}]

(* The fast binary32 exp ([Sequence.Fast]) in tanh GELU: the same bits,
   scalar, LMUL 1 and LMUL 4, cycles under the reference sequence and the fast
   one. *)
let%expect_test "tanh gelu, fast sequences" =
  let grouped = Machine_rivet_riscv32_mir.Rv32_mir_route.Registers.Grouped in
  List.iter
    (fun (name, g) ->
      if String.starts_with ~prefix:"tanh gelu" name then
        List.iter
          (fun (rname, target, registers) ->
            let rc, _ =
              measure ?registers target (Fmt.str "%s, %s" name rname) g
            in
            let fc, _ =
              measure ?registers ~sequence:Machine_lower.Mir_lower.Sequence.Fast
                target
                (Fmt.str "%s, %s, fast" name rname)
                g
            in
            Fmt.pr "  cycles %Ld -> %Ld (%.2fx)@." rc fc
              (Int64.to_float rc /. Int64.to_float fc))
          [
            ("scalar", Ssa_ir.Ssa_target.scalar, None);
            ("m1", Ssa_ir.Ssa_target.zve32f128, None);
            ("m4", Ssa_ir.Ssa_target.zve32f128, Some grouped);
          ])
    T.transformer_fixtures;
  [%expect
    {|
    tanh gelu mul 2x160, scalar (2 invocations): bitwise
    tanh gelu mul 2x160, scalar, fast (2 invocations): bitwise
      cycles 167876 -> 81207 (2.07x)
    tanh gelu mul 2x160, m1 (2 invocations): bitwise
    tanh gelu mul 2x160, m1, fast (2 invocations): bitwise
      cycles 54116 -> 21756 (2.49x)
    tanh gelu mul 2x160, m4 (2 invocations): bitwise
    tanh gelu mul 2x160, m4, fast (2 invocations): bitwise
      cycles 35209 -> 13889 (2.54x)
    tanh gelu mul 1x67, scalar (2 invocations): bitwise
    tanh gelu mul 1x67, scalar, fast (2 invocations): bitwise
      cycles 35260 -> 17102 (2.06x)
    tanh gelu mul 1x67, m1 (2 invocations): bitwise
    tanh gelu mul 1x67, m1, fast (2 invocations): bitwise
      cycles 12604 -> 5300 (2.38x)
    tanh gelu mul 1x67, m4 (2 invocations): bitwise
    tanh gelu mul 1x67, m4, fast (2 invocations): bitwise
      cycles 8881 -> 3721 (2.39x)
    tanh gelu mul 1x33, scalar (2 invocations): bitwise
    tanh gelu mul 1x33, scalar, fast (2 invocations): bitwise
      cycles 17444 -> 8489 (2.05x)
    tanh gelu mul 1x33, m1 (2 invocations): bitwise
    tanh gelu mul 1x33, m1, fast (2 invocations): bitwise
      cycles 6171 -> 2644 (2.33x)
    tanh gelu mul 1x33, m4 (2 invocations): bitwise
    tanh gelu mul 1x33, m4, fast (2 invocations): bitwise
      cycles 4306 -> 1851 (2.33x) |}]
