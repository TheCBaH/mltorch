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
    (T.vector_fixtures @ T.transformer_fixtures @ T.transformer_bf16_fixtures
   @ Machine_model_test.Transformer_fixtures.bf16_out);
  [%expect
    {|
    relu then add over C=32, scalar (2 invocations): bitwise
    relu then add over C=32, zve32f128 (2 invocations): bitwise
      cycles 7492 -> 1601 (4.68x), instructions 4564 -> 1141 (4.00x)
    softmax over H, W=32, scalar (1 invocations): bitwise
    softmax over H, W=32, zve32f128 (1 invocations): bitwise
      cycles 213975 -> 213975 (1.00x), instructions 120886 -> 120886 (1.00x)
    layer_norm over H, W=32, scalar (1 invocations): bitwise
    layer_norm over H, W=32, zve32f128 (1 invocations): bitwise
      cycles 50380 -> 50380 (1.00x), instructions 27189 -> 27189 (1.00x)
    conv, batch norm, relu to C=32, scalar (3 invocations): bitwise
    conv, batch norm, relu to C=32, zve32f128 (3 invocations): bitwise
      cycles 183545 -> 23709 (7.74x), instructions 137969 -> 20356 (6.78x)
    residual add 2x160, scalar (1 invocations): bitwise
    residual add 2x160, zve32f128 (1 invocations): bitwise
      cycles 5850 -> 868 (6.74x), instructions 3594 -> 689 (5.22x)
    residual add 1x67, scalar (1 invocations): bitwise
    residual add 1x67, zve32f128 (1 invocations): bitwise
      cycles 1271 -> 302 (4.21x), instructions 776 -> 214 (3.63x)
    residual add 1x33, scalar (1 invocations): bitwise
    residual add 1x33, zve32f128 (1 invocations): bitwise
      cycles 659 -> 186 (3.54x), instructions 402 -> 127 (3.17x)
    rms norm 2x160, scalar (1 invocations): bitwise
    rms norm 2x160, zve32f128 (1 invocations): bitwise
      cycles 11824 -> 3599 (3.29x), instructions 6564 -> 1634 (4.02x)
    rms norm 1x67, scalar (1 invocations): bitwise
    rms norm 1x67, zve32f128 (1 invocations): bitwise
      cycles 2600 -> 986 (2.64x), instructions 1432 -> 474 (3.02x)
    rms norm 1x33, scalar (1 invocations): bitwise
    rms norm 1x33, zve32f128 (1 invocations): bitwise
      cycles 1376 -> 591 (2.33x), instructions 752 -> 289 (2.60x)
    tanh gelu mul 2x160, scalar (2 invocations): bitwise
    tanh gelu mul 2x160, zve32f128 (2 invocations): bitwise
      cycles 81207 -> 21755 (3.73x), instructions 43855 -> 11220 (3.91x)
    tanh gelu mul 1x67, scalar (2 invocations): bitwise
    tanh gelu mul 1x67, zve32f128 (2 invocations): bitwise
      cycles 17102 -> 5299 (3.23x), instructions 9242 -> 2781 (3.32x)
    tanh gelu mul 1x33, scalar (2 invocations): bitwise
    tanh gelu mul 1x33, zve32f128 (2 invocations): bitwise
      cycles 8489 -> 2643 (3.21x), instructions 4599 -> 1397 (3.29x)
    residual add bf16 2x160, scalar (1 invocations): bitwise
    residual add bf16 2x160, zve32f128 (1 invocations): bitwise
      cycles 7777 -> 1490 (5.22x), instructions 5199 -> 1339 (3.88x)
    residual add bf16 1x67, scalar (1 invocations): bitwise
    residual add bf16 1x67, zve32f128 (1 invocations): bitwise
      cycles 1673 -> 446 (3.75x), instructions 1111 -> 361 (3.08x)
    residual add bf16 1x33, scalar (1 invocations): bitwise
    residual add bf16 1x33, zve32f128 (1 invocations): bitwise
      cycles 857 -> 256 (3.35x), instructions 567 -> 199 (2.85x)
    rms norm bf16 2x160, scalar (1 invocations): bitwise
    rms norm bf16 2x160, zve32f128 (1 invocations): bitwise
      cycles 14068 -> 4807 (2.93x), instructions 8806 -> 2838 (3.10x)
    rms norm bf16 1x67, scalar (1 invocations): bitwise
    rms norm bf16 1x67, zve32f128 (1 invocations): bitwise
      cycles 3069 -> 1240 (2.48x), instructions 1901 -> 731 (2.60x)
    rms norm bf16 1x33, scalar (1 invocations): bitwise
    rms norm bf16 1x33, zve32f128 (1 invocations): bitwise
      cycles 1607 -> 708 (2.27x), instructions 983 -> 411 (2.39x)
    tanh gelu mul bf16 2x160, scalar (2 invocations): bitwise
    tanh gelu mul bf16 2x160, zve32f128 (2 invocations): bitwise
      cycles 83147 -> 23704 (3.51x), instructions 45790 -> 13614 (3.36x)
    tanh gelu mul bf16 1x67, scalar (2 invocations): bitwise
    tanh gelu mul bf16 1x67, zve32f128 (2 invocations): bitwise
      cycles 17504 -> 5714 (3.06x), instructions 9644 -> 3279 (2.94x)
    tanh gelu mul bf16 1x33, scalar (2 invocations): bitwise
    tanh gelu mul bf16 1x33, zve32f128 (2 invocations): bitwise
      cycles 8687 -> 2852 (3.05x), instructions 4797 -> 1646 (2.91x)
    residual add bf16->bf16 2x160, scalar (2 invocations): bitwise
    residual add bf16->bf16 2x160, zve32f128 (2 invocations): bitwise
      cycles 16825 -> 3737 (4.50x), instructions 14548 -> 3137 (4.64x)
    residual add bf16->bf16 1x67, scalar (2 invocations): bitwise
    residual add bf16->bf16 1x67, zve32f128 (2 invocations): bitwise
      cycles 3611 -> 1048 (3.45x), instructions 3092 -> 849 (3.64x)
    residual add bf16->bf16 1x33, scalar (2 invocations): bitwise
    residual add bf16->bf16 1x33, zve32f128 (2 invocations): bitwise
      cycles 1843 -> 582 (3.17x), instructions 1562 -> 453 (3.45x)
    rms norm bf16->bf16 2x160, scalar (2 invocations): bitwise
    rms norm bf16->bf16 2x160, zve32f128 (2 invocations): bitwise
      cycles 23116 -> 7054 (3.28x), instructions 18155 -> 4636 (3.92x)
    rms norm bf16->bf16 1x67, scalar (2 invocations): bitwise
    rms norm bf16->bf16 1x67, zve32f128 (2 invocations): bitwise
      cycles 5007 -> 1842 (2.72x), instructions 3882 -> 1219 (3.18x)
    rms norm bf16->bf16 1x33, scalar (2 invocations): bitwise
    rms norm bf16->bf16 1x33, zve32f128 (2 invocations): bitwise
      cycles 2593 -> 1034 (2.51x), instructions 1978 -> 665 (2.97x)
    tanh gelu mul bf16->bf16 2x160, scalar (3 invocations): bitwise
    tanh gelu mul bf16->bf16 2x160, zve32f128 (3 invocations): bitwise
      cycles 92195 -> 25951 (3.55x), instructions 55139 -> 15412 (3.58x)
    tanh gelu mul bf16->bf16 1x67, scalar (3 invocations): bitwise
    tanh gelu mul bf16->bf16 1x67, zve32f128 (3 invocations): bitwise
      cycles 19442 -> 6316 (3.08x), instructions 11625 -> 3767 (3.09x)
    tanh gelu mul bf16->bf16 1x33, scalar (3 invocations): bitwise
    tanh gelu mul bf16->bf16 1x33, zve32f128 (3 invocations): bitwise
      cycles 9673 -> 3178 (3.04x), instructions 5792 -> 1900 (3.05x) |}]

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
      cycles 195948 -> 38453 (5.10x), instructions 89606 -> 30965 (2.89x); 0.188 -> 0.959 ops/cycle
    linear 9x32 k64 bias, scalar (1 invocations): bitwise
    linear 9x32 k64 bias, zve32f128 (1 invocations): bitwise
      cycles 197694 -> 38684 (5.11x), instructions 90049 -> 31092 (2.90x); 0.186 -> 0.953 ops/cycle
    linear 7x29 k63 bias, scalar (1 invocations): bitwise
    linear 7x29 k63 bias, zve32f128 (1 invocations): bitwise
      cycles 140606 -> 76258 (1.84x), instructions 65021 -> 41616 (1.56x); 0.182 -> 0.335 ops/cycle
    bmm 1x9x32 k64, scalar (1 invocations): bitwise
    bmm 1x9x32 k64, zve32f128 (1 invocations): bitwise
      cycles 199836 -> 27479 (7.27x), instructions 93854 -> 25627 (3.66x); 0.184 -> 1.342 ops/cycle
    bmm 3x7x29 k63, scalar (1 invocations): bitwise
    bmm 3x7x29 k63, zve32f128 (1 invocations): bitwise
      cycles 427351 -> 235926 (1.81x), instructions 203759 -> 129132 (1.58x); 0.180 -> 0.325 ops/cycle |}]

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
      cycles 38453 -> 34871 (1.10x); 0.959 -> 1.057 ops/cycle
    linear 9x32 k64 bias, ordered (1 invocations): bitwise
    linear 9x32 k64 bias, contracted (1 invocations): bitwise
      cycles 38684 -> 35104 (1.10x); 0.953 -> 1.050 ops/cycle
    linear 7x29 k63 bias, ordered (1 invocations): bitwise
    linear 7x29 k63 bias, contracted (1 invocations): bitwise
      cycles 76258 -> 55763 (1.37x); 0.335 -> 0.459 ops/cycle
    bmm 1x9x32 k64, ordered (1 invocations): bitwise
    bmm 1x9x32 k64, contracted (1 invocations): bitwise
      cycles 27479 -> 24141 (1.14x); 1.342 -> 1.527 ops/cycle
    bmm 3x7x29 k63, ordered (1 invocations): bitwise
    bmm 3x7x29 k63, contracted (1 invocations): bitwise
      cycles 235926 -> 160255 (1.47x); 0.325 -> 0.479 ops/cycle |}]

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
      cycles 1601 -> 846 (1.89x)
    conv, batch norm, relu to C=32, ordered, m1 (3 invocations): bitwise
    conv, batch norm, relu to C=32, ordered, m4 (3 invocations): bitwise
      cycles 23709 -> 20821 (1.14x)
    linear 9x32 k64, ordered, m1 (1 invocations): bitwise
    linear 9x32 k64, ordered, m4 (1 invocations): bitwise
      cycles 38453 -> 31620 (1.22x)
    linear 9x32 k64 bias, ordered, m1 (1 invocations): bitwise
    linear 9x32 k64 bias, ordered, m4 (1 invocations): bitwise
      cycles 38684 -> 25255 (1.53x)
    linear 7x29 k63 bias, ordered, m1 (1 invocations): bitwise
    linear 7x29 k63 bias, ordered, m4 (1 invocations): bitwise
      cycles 76258 -> 54638 (1.40x)
    bmm 1x9x32 k64, ordered, m1 (1 invocations): bitwise
    bmm 1x9x32 k64, ordered, m4 (1 invocations): bitwise
      cycles 27479 -> 16209 (1.70x)
    bmm 3x7x29 k63, ordered, m1 (1 invocations): bitwise
    bmm 3x7x29 k63, ordered, m4 (1 invocations): bitwise
      cycles 235926 -> 165991 (1.42x)
    relu then add over C=32, contracted, m1 (2 invocations): bitwise
    relu then add over C=32, contracted, m4 (2 invocations): bitwise
      cycles 1601 -> 846 (1.89x)
    conv, batch norm, relu to C=32, contracted, m1 (3 invocations): bitwise
    conv, batch norm, relu to C=32, contracted, m4 (3 invocations): bitwise
      cycles 21837 -> 19768 (1.10x)
    linear 9x32 k64, contracted, m1 (1 invocations): bitwise
    linear 9x32 k64, contracted, m4 (1 invocations): bitwise
      cycles 34871 -> 38014 (0.92x)
    linear 9x32 k64 bias, contracted, m1 (1 invocations): bitwise
    linear 9x32 k64 bias, contracted, m4 (1 invocations): bitwise
      cycles 35104 -> 31601 (1.11x)
    linear 7x29 k63 bias, contracted, m1 (1 invocations): bitwise
    linear 7x29 k63 bias, contracted, m4 (1 invocations): bitwise
      cycles 55763 -> 51180 (1.09x)
    bmm 1x9x32 k64, contracted, m1 (1 invocations): bitwise
    bmm 1x9x32 k64, contracted, m4 (1 invocations): bitwise
      cycles 24141 -> 18261 (1.32x)
    bmm 3x7x29 k63, contracted, m1 (1 invocations): bitwise
    bmm 3x7x29 k63, contracted, m4 (1 invocations): bitwise
      cycles 160255 -> 147484 (1.09x) |}]

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
      cycles 34871 -> 9230 (3.78x); 1.057 -> 3.994 ops/cycle
    linear 9x32 k64 bias, loops (1 invocations): bitwise
    linear 9x32 k64 bias, generated (1 invocations): bitwise
      cycles 35104 -> 9442 (3.72x); 1.050 -> 3.904 ops/cycle
    linear 7x29 k63 bias, loops (1 invocations): bitwise
    linear 7x29 k63 bias, generated (1 invocations): bitwise
      cycles 55763 -> 9507 (5.87x); 0.459 -> 2.690 ops/cycle
    bmm 1x9x32 k64, loops (1 invocations): bitwise
    bmm 1x9x32 k64, generated (1 invocations): bitwise
      cycles 24141 -> 4561 (5.29x); 1.527 -> 8.082 ops/cycle
    bmm 3x7x29 k63, loops (1 invocations): bitwise
    bmm 3x7x29 k63, generated (1 invocations): bitwise
      cycles 160255 -> 15102 (10.61x); 0.479 -> 5.081 ops/cycle |}]

(* Transformer operations with register groups: LMUL 1 against LMUL 4, with
   binary32 and bfloat16 inputs (sixteen bfloat16 lanes are half a group,
   loaded at LMUL 2 and widened at once). *)
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
    (T.transformer_fixtures @ T.transformer_bf16_fixtures
   @ Machine_model_test.Transformer_fixtures.bf16_out);
  [%expect
    {|
    residual add 2x160, m1 (1 invocations): bitwise
    residual add 2x160, m4 (1 invocations): bitwise
      cycles 868 -> 448 (1.94x)
    residual add 1x67, m1 (1 invocations): bitwise
    residual add 1x67, m4 (1 invocations): bitwise
      cycles 302 -> 218 (1.39x)
    residual add 1x33, m1 (1 invocations): bitwise
    residual add 1x33, m4 (1 invocations): bitwise
      cycles 186 -> 144 (1.29x)
    rms norm 2x160, m1 (1 invocations): bitwise
    rms norm 2x160, m4 (1 invocations): bitwise
      cycles 3599 -> 2130 (1.69x)
    rms norm 1x67, m1 (1 invocations): bitwise
    rms norm 1x67, m4 (1 invocations): bitwise
      cycles 986 -> 685 (1.44x)
    rms norm 1x33, m1 (1 invocations): bitwise
    rms norm 1x33, m4 (1 invocations): bitwise
      cycles 591 -> 435 (1.36x)
    tanh gelu mul 2x160, m1 (2 invocations): bitwise
    tanh gelu mul 2x160, m4 (2 invocations): bitwise
      cycles 21755 -> 13889 (1.57x)
    tanh gelu mul 1x67, m1 (2 invocations): bitwise
    tanh gelu mul 1x67, m4 (2 invocations): bitwise
      cycles 5299 -> 3721 (1.42x)
    tanh gelu mul 1x33, m1 (2 invocations): bitwise
    tanh gelu mul 1x33, m4 (2 invocations): bitwise
      cycles 2643 -> 1851 (1.43x)
    residual add bf16 2x160, m1 (1 invocations): bitwise
    residual add bf16 2x160, m4 (1 invocations): bitwise
      cycles 1490 -> 690 (2.16x)
    residual add bf16 1x67, m1 (1 invocations): bitwise
    residual add bf16 1x67, m4 (1 invocations): bitwise
      cycles 446 -> 286 (1.56x)
    residual add bf16 1x33, m1 (1 invocations): bitwise
    residual add bf16 1x33, m4 (1 invocations): bitwise
      cycles 256 -> 176 (1.45x)
    rms norm bf16 2x160, m1 (1 invocations): bitwise
    rms norm bf16 2x160, m4 (1 invocations): bitwise
      cycles 4807 -> 2497 (1.93x)
    rms norm bf16 1x67, m1 (1 invocations): bitwise
    rms norm bf16 1x67, m4 (1 invocations): bitwise
      cycles 1240 -> 777 (1.60x)
    rms norm bf16 1x33, m1 (1 invocations): bitwise
    rms norm bf16 1x33, m4 (1 invocations): bitwise
      cycles 708 -> 475 (1.49x)
    tanh gelu mul bf16 2x160, m1 (2 invocations): bitwise
    tanh gelu mul bf16 2x160, m4 (2 invocations): bitwise
      cycles 23704 -> 14129 (1.68x)
    tanh gelu mul bf16 1x67, m1 (2 invocations): bitwise
    tanh gelu mul bf16 1x67, m4 (2 invocations): bitwise
      cycles 5714 -> 3787 (1.51x)
    tanh gelu mul bf16 1x33, m1 (2 invocations): bitwise
    tanh gelu mul bf16 1x33, m4 (2 invocations): bitwise
      cycles 2852 -> 1881 (1.52x)
    residual add bf16->bf16 2x160, m1 (2 invocations): bitwise
    residual add bf16->bf16 2x160, m4 (2 invocations): bitwise
      cycles 3737 -> 1916 (1.95x)
    residual add bf16->bf16 1x67, m1 (2 invocations): bitwise
    residual add bf16->bf16 1x67, m4 (2 invocations): bitwise
      cycles 1048 -> 686 (1.53x)
    residual add bf16->bf16 1x33, m1 (2 invocations): bitwise
    residual add bf16->bf16 1x33, m4 (2 invocations): bitwise
      cycles 582 -> 402 (1.45x)
    rms norm bf16->bf16 2x160, m1 (2 invocations): bitwise
    rms norm bf16->bf16 2x160, m4 (2 invocations): bitwise
      cycles 7054 -> 3723 (1.89x)
    rms norm bf16->bf16 1x67, m1 (2 invocations): bitwise
    rms norm bf16->bf16 1x67, m4 (2 invocations): bitwise
      cycles 1842 -> 1177 (1.56x)
    rms norm bf16->bf16 1x33, m1 (2 invocations): bitwise
    rms norm bf16->bf16 1x33, m4 (2 invocations): bitwise
      cycles 1034 -> 701 (1.48x)
    tanh gelu mul bf16->bf16 2x160, m1 (3 invocations): bitwise
    tanh gelu mul bf16->bf16 2x160, m4 (3 invocations): bitwise
      cycles 25951 -> 15355 (1.69x)
    tanh gelu mul bf16->bf16 1x67, m1 (3 invocations): bitwise
    tanh gelu mul bf16->bf16 1x67, m4 (3 invocations): bitwise
      cycles 6316 -> 4187 (1.51x)
    tanh gelu mul bf16->bf16 1x33, m1 (3 invocations): bitwise
    tanh gelu mul bf16->bf16 1x33, m4 (3 invocations): bitwise
      cycles 3178 -> 2107 (1.51x) |}]

(* Thirty-two-lane plans in groups of eight registers (LMUL 8) against
   sixteen-lane ones at LMUL 4: the kernels whose forms need at most two
   vector groups at once (an elementwise add, RMS norm's loops). *)
let%expect_test "groups of eight" =
  let grouped = Machine_rivet_riscv32_mir.Rv32_mir_route.Registers.Grouped in
  let grouped8 = Machine_rivet_riscv32_mir.Rv32_mir_route.Registers.Grouped8 in
  List.iter
    (fun (name, g) ->
      if
        String.starts_with ~prefix:"residual add" name
        || String.starts_with ~prefix:"rms norm" name
      then
        let c4, _ =
          measure ~registers:grouped Ssa_ir.Ssa_target.zve32f128 (name ^ ", m4")
            g
        in
        let c8, _ =
          measure ~registers:grouped8 Ssa_ir.Ssa_target.zve32f128_x32
            (name ^ ", m8") g
        in
        Fmt.pr "  cycles %Ld -> %Ld (%.2fx)@." c4 c8
          (Int64.to_float c4 /. Int64.to_float c8))
    T.transformer_fixtures;
  [%expect
    {|
    residual add 2x160, m4 (1 invocations): bitwise
    residual add 2x160, m8 (1 invocations): bitwise
      cycles 448 -> 398 (1.13x)
    residual add 1x67, m4 (1 invocations): bitwise
    residual add 1x67, m8 (1 invocations): bitwise
      cycles 218 -> 208 (1.05x)
    residual add 1x33, m4 (1 invocations): bitwise
    residual add 1x33, m8 (1 invocations): bitwise
      cycles 144 -> 139 (1.04x)
    rms norm 2x160, m4 (1 invocations): bitwise
    rms norm 2x160, m8 (1 invocations): bitwise
      cycles 2130 -> 2098 (1.02x)
    rms norm 1x67, m4 (1 invocations): bitwise
    rms norm 1x67, m8 (1 invocations): bitwise
      cycles 685 -> 678 (1.01x)
    rms norm 1x33, m4 (1 invocations): bitwise
    rms norm 1x33, m8 (1 invocations): bitwise
      cycles 435 -> 431 (1.01x) |}]

(* RMS norm under the relaxed policy in groups of four: its scheduled sum's
   horizontal combine extracts lanes of a sixteen-lane group. The oracle is
   the policy's own schedule on the generic Machine IR route, bitwise. *)
let%expect_test "relaxed sums, register groups" =
  let grouped = Machine_rivet_riscv32_mir.Rv32_mir_route.Registers.Grouped in
  let numerics = Ssa_ir.Ssa_numerics.Simd_fp32_relaxed
  and target = Ssa_ir.Ssa_target.zve32f128 in
  let oracle =
    Machine_model_test.Plan_oracle.make
      ~pipeline:(Ssa_backends.Pipeline.Planned { numerics; target })
  in
  List.iter
    (fun (name, g) ->
      if String.starts_with ~prefix:"rms norm" name then
        let c, _ =
          measure ~numerics ~oracle ~registers:grouped target (name ^ ", m4") g
        in
        Fmt.pr "  cycles %Ld@." c)
    T.transformer_fixtures;
  [%expect
    {|
    rms norm 2x160, m4 (1 invocations): bitwise
      cycles 1781
    rms norm 1x67, m4 (1 invocations): bitwise
      cycles 744
    rms norm 1x33, m4 (1 invocations): bitwise
      cycles 785 |}]
