(* Whole binary32 sequences and grouped-register plans on RV32 processes under
   qemu, bitwise against the binary32 oracle. Split from [model_test.ml] to
   keep each partition under the inline-test timing threshold. *)

module T = Machine_model_test.Model_test
module R = Machine_rivet_riscv32_mir.Rv32_mir_route
open Model_pipelines

(* The binary32 sequences as RV32 processes, scalar and Zve32f at VLEN 128
   and 1024: the reference's bits. *)
let%expect_test "binary32 sequences under qemu" =
  let module S = Machine_model_test.Sequence_test in
  S.each ~route:(R.route ()) ~pipeline:fp32_scalar "scalar";
  List.iter
    (fun vlen ->
      S.each
        ~route:(R.route ~cpu:(cpu vlen) ())
        ~pipeline:zve32f
        (Fmt.str "zve32f vlen %d" vlen))
    [ 128; 1024 ];
  [%expect
    {|
    exp, scalar (1 invocations): bitwise
    exp, around each k step, scalar (1 invocations): bitwise
    erf, scalar (1 invocations): bitwise
    exact gelu, scalar (1 invocations): bitwise
    softmax over H, W=32, scalar (1 invocations): bitwise
    tanh gelu mul 1x67, scalar (2 invocations): bitwise
    tanh gelu mul bf16 1x33, scalar (2 invocations): bitwise
    exp, zve32f vlen 128 (1 invocations): bitwise
    exp, around each k step, zve32f vlen 128 (1 invocations): bitwise
    erf, zve32f vlen 128 (1 invocations): bitwise
    exact gelu, zve32f vlen 128 (1 invocations): bitwise
    softmax over H, W=32, zve32f vlen 128 (1 invocations): bitwise
    tanh gelu mul 1x67, zve32f vlen 128 (2 invocations): bitwise
    tanh gelu mul bf16 1x33, zve32f vlen 128 (2 invocations): bitwise
    exp, zve32f vlen 1024 (1 invocations): bitwise
    exp, around each k step, zve32f vlen 1024 (1 invocations): bitwise
    erf, zve32f vlen 1024 (1 invocations): bitwise
    exact gelu, zve32f vlen 1024 (1 invocations): bitwise
    softmax over H, W=32, zve32f vlen 1024 (1 invocations): bitwise
    tanh gelu mul 1x67, zve32f vlen 1024 (2 invocations): bitwise
    tanh gelu mul bf16 1x33, zve32f vlen 1024 (2 invocations): bitwise |}]

(* The relaxed policy's scheduled sums (RMSNorm's) under grouped registers,
   as processes at VLEN 128 and 1024: their horizontal combine extracts lanes
   of a sixteen-lane group. The oracle is the same plan on the generic Machine
   IR route, the policy's own schedule, bitwise. *)
let%expect_test "relaxed sums under grouped registers" =
  let pipeline =
    Ssa_backends.Pipeline.Planned
      {
        numerics = Ssa_ir.Ssa_numerics.Simd_fp32_relaxed;
        target = Ssa_ir.Ssa_target.zve32f128;
      }
  in
  let plan_oracle = Machine_model_test.Plan_oracle.make ~pipeline in
  List.iter
    (fun vlen ->
      List.iter
        (fun (name, g) ->
          if String.starts_with ~prefix:"rms norm" name then
            T.check
              ~route:(R.route ~cpu:(cpu vlen) ~registers:R.Registers.Grouped ())
              ~pipeline ~oracle:plan_oracle ~constant:T.positive
              (Fmt.str "vlen %d, %s" vlen name)
              g)
        T.transformer_fixtures)
    [ 128; 1024 ];
  [%expect
    {|
    vlen 128, rms norm 2x160 (1 invocations): bitwise, bitwise
    vlen 128, rms norm 1x67 (1 invocations): bitwise, bitwise
    vlen 128, rms norm 1x33 (1 invocations): bitwise, bitwise
    vlen 1024, rms norm 2x160 (1 invocations): bitwise, bitwise
    vlen 1024, rms norm 1x67 (1 invocations): bitwise, bitwise
    vlen 1024, rms norm 1x33 (1 invocations): bitwise, bitwise |}]

(* The bfloat16 fixtures under register groups at VLEN 128 and 1024: sixteen
   16-bit lanes are half a group, loaded at LMUL 2 and widened at once into a
   whole one, bitwise against the binary32 oracle. *)
let%expect_test "bfloat16 under register groups" =
  let pipeline =
    Ssa_backends.Pipeline.Planned
      {
        numerics = Ssa_ir.Ssa_numerics.Fp32_ordered;
        target = Ssa_ir.Ssa_target.zve32f128;
      }
  in
  List.iter
    (fun vlen ->
      List.iter
        (fun (name, g) ->
          T.check
            ~route:(R.route ~cpu:(cpu vlen) ~registers:R.Registers.Grouped ())
            ~pipeline ~oracle:T.Fp32 ~constant:T.positive ~calls:1
            (Fmt.str "vlen %d, %s" vlen name)
            g)
        T.transformer_bf16_fixtures)
    [ 128; 1024 ];
  [%expect
    {|
    vlen 128, residual add bf16 2x160 (1 invocations): bitwise
    vlen 128, residual add bf16 1x67 (1 invocations): bitwise
    vlen 128, residual add bf16 1x33 (1 invocations): bitwise
    vlen 128, rms norm bf16 2x160 (1 invocations): bitwise
    vlen 128, rms norm bf16 1x67 (1 invocations): bitwise
    vlen 128, rms norm bf16 1x33 (1 invocations): bitwise
    vlen 128, tanh gelu mul bf16 2x160 (2 invocations): bitwise
    vlen 128, tanh gelu mul bf16 1x67 (2 invocations): bitwise
    vlen 128, tanh gelu mul bf16 1x33 (2 invocations): bitwise
    vlen 1024, residual add bf16 2x160 (1 invocations): bitwise
    vlen 1024, residual add bf16 1x67 (1 invocations): bitwise
    vlen 1024, residual add bf16 1x33 (1 invocations): bitwise
    vlen 1024, rms norm bf16 2x160 (1 invocations): bitwise
    vlen 1024, rms norm bf16 1x67 (1 invocations): bitwise
    vlen 1024, rms norm bf16 1x33 (1 invocations): bitwise
    vlen 1024, tanh gelu mul bf16 2x160 (2 invocations): bitwise
    vlen 1024, tanh gelu mul bf16 1x67 (2 invocations): bitwise
    vlen 1024, tanh gelu mul bf16 1x33 (2 invocations): bitwise |}]

(* The strict sums as ordered vector reductions ([vfredosum.vs]): one a
   register at LMUL 1, one a group at LMUL 4, at VLEN 128 and 1024, on
   adversarial operands (cancellation, signed zeros, infinities, NaN,
   subnormals) and on operands that round at every addition, bitwise against
   the binary32 oracle. Adding the lanes in reverse order in the schedule
   turns the rounding runs red. *)
let%expect_test "ordered vector sums under qemu" =
  let pipeline =
    Ssa_backends.Pipeline.Planned
      {
        numerics = Ssa_ir.Ssa_numerics.Fp32_ordered;
        target = Ssa_ir.Ssa_target.zve32f128;
      }
  in
  let sums =
    List.filter
      (fun (name, _) ->
        List.exists
          (fun p -> String.starts_with ~prefix:p name)
          [ "rms norm 2x160"; "rms norm 1x67"; "softmax"; "layer_norm" ])
      (T.vector_fixtures @ T.transformer_fixtures)
  in
  List.iter
    (fun (vlen, registers) ->
      List.iter
        (fun (name, g) ->
          let label inputs =
            Fmt.str "vlen %d, %s, %s, %s" vlen
              (match registers with
              | R.Registers.Grouped -> "m4"
              | R.Registers.Grouped8 -> "m8"
              | R.Registers.Single -> "m1")
              name inputs
          in
          T.check
            ~route:(R.route ~cpu:(cpu vlen) ~registers ())
            ~pipeline ~oracle:T.Fp32 ~same:T.same_portable
            ~input:T.adversarial_input
            ~constant:(fun sg -> T.adversarial ~salt:3 sg)
            ~calls:1 (label "adversarial") g;
          (* rounding at every addition: an order the fold does not take
             shows here, where the specials above saturate *)
          T.check
            ~route:(R.route ~cpu:(cpu vlen) ~registers ())
            ~pipeline ~oracle:T.Fp32
            ~input:(fun ~salt _ sg ->
              Tensor.materialize sg.Tensor_sig.shape (fun c ->
                  let k = (Vec6.offset sg.Tensor_sig.shape c :> int) + salt in
                  F32_math.round32 (1. /. float_of_int ((k mod 13) + 3))))
            ~constant:T.positive ~calls:1 (label "rounding") g)
        sums)
    [
      (128, R.Registers.Single);
      (128, R.Registers.Grouped);
      (1024, R.Registers.Single);
      (1024, R.Registers.Grouped);
    ];
  [%expect
    {|
    vlen 128, m1, softmax over H, W=32, adversarial (1 invocations): bitwise
    vlen 128, m1, softmax over H, W=32, rounding (1 invocations): bitwise
    vlen 128, m1, layer_norm over H, W=32, adversarial (1 invocations): bitwise
    vlen 128, m1, layer_norm over H, W=32, rounding (1 invocations): bitwise
    vlen 128, m1, rms norm 2x160, adversarial (1 invocations): bitwise
    vlen 128, m1, rms norm 2x160, rounding (1 invocations): bitwise
    vlen 128, m1, rms norm 1x67, adversarial (1 invocations): bitwise
    vlen 128, m1, rms norm 1x67, rounding (1 invocations): bitwise
    vlen 128, m4, softmax over H, W=32, adversarial (1 invocations): bitwise
    vlen 128, m4, softmax over H, W=32, rounding (1 invocations): bitwise
    vlen 128, m4, layer_norm over H, W=32, adversarial (1 invocations): bitwise
    vlen 128, m4, layer_norm over H, W=32, rounding (1 invocations): bitwise
    vlen 128, m4, rms norm 2x160, adversarial (1 invocations): bitwise
    vlen 128, m4, rms norm 2x160, rounding (1 invocations): bitwise
    vlen 128, m4, rms norm 1x67, adversarial (1 invocations): bitwise
    vlen 128, m4, rms norm 1x67, rounding (1 invocations): bitwise
    vlen 1024, m1, softmax over H, W=32, adversarial (1 invocations): bitwise
    vlen 1024, m1, softmax over H, W=32, rounding (1 invocations): bitwise
    vlen 1024, m1, layer_norm over H, W=32, adversarial (1 invocations): bitwise
    vlen 1024, m1, layer_norm over H, W=32, rounding (1 invocations): bitwise
    vlen 1024, m1, rms norm 2x160, adversarial (1 invocations): bitwise
    vlen 1024, m1, rms norm 2x160, rounding (1 invocations): bitwise
    vlen 1024, m1, rms norm 1x67, adversarial (1 invocations): bitwise
    vlen 1024, m1, rms norm 1x67, rounding (1 invocations): bitwise
    vlen 1024, m4, softmax over H, W=32, adversarial (1 invocations): bitwise
    vlen 1024, m4, softmax over H, W=32, rounding (1 invocations): bitwise
    vlen 1024, m4, layer_norm over H, W=32, adversarial (1 invocations): bitwise
    vlen 1024, m4, layer_norm over H, W=32, rounding (1 invocations): bitwise
    vlen 1024, m4, rms norm 2x160, adversarial (1 invocations): bitwise
    vlen 1024, m4, rms norm 2x160, rounding (1 invocations): bitwise
    vlen 1024, m4, rms norm 1x67, adversarial (1 invocations): bitwise
    vlen 1024, m4, rms norm 1x67, rounding (1 invocations): bitwise |}]

(* Thirty-two binary32 lanes a vector iteration ([zve32f128_x32]) as processes
   at VLEN 128 and 1024: eight registers a vector. *)
let%expect_test "32-lane plans under qemu" =
  let pipeline =
    Ssa_backends.Pipeline.Planned
      {
        numerics = Ssa_ir.Ssa_numerics.Fp32_ordered;
        target = Ssa_ir.Ssa_target.zve32f128_x32;
      }
  in
  List.iter
    (fun vlen ->
      List.iter
        (fun (name, g) ->
          T.check
            ~route:(R.route ~cpu:(cpu vlen) ())
            ~pipeline ~oracle:T.Fp32 ~constant:T.positive ~calls:1
            (Fmt.str "vlen %d, %s" vlen name)
            g)
        (T.vector_fixtures @ T.transformer_fixtures))
    [ 128; 1024 ];
  [%expect
    {|
    vlen 128, relu then add over C=32 (2 invocations): bitwise
    vlen 128, softmax over H, W=32 (1 invocations): bitwise
    vlen 128, layer_norm over H, W=32 (1 invocations): bitwise
    vlen 128, conv, batch norm, relu to C=32 (3 invocations): bitwise
    vlen 128, residual add 2x160 (1 invocations): bitwise
    vlen 128, residual add 1x67 (1 invocations): bitwise
    vlen 128, residual add 1x33 (1 invocations): bitwise
    vlen 128, rms norm 2x160 (1 invocations): bitwise
    vlen 128, rms norm 1x67 (1 invocations): bitwise
    vlen 128, rms norm 1x33 (1 invocations): bitwise
    vlen 128, tanh gelu mul 2x160 (2 invocations): bitwise
    vlen 128, tanh gelu mul 1x67 (2 invocations): bitwise
    vlen 128, tanh gelu mul 1x33 (2 invocations): bitwise
    vlen 1024, relu then add over C=32 (2 invocations): bitwise
    vlen 1024, softmax over H, W=32 (1 invocations): bitwise
    vlen 1024, layer_norm over H, W=32 (1 invocations): bitwise
    vlen 1024, conv, batch norm, relu to C=32 (3 invocations): bitwise
    vlen 1024, residual add 2x160 (1 invocations): bitwise
    vlen 1024, residual add 1x67 (1 invocations): bitwise
    vlen 1024, residual add 1x33 (1 invocations): bitwise
    vlen 1024, rms norm 2x160 (1 invocations): bitwise
    vlen 1024, rms norm 1x67 (1 invocations): bitwise
    vlen 1024, rms norm 1x33 (1 invocations): bitwise
    vlen 1024, tanh gelu mul 2x160 (2 invocations): bitwise
    vlen 1024, tanh gelu mul 1x67 (2 invocations): bitwise
    vlen 1024, tanh gelu mul 1x33 (2 invocations): bitwise |}]

(* Thirty-two-lane vectors in groups of eight registers (LMUL 8,
   [Registers.Grouped8]) under qemu at VLEN 128 and 1024, bitwise against the
   binary32 oracle. The pool holds two groups: a kernel whose forms need a
   third vector at once (a select's else, then and mask) leaves an operand in
   a slot, which the physical verifier refuses, and one that spills heavily
   (GELU) outgrows the frame's code model: refusals, never wrong answers. *)
let%expect_test "groups of eight under qemu" =
  let pipeline =
    Ssa_backends.Pipeline.Planned
      {
        numerics = Ssa_ir.Ssa_numerics.Fp32_ordered;
        target = Ssa_ir.Ssa_target.zve32f128_x32;
      }
  in
  List.iter
    (fun vlen ->
      List.iter
        (fun (name, g) ->
          T.check
            ~route:(R.route ~cpu:(cpu vlen) ~registers:R.Registers.Grouped8 ())
            ~pipeline ~oracle:T.Fp32 ~constant:T.positive ~calls:1
            (Fmt.str "vlen %d, %s" vlen name)
            g)
        (T.vector_fixtures @ T.transformer_fixtures))
    [ 128; 1024 ];
  [%expect
    {|
    vlen 128, relu then add over C=32: refused: invocation 0 (n0): publish: physical verifier: allocated fn0 bb7: target constraint: an undeclared slot
    vlen 128, softmax over H, W=32 (1 invocations): bitwise
    vlen 128, layer_norm over H, W=32 (1 invocations): bitwise
    vlen 128, conv, batch norm, relu to C=32: refused: invocation 0 (n0): publish: physical verifier: allocated fn0 bb32: target constraint: an undeclared slot; invocation 2 (n2): publish: physical verifier: allocated fn0 bb7: target constraint: an undeclared slot
    vlen 128, residual add 2x160 (1 invocations): bitwise
    vlen 128, residual add 1x67 (1 invocations): bitwise
    vlen 128, residual add 1x33 (1 invocations): bitwise
    vlen 128, rms norm 2x160 (1 invocations): bitwise
    vlen 128, rms norm 1x67 (1 invocations): bitwise
    vlen 128, rms norm 1x33 (1 invocations): bitwise
    vlen 128, tanh gelu mul 2x160: refused: invocation 0 (n0): publish: physical verifier: allocated fn0 bb4: target constraint: an undeclared slot
    vlen 128, tanh gelu mul 1x67: refused: invocation 0 (n0): publish: physical verifier: allocated fn0 bb7: target constraint: an undeclared slot
    vlen 128, tanh gelu mul 1x33: refused: invocation 0 (n0): publish: physical verifier: allocated fn0 bb7: target constraint: an undeclared slot
    vlen 1024, relu then add over C=32: refused: invocation 0 (n0): publish: physical verifier: allocated fn0 bb7: target constraint: an undeclared slot
    vlen 1024, softmax over H, W=32 (1 invocations): bitwise
    vlen 1024, layer_norm over H, W=32 (1 invocations): bitwise
    vlen 1024, conv, batch norm, relu to C=32: refused: invocation 0 (n0): publish: physical verifier: allocated fn0 bb32: target constraint: an undeclared slot; invocation 2 (n2): publish: physical verifier: allocated fn0 bb7: target constraint: an undeclared slot
    vlen 1024, residual add 2x160 (1 invocations): bitwise
    vlen 1024, residual add 1x67 (1 invocations): bitwise
    vlen 1024, residual add 1x33 (1 invocations): bitwise
    vlen 1024, rms norm 2x160 (1 invocations): bitwise
    vlen 1024, rms norm 1x67 (1 invocations): bitwise
    vlen 1024, rms norm 1x33 (1 invocations): bitwise
    vlen 1024, tanh gelu mul 2x160: refused: invocation 0 (n0): publish: physical verifier: allocated fn0 bb4: target constraint: an undeclared slot
    vlen 1024, tanh gelu mul 1x67: refused: invocation 0 (n0): publish: physical verifier: allocated fn0 bb7: target constraint: an undeclared slot
    vlen 1024, tanh gelu mul 1x33: refused: invocation 0 (n0): publish: physical verifier: allocated fn0 bb7: target constraint: an undeclared slot |}]
