(* Whole binary32 bundles on an RV32IMF process under qemu: every invocation
   selected onto ILP32, linear-scan allocated, framed, published, emitted as a
   typed Rivet module and run as its own static ELF. Each call is bitwise
   against the binary32 oracle. *)

module M = Machine_model.Mir_model
module T = Machine_model_test.Model_test
module R = Machine_rivet_riscv32_mir.Rv32_mir_route

let fp32_scalar =
  Ssa_backends.Pipeline.Planned
    {
      numerics = Ssa_ir.Ssa_numerics.Fp32_ordered;
      target = Ssa_ir.Ssa_target.scalar;
    }

let%expect_test "binary32 bundles under qemu" =
  List.iter
    (fun (name, g) ->
      T.check ~route:(R.route ()) ~pipeline:fp32_scalar ~oracle:T.Fp32
        ~constant:T.positive name g)
    T.vector_fixtures;
  [%expect
    {|
    relu then add over C=32 (2 invocations): bitwise, bitwise
    softmax over H, W=32 (1 invocations): bitwise, bitwise
    layer_norm over H, W=32 (1 invocations): bitwise, bitwise
    conv, batch norm, relu to C=32 (3 invocations): bitwise, bitwise |}]

(* A NaN compares equal to any NaN: the oracle runs on the host, whose default
   NaN is signed by the host (x86-64 sets the sign, AArch64 and RISC-V clear
   it), and payloads are not modelled. *)
let%expect_test "binary32 transcendentals under qemu" =
  List.iter
    (fun (name, g) ->
      T.check ~route:(R.route ()) ~pipeline:fp32_scalar ~oracle:T.Fp32 ~calls:1
        ~same:T.same_portable ~input:T.sample_input name g)
    T.transcendental_fixtures;
  [%expect
    {|
    exp (1 invocations): bitwise
    log (1 invocations): bitwise
    sin (1 invocations): bitwise
    cos (1 invocations): bitwise
    gelu (erf) (1 invocations): bitwise |}]

(* The same bundles with one emission defect each: every one is caught by
   the process's own outputs, so the comparison above is not vacuous. *)
let%expect_test "emission mutations are caught under qemu" =
  let conv = List.assoc "conv, batch norm, relu to C=32" T.vector_fixtures
  and softmax = List.assoc "softmax over H, W=32" T.vector_fixtures in
  List.iter
    (fun (name, mutation, g) ->
      T.check ~route:(R.route ~mutation ()) ~pipeline:fp32_scalar ~oracle:T.Fp32
        ~constant:T.positive ~calls:1 name g)
    Machine_rivet_riscv32_mir.Rv32_mir_form.Mutation.(
      [
        ("branch sense", Branch_sense, conv);
        ("select arms", Select_arms, softmax);
      ]
      @ [
          ("commuted sub, chain", Commuted_sub, Native_test.Graph_fixtures.chain);
          ("shifted slot, chain", Shifted_slot, Native_test.Graph_fixtures.chain);
        ]);
  [%expect
    {|
    branch sense (3 invocations): DIFFERS
    select arms (1 invocations): DIFFERS
    commuted sub, chain (3 invocations): DIFFERS
    shifted slot, chain (3 invocations): DIFFERS |}]

(* The same bundles planned for Zve32f: vector loops where the policy keeps
   each output's order, as RVV forms, at VLEN 128 and 1024 with tail and
   masked-off elements filled with ones. *)
let zve32f =
  Ssa_backends.Pipeline.Planned
    {
      numerics = Ssa_ir.Ssa_numerics.Fp32_ordered;
      target = Ssa_ir.Ssa_target.zve32f128;
    }

let zve32f_forced =
  Ssa_backends.Pipeline.Planned
    {
      numerics = Ssa_ir.Ssa_numerics.Fp32_ordered;
      target = Ssa_ir.Ssa_target.forced Ssa_ir.Ssa_target.zve32f128;
    }

let cpu vlen =
  Fmt.str
    "rv32,d=false,zve32f=true,vlen=%d,elen=32,rvv_ta_all_1s=true,rvv_ma_all_1s=true"
    vlen

let%expect_test "Zve32f bundles under qemu" =
  List.iter
    (fun vlen ->
      List.iter
        (fun (name, g) ->
          T.check
            ~route:(R.route ~cpu:(cpu vlen) ())
            ~pipeline:zve32f ~oracle:T.Fp32 ~constant:T.positive
            (Fmt.str "vlen %d, %s" vlen name)
            g)
        T.vector_fixtures;
      List.iter
        (fun (name, g) ->
          T.check
            ~route:(R.route ~cpu:(cpu vlen) ())
            ~pipeline:zve32f_forced ~oracle:T.Fp32 ~calls:1
            ~same:T.same_portable ~input:T.sample_input
            (Fmt.str "vlen %d, %s, every legal loop" vlen name)
            g)
        T.transcendental_fixtures)
    [ 128; 1024 ];
  [%expect
    {|
    vlen 128, relu then add over C=32 (2 invocations): bitwise, bitwise
    vlen 128, softmax over H, W=32 (1 invocations): bitwise, bitwise
    vlen 128, layer_norm over H, W=32 (1 invocations): bitwise, bitwise
    vlen 128, conv, batch norm, relu to C=32 (3 invocations): bitwise, bitwise
    vlen 128, exp, every legal loop (1 invocations): bitwise
    vlen 128, log, every legal loop (1 invocations): bitwise
    vlen 128, sin, every legal loop (1 invocations): bitwise
    vlen 128, cos, every legal loop (1 invocations): bitwise
    vlen 128, gelu (erf), every legal loop (1 invocations): bitwise
    vlen 1024, relu then add over C=32 (2 invocations): bitwise, bitwise
    vlen 1024, softmax over H, W=32 (1 invocations): bitwise, bitwise
    vlen 1024, layer_norm over H, W=32 (1 invocations): bitwise, bitwise
    vlen 1024, conv, batch norm, relu to C=32 (3 invocations): bitwise, bitwise
    vlen 1024, exp, every legal loop (1 invocations): bitwise
    vlen 1024, log, every legal loop (1 invocations): bitwise
    vlen 1024, sin, every legal loop (1 invocations): bitwise
    vlen 1024, cos, every legal loop (1 invocations): bitwise
    vlen 1024, gelu (erf), every legal loop (1 invocations): bitwise |}]

(* The matrix products under qemu: [Fp32_ordered] against the binary32
   oracle, and [Fp32_contracted] ([fmadd.s] scalar, [vfmacc.vv] at VLEN 128
   and 1024) against the contracted oracle, on adversarial inputs. *)
let%expect_test "matrix products under qemu, ordered and contracted" =
  let planned numerics target =
    Ssa_backends.Pipeline.Planned { numerics; target }
  in
  List.iter
    (fun (name, g) ->
      List.iter
        (fun (pname, pipeline, oracle, cpu) ->
          T.check ~route:(R.route ?cpu ()) ~pipeline ~oracle ~calls:1
            ~same:T.same_portable ~input:T.adversarial_input
            ~constant:(fun sg -> T.adversarial ~salt:3 sg)
            (Fmt.str "%s, %s" name pname)
            g)
        Ssa_ir.Ssa_numerics.
          [
            ( "ordered scalar",
              planned Fp32_ordered Ssa_ir.Ssa_target.scalar,
              T.Fp32,
              None );
            ( "ordered zve32f128",
              planned Fp32_ordered Ssa_ir.Ssa_target.zve32f128,
              T.Fp32,
              Some (cpu 128) );
            ( "contracted scalar",
              planned Fp32_contracted Ssa_ir.Ssa_target.scalar,
              T.Fp32_contracted,
              None );
            ( "contracted zve32f128 vlen 128",
              planned Fp32_contracted Ssa_ir.Ssa_target.zve32f128,
              T.Fp32_contracted,
              Some (cpu 128) );
            ( "contracted zve32f128 vlen 1024",
              planned Fp32_contracted Ssa_ir.Ssa_target.zve32f128,
              T.Fp32_contracted,
              Some (cpu 1024) );
          ])
    T.contraction_fixtures;
  [%expect
    {|
    linear 9x32 k64, ordered scalar (1 invocations): bitwise
    linear 9x32 k64, ordered zve32f128 (1 invocations): bitwise
    linear 9x32 k64, contracted scalar (1 invocations): bitwise
    linear 9x32 k64, contracted zve32f128 vlen 128 (1 invocations): bitwise
    linear 9x32 k64, contracted zve32f128 vlen 1024 (1 invocations): bitwise
    linear 9x32 k64 bias, ordered scalar (1 invocations): bitwise
    linear 9x32 k64 bias, ordered zve32f128 (1 invocations): bitwise
    linear 9x32 k64 bias, contracted scalar (1 invocations): bitwise
    linear 9x32 k64 bias, contracted zve32f128 vlen 128 (1 invocations): bitwise
    linear 9x32 k64 bias, contracted zve32f128 vlen 1024 (1 invocations): bitwise
    linear 7x29 k63 bias, ordered scalar (1 invocations): bitwise
    linear 7x29 k63 bias, ordered zve32f128 (1 invocations): bitwise
    linear 7x29 k63 bias, contracted scalar (1 invocations): bitwise
    linear 7x29 k63 bias, contracted zve32f128 vlen 128 (1 invocations): bitwise
    linear 7x29 k63 bias, contracted zve32f128 vlen 1024 (1 invocations): bitwise
    bmm 1x9x32 k64, ordered scalar (1 invocations): bitwise
    bmm 1x9x32 k64, ordered zve32f128 (1 invocations): bitwise
    bmm 1x9x32 k64, contracted scalar (1 invocations): bitwise
    bmm 1x9x32 k64, contracted zve32f128 vlen 128 (1 invocations): bitwise
    bmm 1x9x32 k64, contracted zve32f128 vlen 1024 (1 invocations): bitwise
    bmm 3x7x29 k63, ordered scalar (1 invocations): bitwise
    bmm 3x7x29 k63, ordered zve32f128 (1 invocations): bitwise
    bmm 3x7x29 k63, contracted scalar (1 invocations): bitwise
    bmm 3x7x29 k63, contracted zve32f128 vlen 128 (1 invocations): bitwise
    bmm 3x7x29 k63, contracted zve32f128 vlen 1024 (1 invocations): bitwise |}]

(* Sixteen-lane vectors in groups of four registers (LMUL 4) under qemu at
   VLEN 128 and 1024, ordered and contracted, then three defects the runs
   must catch: a group copy that moves its first register alone, and a group
   spilled into half its bytes (the physical verifier refuses it). *)
let%expect_test "register groups under qemu" =
  let planned numerics =
    Ssa_backends.Pipeline.Planned
      { numerics; target = Ssa_ir.Ssa_target.zve32f128 }
  in
  let grouped ?mutation ?scan_mutation vlen =
    R.route ?mutation ?scan_mutation ~registers:R.Registers.Grouped
      ~cpu:(cpu vlen) ()
  in
  List.iter
    (fun vlen ->
      List.iter
        (fun (name, g) ->
          T.check ~route:(grouped vlen)
            ~pipeline:(planned Ssa_ir.Ssa_numerics.Fp32_ordered)
            ~oracle:T.Fp32 ~constant:T.positive ~calls:1
            (Fmt.str "vlen %d, %s" vlen name)
            g;
          T.check ~route:(grouped vlen)
            ~pipeline:(planned Ssa_ir.Ssa_numerics.Fp32_contracted)
            ~oracle:T.Fp32_contracted ~same:T.same_portable
            ~input:T.adversarial_input
            ~constant:(fun sg -> T.adversarial ~salt:3 sg)
            ~calls:1
            (Fmt.str "vlen %d, %s, contracted" vlen name)
            g)
        (List.hd T.vector_fixtures
        :: List.nth T.vector_fixtures 3
        :: T.contraction_fixtures))
    [ 128; 1024 ];
  let conv = snd (List.nth T.vector_fixtures 3) in
  List.iter
    (fun (name, route) ->
      T.check ~route
        ~pipeline:(planned Ssa_ir.Ssa_numerics.Fp32_ordered)
        ~oracle:T.Fp32 ~constant:T.positive ~calls:1 name conv)
    [
      ( "group copy first",
        grouped
          ~mutation:
            Machine_rivet_riscv32_mir.Rv32_mir_form.Mutation.Group_copy_first
          128 );
      ( "half spill",
        grouped ~scan_mutation:Machine_alloc.Mir_scan.Mutation.Half_spill 128 );
    ];
  [%expect
    {|
    vlen 128, relu then add over C=32 (2 invocations): bitwise
    vlen 128, relu then add over C=32, contracted (2 invocations): bitwise
    vlen 128, conv, batch norm, relu to C=32 (3 invocations): bitwise
    vlen 128, conv, batch norm, relu to C=32, contracted (3 invocations): bitwise
    vlen 128, linear 9x32 k64 (1 invocations): bitwise
    vlen 128, linear 9x32 k64, contracted (1 invocations): bitwise
    vlen 128, linear 9x32 k64 bias (1 invocations): bitwise
    vlen 128, linear 9x32 k64 bias, contracted (1 invocations): bitwise
    vlen 128, linear 7x29 k63 bias (1 invocations): bitwise
    vlen 128, linear 7x29 k63 bias, contracted (1 invocations): bitwise
    vlen 128, bmm 1x9x32 k64 (1 invocations): bitwise
    vlen 128, bmm 1x9x32 k64, contracted (1 invocations): bitwise
    vlen 128, bmm 3x7x29 k63 (1 invocations): bitwise
    vlen 128, bmm 3x7x29 k63, contracted (1 invocations): bitwise
    vlen 1024, relu then add over C=32 (2 invocations): bitwise
    vlen 1024, relu then add over C=32, contracted (2 invocations): bitwise
    vlen 1024, conv, batch norm, relu to C=32 (3 invocations): bitwise
    vlen 1024, conv, batch norm, relu to C=32, contracted (3 invocations): bitwise
    vlen 1024, linear 9x32 k64 (1 invocations): bitwise
    vlen 1024, linear 9x32 k64, contracted (1 invocations): bitwise
    vlen 1024, linear 9x32 k64 bias (1 invocations): bitwise
    vlen 1024, linear 9x32 k64 bias, contracted (1 invocations): bitwise
    vlen 1024, linear 7x29 k63 bias (1 invocations): bitwise
    vlen 1024, linear 7x29 k63 bias, contracted (1 invocations): bitwise
    vlen 1024, bmm 1x9x32 k64 (1 invocations): bitwise
    vlen 1024, bmm 1x9x32 k64, contracted (1 invocations): bitwise
    vlen 1024, bmm 3x7x29 k63 (1 invocations): bitwise
    vlen 1024, bmm 3x7x29 k63, contracted (1 invocations): bitwise
    group copy first (3 invocations): DIFFERS
    half spill: refused: invocation 0 (n0): publish: physical verifier: allocated fn0 bb18: target constraint: a location that does not fit its value |}]

(* Contractions as generated RVV functions linked beside each module, under
   qemu at VLEN 128 and 1024, ordered and fused; then two kernel defects the
   runs must catch: A's rows one element apart too far, the last column tile
   one lane short. *)
let%expect_test "generated contractions under qemu" =
  let planned numerics =
    Ssa_backends.Pipeline.Planned
      { numerics; target = Ssa_ir.Ssa_target.zve32f128 }
  in
  let run ?kernel_mutation ~vlen name numerics oracle g =
    T.check
      ~route:(R.route ?kernel_mutation ~cpu:(cpu vlen) ())
      ~contractions:true ~pipeline:(planned numerics) ~oracle
      ~same:T.same_portable ~input:T.adversarial_input
      ~constant:(fun sg -> T.adversarial ~salt:3 sg)
      ~calls:1 name g
  in
  List.iter
    (fun vlen ->
      List.iter
        (fun (name, g) ->
          run ~vlen
            (Fmt.str "vlen %d, %s, ordered" vlen name)
            Ssa_ir.Ssa_numerics.Fp32_ordered T.Fp32 g;
          run ~vlen
            (Fmt.str "vlen %d, %s, contracted" vlen name)
            Ssa_ir.Ssa_numerics.Fp32_contracted T.Fp32_contracted g)
        T.contraction_fixtures)
    [ 128; 1024 ];
  let tails = List.assoc "linear 7x29 k63 bias" T.contraction_fixtures in
  List.iter
    (fun (name, kernel_mutation) ->
      run ~kernel_mutation ~vlen:128 name Ssa_ir.Ssa_numerics.Fp32_contracted
        T.Fp32_contracted tails)
    Coralnpu_kernels.Rvv_contraction.Mutation.
      [ ("A row stride", A_row_stride); ("tail vl", Tail_vl) ];
  [%expect
    {|
    vlen 128, linear 9x32 k64, ordered (1 invocations): bitwise
    vlen 128, linear 9x32 k64, contracted (1 invocations): bitwise
    vlen 128, linear 9x32 k64 bias, ordered (1 invocations): bitwise
    vlen 128, linear 9x32 k64 bias, contracted (1 invocations): bitwise
    vlen 128, linear 7x29 k63 bias, ordered (1 invocations): bitwise
    vlen 128, linear 7x29 k63 bias, contracted (1 invocations): bitwise
    vlen 128, bmm 1x9x32 k64, ordered (1 invocations): bitwise
    vlen 128, bmm 1x9x32 k64, contracted (1 invocations): bitwise
    vlen 128, bmm 3x7x29 k63, ordered (1 invocations): bitwise
    vlen 128, bmm 3x7x29 k63, contracted (1 invocations): bitwise
    vlen 1024, linear 9x32 k64, ordered (1 invocations): bitwise
    vlen 1024, linear 9x32 k64, contracted (1 invocations): bitwise
    vlen 1024, linear 9x32 k64 bias, ordered (1 invocations): bitwise
    vlen 1024, linear 9x32 k64 bias, contracted (1 invocations): bitwise
    vlen 1024, linear 7x29 k63 bias, ordered (1 invocations): bitwise
    vlen 1024, linear 7x29 k63 bias, contracted (1 invocations): bitwise
    vlen 1024, bmm 1x9x32 k64, ordered (1 invocations): bitwise
    vlen 1024, bmm 1x9x32 k64, contracted (1 invocations): bitwise
    vlen 1024, bmm 3x7x29 k63, ordered (1 invocations): bitwise
    vlen 1024, bmm 3x7x29 k63, contracted (1 invocations): bitwise
    A row stride (1 invocations): DIFFERS
    tail vl (1 invocations): DIFFERS |}]

(* The generated functions over bfloat16 operands and over 1x1 convolutions'
   flattened rows, under qemu at VLEN 128, ordered and contracted; then A's
   rows one element too far apart over bfloat16, which the runs must catch. *)
let%expect_test "generated bfloat16 and 1x1 convolution contractions under qemu"
    =
  let planned numerics =
    Ssa_backends.Pipeline.Planned
      { numerics; target = Ssa_ir.Ssa_target.zve32f128 }
  in
  let run ?kernel_mutation name numerics oracle g =
    T.check
      ~route:(R.route ?kernel_mutation ~cpu:(cpu 128) ())
      ~contractions:true ~pipeline:(planned numerics) ~oracle
      ~same:T.same_portable
      ~input:(fun ~salt id sg ->
        T.in_format sg (T.adversarial_input ~salt id sg))
      ~constant:(fun sg -> T.in_format sg (T.adversarial ~salt:3 sg))
      ~calls:1 name g
  in
  List.iter
    (fun (name, g) ->
      run (name ^ ", ordered") Ssa_ir.Ssa_numerics.Fp32_ordered T.Fp32 g;
      run (name ^ ", contracted") Ssa_ir.Ssa_numerics.Fp32_contracted
        T.Fp32_contracted g)
    (T.bf16_fixtures
    @ List.filter
        (fun (n, _) ->
          List.mem n [ "conv1x1 3x5 c19 to 29"; "conv1x1 3x5 c1 to 4" ])
        T.pointwise_fixtures);
  run ~kernel_mutation:Coralnpu_kernels.Rvv_contraction.Mutation.A_row_stride
    "bfloat16, A row stride" Ssa_ir.Ssa_numerics.Fp32_contracted
    T.Fp32_contracted
    (List.assoc "linear bf16 7x29 k63 bias" T.bf16_fixtures);
  [%expect
    {|
    bmm bf16 3x7x29 k63, ordered (1 invocations): bitwise
    bmm bf16 3x7x29 k63, contracted (1 invocations): bitwise
    linear bf16 9x32 k64, ordered (1 invocations): bitwise
    linear bf16 9x32 k64, contracted (1 invocations): bitwise
    linear bf16 7x29 k63 bias, ordered (1 invocations): bitwise
    linear bf16 7x29 k63 bias, contracted (1 invocations): bitwise
    conv1x1 3x5 c19 to 29, ordered (1 invocations): bitwise
    conv1x1 3x5 c19 to 29, contracted (1 invocations): bitwise
    conv1x1 3x5 c1 to 4, ordered (1 invocations): bitwise
    conv1x1 3x5 c1 to 4, contracted (1 invocations): bitwise
    bfloat16, A row stride (1 invocations): DIFFERS |}]

(* Prepacked weights through the generated functions under qemu: a Linear's
   and a 1x1 convolution's weight read as packed copies with contiguous
   columns, binary32 and bfloat16, bitwise call after call. *)
let%expect_test "generated contractions over prepacked weights under qemu" =
  List.iter
    (fun (name, g) ->
      T.check
        ~route:(R.route ~cpu:(cpu 128) ())
        ~contractions:true ~prepack:true
        ~pipeline:
          (Ssa_backends.Pipeline.Planned
             {
               numerics = Ssa_ir.Ssa_numerics.Fp32_contracted;
               target = Ssa_ir.Ssa_target.zve32f128;
             })
        ~oracle:T.Fp32_contracted ~same:T.same_portable
        ~input:(fun ~salt id sg ->
          T.in_format sg (T.adversarial_input ~salt id sg))
        ~constant:(fun sg -> T.in_format sg (T.adversarial ~salt:3 sg))
        ~calls:2 name g)
    (List.filter
       (fun (n, _) ->
         List.mem n
           [
             "linear 9x32 k64";
             "linear 7x29 k63 bias";
             "linear bf16 7x29 k63 bias";
             "conv1x1 3x5 c19 to 29";
           ])
       (T.contraction_fixtures @ T.bf16_fixtures @ T.pointwise_fixtures));
  [%expect
    {|
    linear 9x32 k64 (1 invocations, 1 packed): bitwise, bitwise
    linear 7x29 k63 bias (1 invocations, 1 packed): bitwise, bitwise
    linear bf16 7x29 k63 bias (1 invocations, 1 packed): bitwise, bitwise
    conv1x1 3x5 c19 to 29 (1 invocations, 1 packed): bitwise, bitwise |}]

(* A selected attention has no realization under qemu: the generated function
   is the matrix engine's, so the route refuses the node rather than running
   anything else. *)
let%expect_test "an attention under qemu is refused" =
  let module B = Machine_model_test.Attention_binding_test in
  let g = T.sdpa_bf16_fixture ~tq:16 ~tk:32 ~d:32 () in
  let node = B.sdpa_node g in
  let site =
    Result.get_ok
      (Machine_model.Sdpa_select.choose
         ~grant:(Some Machine_model.Sdpa_select.Grant.default) B.engine g node)
  in
  T.check
    ~route:(R.route ~cpu:(cpu 128) ())
    ~attention:(fun id ->
      if Graph_ir.Node_id.equal id node then Some site else None)
    ~pipeline:
      (Ssa_backends.Pipeline.Planned
         {
           numerics = Ssa_ir.Ssa_numerics.Fp32_ordered;
           target = Ssa_ir.Ssa_target.zve32f128;
         })
    ~oracle:T.Fp32 ~calls:1 "sdpa bf16 16x32 d32"
    (fun () -> g);
  [%expect
    {| sdpa bf16 16x32 d32: refused: invocation 0 (n0): attention attend.b1.h1.q16.k32.d32.sq0_0_32_1.sk0_0_32_1.sv0_0_32_1.so0_0_32_1.nomask.x3e3504f3.n32: an attention has no realization on this route |}]
