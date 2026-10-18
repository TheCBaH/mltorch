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
