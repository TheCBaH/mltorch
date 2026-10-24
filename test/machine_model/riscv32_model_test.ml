(* Whole bundles through the RV32 target, interpreted: every invocation's
   binary32 kernel selected onto ILP32, scheduled, linear-scan allocated,
   framed and published, then run on the physical interpreter against the
   binary32 oracle. The scalar plan and the target's i64 pairs are what an
   RV32IMF core runs. *)

module M = Machine_model.Mir_model
module T = Model_test

let fp32_scalar =
  Ssa_backends.Pipeline.Planned
    {
      numerics = Ssa_ir.Ssa_numerics.Fp32_ordered;
      target = Ssa_ir.Ssa_target.scalar;
    }

let routes =
  [
    ("riscv32 selected", Some (M.Route.Riscv32 M.Stage.Selected));
    ("riscv32 scanned", Some (M.Route.Riscv32 M.Stage.Scanned));
  ]

let%expect_test "binary32 bundles on RV32" =
  List.iter
    (fun (name, g) ->
      List.iter
        (fun (rname, route) ->
          T.check ?route ~pipeline:fp32_scalar ~oracle:T.Fp32
            ~constant:T.positive
            (Fmt.str "%s, %s" name rname)
            g)
        routes)
    T.vector_fixtures;
  [%expect
    {|
    relu then add over C=32, riscv32 selected (2 invocations): bitwise, bitwise
    relu then add over C=32, riscv32 scanned (2 invocations): bitwise, bitwise
    softmax over H, W=32, riscv32 selected (1 invocations): bitwise, bitwise
    softmax over H, W=32, riscv32 scanned (1 invocations): bitwise, bitwise
    layer_norm over H, W=32, riscv32 selected (1 invocations): bitwise, bitwise
    layer_norm over H, W=32, riscv32 scanned (1 invocations): bitwise, bitwise
    conv, batch norm, relu to C=32, riscv32 selected (3 invocations): bitwise, bitwise
    conv, batch norm, relu to C=32, riscv32 scanned (3 invocations): bitwise, bitwise |}]

let%expect_test "binary32 transcendentals on RV32" =
  List.iter
    (fun (name, g) ->
      T.check ~route:(M.Route.Riscv32 M.Stage.Scanned) ~pipeline:fp32_scalar
        ~oracle:T.Fp32 ~calls:1 ~input:T.sample_input name g)
    T.transcendental_fixtures;
  [%expect
    {|
    exp (1 invocations): bitwise
    log (1 invocations): bitwise
    sin (1 invocations): bitwise
    cos (1 invocations): bitwise
    gelu (erf) (1 invocations): bitwise |}]

(* The Zve32f target's plans: vector loops where the policy keeps each
   output's order (strided ones included, which are native there), run on the
   generic route, bitwise against the binary32 oracle. *)
let zve32f_planned target =
  Ssa_backends.Pipeline.Planned
    { numerics = Ssa_ir.Ssa_numerics.Fp32_ordered; target }

let%expect_test "Zve32f plans are the binary32 oracle" =
  List.iter
    (fun (name, g) ->
      List.iter
        (fun (tname, target) ->
          T.check ~pipeline:(zve32f_planned target) ~oracle:T.Fp32
            ~constant:T.positive
            (Fmt.str "%s, %s" name tname)
            g)
        [
          ("zve32f128", Ssa_ir.Ssa_target.zve32f128);
          ( "zve32f128 forced",
            Ssa_ir.Ssa_target.forced Ssa_ir.Ssa_target.zve32f128 );
        ])
    T.vector_fixtures;
  List.iter
    (fun (name, g) ->
      T.check
        ~pipeline:
          (zve32f_planned
             (Ssa_ir.Ssa_target.forced Ssa_ir.Ssa_target.zve32f128))
        ~oracle:T.Fp32 ~calls:1 ~input:T.sample_input name g)
    T.transcendental_fixtures;
  [%expect
    {|
    relu then add over C=32, zve32f128 (2 invocations): bitwise, bitwise
    relu then add over C=32, zve32f128 forced (2 invocations): bitwise, bitwise
    softmax over H, W=32, zve32f128 (1 invocations): bitwise, bitwise
    softmax over H, W=32, zve32f128 forced (1 invocations): bitwise, bitwise
    layer_norm over H, W=32, zve32f128 (1 invocations): bitwise, bitwise
    layer_norm over H, W=32, zve32f128 forced (1 invocations): bitwise, bitwise
    conv, batch norm, relu to C=32, zve32f128 (3 invocations): bitwise, bitwise
    conv, batch norm, relu to C=32, zve32f128 forced (3 invocations): bitwise, bitwise
    exp (1 invocations): bitwise
    log (1 invocations): bitwise
    sin (1 invocations): bitwise
    cos (1 invocations): bitwise
    gelu (erf) (1 invocations): bitwise |}]

(* How many loops each target's plan vectorizes: Zve32f takes the binary32
   plans' loops and, with no binary64 elements, none of a binary64 plan's
   under its cost model (an expanded loop stays legal, so the forced target
   still takes them, lane by lane). *)
let vector_loops ~numerics ~target b =
  List.fold_left
    (fun n inv ->
      match Ssa_backends.plan ~numerics ~target inv with
      | Ok p ->
          n
          + List.length
              (List.filter
                 (fun (d : Ssa_ir.Ssa_vectorize.Decision.t) ->
                   d.Ssa_ir.Ssa_vectorize.Decision.outcome
                   = Ssa_ir.Ssa_vectorize.Decision.Vectorized)
                 p.Ssa_ir.Ssa_plan.vectorized)
      | Error _ -> n)
    0 b.Loop_ir.Loop_bundle.invocations

let%expect_test "vector loops by target and policy" =
  let open Ssa_ir in
  List.iter
    (fun (name, g) ->
      let b = T.bundle (g ()) in
      let count numerics target = vector_loops ~numerics ~target b in
      Fmt.pr
        "%s: fp32 neon128 %d, zve32f128 %d, zve32f128 forced %d; reference_f64 \
         zve32f128 %d@."
        name
        (count Ssa_numerics.Fp32_ordered Ssa_target.neon128)
        (count Ssa_numerics.Fp32_ordered Ssa_target.zve32f128)
        (count Ssa_numerics.Fp32_ordered
           (Ssa_target.forced Ssa_target.zve32f128))
        (count Ssa_numerics.Reference_f64 Ssa_target.zve32f128))
    T.vector_fixtures;
  [%expect
    {|
    relu then add over C=32: fp32 neon128 2, zve32f128 2, zve32f128 forced 2; reference_f64 zve32f128 0
    softmax over H, W=32: fp32 neon128 0, zve32f128 0, zve32f128 forced 0; reference_f64 zve32f128 0
    layer_norm over H, W=32: fp32 neon128 0, zve32f128 0, zve32f128 forced 0; reference_f64 zve32f128 0
    conv, batch norm, relu to C=32: fp32 neon128 3, zve32f128 3, zve32f128 forced 3; reference_f64 zve32f128 0 |}]

let%expect_test "matrix products on RV32" =
  List.iter
    (fun (name, g) ->
      List.iter
        (fun (iname, input, constant) ->
          T.check ~route:(M.Route.Riscv32 M.Stage.Scanned) ~pipeline:fp32_scalar
            ~oracle:T.Fp32 ~input ~constant
            (Fmt.str "%s, %s" name iname)
            g)
        [
          ("values", (fun ~salt _ -> T.values ~salt), T.values ~salt:3);
          ( "adversarial",
            T.adversarial_input,
            fun sg -> T.adversarial ~salt:3 sg );
        ])
    T.contraction_fixtures;
  [%expect
    {|
    linear 9x32 k64, values (1 invocations): bitwise, bitwise
    linear 9x32 k64, adversarial (1 invocations): bitwise, bitwise
    linear 9x32 k64 bias, values (1 invocations): bitwise, bitwise
    linear 9x32 k64 bias, adversarial (1 invocations): bitwise, bitwise
    linear 7x29 k63 bias, values (1 invocations): bitwise, bitwise
    linear 7x29 k63 bias, adversarial (1 invocations): bitwise, bitwise
    bmm 1x9x32 k64, values (1 invocations): bitwise, bitwise
    bmm 1x9x32 k64, adversarial (1 invocations): bitwise, bitwise
    bmm 3x7x29 k63, values (1 invocations): bitwise, bitwise
    bmm 3x7x29 k63, adversarial (1 invocations): bitwise, bitwise |}]

(* [Fp32_contracted] on RV32, scalar ([fmadd.s]) and Zve32f ([vfmacc.vv]),
   against the contracted oracle. *)
let%expect_test "contracted matrix products on RV32" =
  List.iter
    (fun (name, g) ->
      List.iter
        (fun (tname, target) ->
          T.check ~route:(M.Route.Riscv32 M.Stage.Scanned)
            ~pipeline:
              (Ssa_backends.Pipeline.Planned
                 { numerics = Ssa_ir.Ssa_numerics.Fp32_contracted; target })
            ~oracle:T.Fp32_contracted ~same:T.same_portable
            ~input:T.adversarial_input
            ~constant:(fun sg -> T.adversarial ~salt:3 sg)
            (Fmt.str "%s, %s" name tname)
            g)
        [
          ("scalar", Ssa_ir.Ssa_target.scalar);
          ("zve32f128", Ssa_ir.Ssa_target.zve32f128);
        ])
    (T.contraction_fixtures @ [ List.nth T.vector_fixtures 3 ]);
  [%expect
    {|
    linear 9x32 k64, scalar (1 invocations): bitwise, bitwise
    linear 9x32 k64, zve32f128 (1 invocations): bitwise, bitwise
    linear 9x32 k64 bias, scalar (1 invocations): bitwise, bitwise
    linear 9x32 k64 bias, zve32f128 (1 invocations): bitwise, bitwise
    linear 7x29 k63 bias, scalar (1 invocations): bitwise, bitwise
    linear 7x29 k63 bias, zve32f128 (1 invocations): bitwise, bitwise
    bmm 1x9x32 k64, scalar (1 invocations): bitwise, bitwise
    bmm 1x9x32 k64, zve32f128 (1 invocations): bitwise, bitwise
    bmm 3x7x29 k63, scalar (1 invocations): bitwise, bitwise
    bmm 3x7x29 k63, zve32f128 (1 invocations): bitwise, bitwise
    conv, batch norm, relu to C=32, scalar (3 invocations): bitwise, bitwise
    conv, batch norm, relu to C=32, zve32f128 (3 invocations): bitwise, bitwise |}]

(* Sixteen-lane vectors in groups of four registers, selected and scanned,
   on the physical interpreter and checker; the reference allocator refuses
   them. *)
let%expect_test "register groups on RV32" =
  let planned numerics =
    Ssa_backends.Pipeline.Planned
      { numerics; target = Ssa_ir.Ssa_target.zve32f128 }
  in
  List.iter
    (fun (name, g) ->
      List.iter
        (fun (sname, stage) ->
          T.check ~route:(M.Route.Riscv32_grouped stage)
            ~pipeline:(planned Ssa_ir.Ssa_numerics.Fp32_ordered)
            ~oracle:T.Fp32 ~constant:T.positive ~calls:1
            (Fmt.str "%s, %s" name sname)
            g)
        [ ("selected", M.Stage.Selected); ("scanned", M.Stage.Scanned) ];
      T.check ~route:(M.Route.Riscv32_grouped M.Stage.Scanned)
        ~pipeline:(planned Ssa_ir.Ssa_numerics.Fp32_contracted)
        ~oracle:T.Fp32_contracted ~same:T.same_portable
        ~input:T.adversarial_input
        ~constant:(fun sg -> T.adversarial ~salt:3 sg)
        ~calls:1
        (Fmt.str "%s, scanned, contracted" name)
        g)
    (T.vector_fixtures @ T.contraction_fixtures);
  T.check ~route:(M.Route.Riscv32_grouped M.Stage.Realized)
    ~pipeline:(planned Ssa_ir.Ssa_numerics.Fp32_ordered)
    ~oracle:T.Fp32 ~calls:1 "reference allocation"
    (List.hd T.vector_fixtures |> snd);
  [%expect
    {|
    relu then add over C=32, selected (2 invocations): bitwise
    relu then add over C=32, scanned (2 invocations): bitwise
    relu then add over C=32, scanned, contracted (2 invocations): bitwise
    softmax over H, W=32, selected (1 invocations): bitwise
    softmax over H, W=32, scanned (1 invocations): bitwise
    softmax over H, W=32, scanned, contracted (1 invocations): bitwise
    layer_norm over H, W=32, selected (1 invocations): bitwise
    layer_norm over H, W=32, scanned (1 invocations): bitwise
    layer_norm over H, W=32, scanned, contracted (1 invocations): bitwise
    conv, batch norm, relu to C=32, selected (3 invocations): bitwise
    conv, batch norm, relu to C=32, scanned (3 invocations): bitwise
    conv, batch norm, relu to C=32, scanned, contracted (3 invocations): bitwise
    linear 9x32 k64, selected (1 invocations): bitwise
    linear 9x32 k64, scanned (1 invocations): bitwise
    linear 9x32 k64, scanned, contracted (1 invocations): bitwise
    linear 9x32 k64 bias, selected (1 invocations): bitwise
    linear 9x32 k64 bias, scanned (1 invocations): bitwise
    linear 9x32 k64 bias, scanned, contracted (1 invocations): bitwise
    linear 7x29 k63 bias, selected (1 invocations): bitwise
    linear 7x29 k63 bias, scanned (1 invocations): bitwise
    linear 7x29 k63 bias, scanned, contracted (1 invocations): bitwise
    bmm 1x9x32 k64, selected (1 invocations): bitwise
    bmm 1x9x32 k64, scanned (1 invocations): bitwise
    bmm 1x9x32 k64, scanned, contracted (1 invocations): bitwise
    bmm 3x7x29 k63, selected (1 invocations): bitwise
    bmm 3x7x29 k63, scanned (1 invocations): bitwise
    bmm 3x7x29 k63, scanned, contracted (1 invocations): bitwise
    reference allocation: refused: invocation 0 (n0): the reference allocator does not allocate register groups (realized); invocation 1 (n1): the reference allocator does not allocate register groups (realized) |}]
