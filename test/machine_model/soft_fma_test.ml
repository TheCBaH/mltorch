(* A fused multiply-add where the target has none, in the case that separates it
   from a product and an add: [m * m] with [m = 1 + 2^-12] is
   [1 + 2^-11 + 2^-24], exactly halfway between two binary32 neighbours, and a
   tiny addend of either sign decides which way the one rounding goes. A product
   rounded to binary64 first, then to binary32, takes the tie to even and is
   wrong for one of the two. The addend is the first step's accumulator of a sum
   over k = 2, so the second step is the fused one. Every route must give what
   [F32_math.fma32] gives (the contracted oracle). *)

open Model_test
module M = Machine_model.Mir_model
module Tid = Native_test.Graph_fixtures

let contracted =
  Ssa_backends.Pipeline.Planned
    {
      numerics = Ssa_ir.Ssa_numerics.Fp32_contracted;
      target = Ssa_ir.Ssa_target.scalar;
    }

let%expect_test "the fused step is the single rounding on every route" =
  let m = 1. +. ldexp 1. (-12) and tiny = ldexp 1. (-30) in
  List.iter
    (fun (name, a, b) ->
      List.iter
        (fun (rname, route) ->
          check ?route ~pipeline:contracted ~oracle:Fp32_contracted
            ~same:same_portable ~calls:1
            ~input:(fun ~salt:_ i (sg : Tensor_sig.t) ->
              let cells = if i = 0 then a else b in
              Tensor.materialize sg.Tensor_sig.shape (fun c ->
                  cells.((Vec6.offset sg.Tensor_sig.shape c :> int))))
            (Fmt.str "%s, %s" name rname)
            (bmm_fixture ~batch:1 ~m:1 ~n:1 ~k:2))
        [
          ("generic", None);
          ("aarch64", Some (M.Route.Aarch64 M.Stage.Scanned));
          ("x86_64", Some (M.Route.X86_64 M.Stage.Scanned));
        ])
    [
      ("tiny addend above", [| tiny; m |], [| tiny; m |]);
      ("tiny addend below", [| tiny; m |], [| -.tiny; m |]);
      ("negated, above", [| tiny; -.m |], [| -.tiny; m |]);
      ("negated, below", [| tiny; -.m |], [| tiny; m |]);
    ];
  [%expect
    {|
    tiny addend above, generic (1 invocations): bitwise
    tiny addend above, aarch64 (1 invocations): bitwise
    tiny addend above, x86_64 (1 invocations): bitwise
    tiny addend below, generic (1 invocations): bitwise
    tiny addend below, aarch64 (1 invocations): bitwise
    tiny addend below, x86_64 (1 invocations): bitwise
    negated, above, generic (1 invocations): bitwise
    negated, above, aarch64 (1 invocations): bitwise
    negated, above, x86_64 (1 invocations): bitwise
    negated, below, generic (1 invocations): bitwise
    negated, below, aarch64 (1 invocations): bitwise
    negated, below, x86_64 (1 invocations): bitwise |}]
