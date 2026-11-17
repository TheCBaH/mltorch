(* The band driver's chains on the CPU. Each banded program of the pointwise
   and window chains in the band-driver and view-call suites is lowered through
   the contraction helper route, selected, allocated and published for AArch64,
   run natively through Rivet, and its output bits compared with the
   whole-tensor chain on the SSA interpreter ({!Band_driver.whole}). The
   contraction helpers are the route's own generated code
   ({!Machine_rivet_aarch64.Rivet_a64_contraction}). *)

module Src = Machine_source_test.Mir_source
module H = Native_harness
module C = Band_chains

let native ?mutation (p : Ssa_ir.Ssa_program.t) ~inputs =
  match
    Src.case_of_program p
      ~inputs:(List.map (fun (i, c) -> (i, Ssa_ir.Ssa_memory.Floats c)) inputs)
      ~precision:Machine_ir.Mir_planning.Precision.F32 ()
  with
  | Error e -> Error e
  | Ok case -> (
      match H.build case ~sites:[||] with
      | Error e -> Error ("not built: " ^ e)
      | Ok b -> (
          match H.native ?mutation case ~sites:[||] b with
          | Error e -> Error ("native: " ^ e)
          | Ok o -> C.observed o))

let report ?driver c =
  Fmt.pr "%s@." (C.verdict ?driver ~run:(native ?mutation:None) c)

let%expect_test "banded chains run natively equal the whole chain" =
  List.iter (fun c -> report c) C.chains;
  [%expect
    {|
    pointwise 3 stages, 8 rows, unrolled: equal
    pointwise 3 stages, 16 rows, looped: equal
    pointwise 3 stages, 12 rows, block 1, looped: equal
    pointwise then 3x3, 10 rows, unrolled: equal
    pointwise then 3x3, 10 rows, looped: equal
    3x3, pointwise, 3x3, 20 rows, looped: equal
    pointwise then 1x3 pad 1 along the columns, 14 rows, looped: equal
    pointwise then 3x1 pad 1 along the rows, 14 rows, looped: equal
    3x5 pad 1 by 2 then pointwise, 12 rows, looped: equal
    3x3 stride 2 by 1 then 5x3 stride 1 by 2 pad 1, 21 rows, looped: equal
    3x3 dilation 2 by 1, pad 2 by 0, 1x1 stride 1 by 2, 20 rows, looped: equal |}]

(* The driver's own defects reach the CPU as a wrong program: a window reading
   the first row of its ring for every row, bases that do not move between
   repetitions, a loop one repetition short. *)
let%expect_test "driver mutations are red natively" =
  let pw = List.nth C.chains 4 and looped = List.nth C.chains 1 in
  List.iter
    (fun (name, driver, c) ->
      Fmt.pr "== %s@." name;
      report ~driver c)
    Band_driver.
      [
        ("window offset dropped", Window_offset_dropped, pw);
        ("step dropped", Step_dropped, looped);
        ("iterations short", Iterations_short, looped);
      ];
  [%expect
    {|
    == window offset dropped
    pointwise then 3x3, 10 rows, looped: DIFFERS
    == step dropped
    pointwise 3 stages, 16 rows, looped: DIFFERS
    == iterations short
    pointwise 3 stages, 16 rows, looped: DIFFERS |}]

(* A defect in the code the CPU runs: a loop branch tested with the wrong sense,
   a symbol address missing its low bits. *)
let%expect_test "native code mutations are red" =
  let looped = List.nth C.chains 1 in
  List.iter
    (fun (name, mutation) ->
      Fmt.pr "== %s@." name;
      Fmt.pr "%s@." (C.verdict ~run:(native ~mutation) looped))
    Machine_rivet_aarch64.Rivet_a64_form.Mutation.
      [ ("branch sense", Branch_sense); ("lo12 dropped", Dropped_lo12) ];
  [%expect
    {|
    == branch sense
    pointwise 3 stages, 16 rows, looped: the run did not succeed
    == lo12 dropped
    pointwise 3 stages, 16 rows, looped: DIFFERS |}]

(* A defect in the generated contraction: the seed not read from C (the taps of
   a window accumulate into it), or the product and sum fused (one rounding in
   place of two). *)
let%expect_test "contraction mutations are red" =
  let pw = List.nth C.chains 4 in
  List.iter
    (fun (name, mutation) ->
      Fmt.pr "== %s@." name;
      Fmt.pr "%s@." (C.verdict ~run:(native ~mutation) pw))
    Machine_rivet_aarch64.Rivet_a64_form.Mutation.
      [
        ("seed dropped", Contraction_seed_dropped); ("fused", Contraction_fused);
      ];
  [%expect
    {|
    == seed dropped
    pointwise then 3x3, 10 rows, looped: DIFFERS
    == fused
    pointwise then 3x3, 10 rows, looped: DIFFERS |}]

(* Operands and rounding orders the route does not realize are typed refusals. *)
let%expect_test "unrealized contractions are refused" =
  let module K = Contraction in
  let make format numerics =
    let op = { K.Operand.batch = 0L; outer = 4L; k = 1L } in
    match
      Err.payload
        (K.make ~batch:1L ~m:2L ~n:2L ~k:4L ~a:op ~b:op
           ~c:{ K.Output.batch = 0L; row = 2L; col = 1L }
           ~format ~seed:K.Seed.Zero ~numerics)
    with
    | Ok d -> d
    | Error _ -> assert false
  in
  List.iter
    (fun d ->
      Fmt.pr "%a@."
        (Fmt.result ~ok:(Fmt.any "realized")
           ~error:Machine_rivet_aarch64.Rivet_a64_refusal.pp)
        (Err.payload
           (Machine_rivet_aarch64.Rivet_a64_contraction.module_ [ d ])))
    [
      make K.Element.F32 K.Numerics.Ordered;
      make K.Element.F32 K.Numerics.Fused_sequential;
      make K.Element.Bf16 K.Numerics.Ordered;
      make
        (K.Element.I8 { a = K.Sign.Signed; b = K.Sign.Signed })
        K.Numerics.Ordered;
    ];
  [%expect
    {|
    realized
    no AArch64 contraction under fused_sequential numerics
    no AArch64 contraction over bf16 operands
    no AArch64 contraction over i8ss operands |}]
