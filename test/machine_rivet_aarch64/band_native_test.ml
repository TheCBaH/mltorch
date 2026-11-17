(* The band driver's chains on the CPU. Each banded program of the pointwise
   and window chains in the band-driver and view-call suites is lowered through
   the contraction helper route, selected, allocated and published for AArch64,
   run natively through Rivet, and its output bits compared with the
   whole-tensor chain on the SSA interpreter ({!Band_driver.whole}). The
   contraction itself is host C ({!Contract_host}); the native code is the
   driver's loops, every base's address arithmetic and the calls. *)

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
          match
            H.native ?mutation ~extra:Contract_host.extra case ~sites:[||] b
          with
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
    3x3, pointwise, 3x3, 20 rows, looped: equal |}]

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
