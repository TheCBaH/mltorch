let%expect_test "the inspector shows the kernel the builders used" =
  Fidelity.run ();
  [%expect
    {|
    C loop reference: 415 invocations, 415 equal
    C loop simd_fp32_relaxed: 415 invocations, 415 equal
    C ssa planned: 415 invocations, 415 equal
    JS loop: 415 invocations, 415 equal
    JS ssa planned: 415 invocations, 415 equal
    WAT loop reference (0 with constant tables, rebased: skipped): 415 invocations, 415 equal
    WAT loop simd_fp32_relaxed (0 with constant tables, rebased: skipped): 415 invocations, 415 equal
    WAT ssa planned (0 with constant tables, rebased: skipped): 415 invocations, 415 equal |}]

(* The overlay summary of every invocation of the model: total, and cheap.
   MLTORCH_TIMING=1 prints the time, which varies, to stderr. *)
let%expect_test "every invocation of the model has a summary" =
  let b = Fidelity.bundle () in
  let t0 = Sys.time () in
  let ss = Lowering_view.Summary.of_bundle ~granted:(fun _ -> false) b in
  if Sys.getenv_opt "MLTORCH_TIMING" <> None then
    Printf.eprintf "summary: %d invocations in %.2fs\n" (List.length ss)
      (Sys.time () -. t0);
  let module S = Lowering_view.Summary in
  let count p = List.length (List.filter p ss) in
  Fmt.pr "invocations %d, refused %d, vectorized %d, cycles %d@."
    (List.length ss)
    (count (fun s -> s.S.ssa_refusal <> None))
    (count (fun s -> s.S.vectorized))
    (count (fun s -> s.S.coral_cycles <> None));
  [%expect {| invocations 415, refused 0, vectorized 205, cycles 415 |}]
