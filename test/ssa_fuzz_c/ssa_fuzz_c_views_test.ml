open Ssa_fuzz.Fuzz

(* The view-call half of the C fuzz, in its own file so the inline-test timing
   gate measures it apart from the plain-program run. *)

let c_seeds = max 1 (seeds / 10)
let timed = Fuzz_c.timed
let fail_on = Fuzz_c.fail_on

(* The expansion of a view call, which the C emitter refuses by name, runs
   through it and agrees with the interpreter on the view call itself. *)
let%expect_test "the C emitter runs the expansion of view calls" =
  let c = expanded Fuzz_c.backend in
  let refused = ref 0 in
  timed (fun () ->
      for s = 0 to (c_seeds / 2) - 1 do
        let case = generate_views ~seed:(Int64.of_int s) ~size:3 () in
        (match c.run case with Refused _ -> incr refused | Ran _ -> ());
        fail_on (check case [ c ])
      done);
  Fmt.pr "no refusals: %b@." (!refused = 0);
  [%expect {| no refusals: true |}]
