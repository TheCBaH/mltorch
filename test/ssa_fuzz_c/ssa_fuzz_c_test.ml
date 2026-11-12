open Ssa_fuzz.Fuzz

(* The C emitter over a tenth of the interpreter seeds: each costs a [gcc] run.
   A runaway generator or a slow compiler fails the run, never skips it. *)

let c_seeds = max 1 (seeds / 10)
let time_cap_s = 300.

let timed f =
  let t0 = Unix.gettimeofday () in
  let r = f () in
  let dt = Unix.gettimeofday () -. t0 in
  if dt > time_cap_s then Fmt.failwith "over the time cap: %.1fs" dt;
  r

let fail_on = function
  | Ok _ -> ()
  | Error d -> Fmt.failwith "%a" pp_disagreement d

let%expect_test "the C emitter agrees with the interpreter, or refuses" =
  let c = Fuzz_c.backend in
  let refused = ref 0 in
  timed (fun () ->
      for s = 0 to c_seeds - 1 do
        let case = generate ~seed:(Int64.of_int s) ~size:default_size () in
        (match c.run case with Refused _ -> incr refused | Ran _ -> ());
        fail_on (check case [ c ])
      done);
  (* A construct every seed refuses would be skipped, not tested. *)
  Fmt.pr "%d seeds, refusals under a tenth: %b@." c_seeds
    (!refused * 10 < c_seeds);
  [%expect {| 200 seeds, refusals under a tenth: true |}]
