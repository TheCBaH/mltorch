open Ssa_fuzz.Fuzz

(* The Wasm emitter over a twentieth of the interpreter seeds (100): each costs a
   module build and a node start. A runaway generator or a slow node fails the
   run, never skips it. *)

let wasm_seeds = max 1 (seeds / 20)
let time_cap_s = 300.
let wasm = Fuzz_wasm.backend

let timed f =
  let t0 = Unix.gettimeofday () in
  let r = f () in
  let dt = Unix.gettimeofday () -. t0 in
  if dt > time_cap_s then Fmt.failwith "over the time cap: %.1fs" dt;
  r

(* Ssa_wasm has no form for a binary64 fused multiply-add, and says so with a
   typed error. Programs without one are driven through it and must agree; a
   refusal on this profile is a defect. *)
let profile = { full with fma = false }

let%expect_test "the Wasm emitter agrees with the interpreter" =
  let refused = ref 0 in
  timed (fun () ->
      for s = 0 to wasm_seeds - 1 do
        let case =
          generate ~profile ~seed:(Int64.of_int s) ~size:default_size ()
        in
        (match wasm.run case with Refused _ -> incr refused | Ran _ -> ());
        match check case [ wasm ] with
        | Ok _ -> ()
        | Error d -> Fmt.failwith "%a" pp_disagreement d
      done);
  Fmt.pr "%d seeds, refused: %d@." wasm_seeds !refused;
  [%expect {| 100 seeds, refused: 0 |}]

(* The capability gap stays visible: with every operation, some seeds are
   refused for it, instead of being silently skipped. *)
let%expect_test
    "the operation the Wasm emitter has no form for is refused with a typed \
     error" =
  let reasons = Hashtbl.create 4 in
  for s = 0 to 99 do
    match wasm.run (generate ~seed:(Int64.of_int s) ~size:default_size ()) with
    | Refused why -> Hashtbl.replace reasons why ()
    | Ran _ -> ()
  done;
  Hashtbl.to_seq_keys reasons
  |> List.of_seq |> List.sort compare |> List.iter print_endline;
  [%expect {| a binary64 fused multiply-add has no WebAssembly form |}]
