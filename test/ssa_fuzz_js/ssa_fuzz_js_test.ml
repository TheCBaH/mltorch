open Ssa_fuzz.Fuzz

(* The JavaScript emitter over a quarter of the interpreter seeds (500): the
   engine compiles each program in process, which is cheap. *)

let js_seeds = max 1 (seeds / 4)
let js = Fuzz_js.backend

(* Ssa_js has no form for a binary64 fused multiply-add or an int64 to binary32
   conversion, and says so with a typed error. Programs without them are driven
   through it and must agree; a refusal on this profile is a defect. *)
let profile = { fma = false; i64_to_f32 = false }

let%expect_test "the JavaScript emitter agrees with the interpreter" =
  let refused = ref 0 in
  for s = 0 to js_seeds - 1 do
    let case = generate ~profile ~seed:(Int64.of_int s) ~size:default_size () in
    (match js.run case with Refused _ -> incr refused | Ran _ -> ());
    match check case [ js ] with
    | Ok _ -> ()
    | Error d -> Fmt.failwith "%a" pp_disagreement d
  done;
  Fmt.pr "%d seeds, refused: %d@." js_seeds !refused;
  [%expect {| 500 seeds, refused: 0 |}]

(* The capability gaps stay visible: with every operation, some seeds are refused
   for each of the two reasons, instead of being silently skipped. *)
let%expect_test
    "the operations the JavaScript emitter has no form for are refused with a \
     typed error" =
  let reasons = Hashtbl.create 4 in
  for s = 0 to 199 do
    match js.run (generate ~seed:(Int64.of_int s) ~size:default_size ()) with
    | Refused why -> Hashtbl.replace reasons why ()
    | Ran _ -> ()
  done;
  Hashtbl.to_seq_keys reasons
  |> List.of_seq |> List.sort compare |> List.iter print_endline;
  [%expect
    {|
    a binary64 fused multiply-add has no JavaScript form
    an int64 to binary32 conversion has no JavaScript form |}]
