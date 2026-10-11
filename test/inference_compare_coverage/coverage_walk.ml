(* Runs the rows of the coverage manifest against a real model through the pair
   driver and reports each as ok, pending (the engine is not built here or a
   prerequisite is missing, with the reason) or FAIL.

   argv: <manifest> <fixture_pair_run.exe> <model_dir> <inputs.pt>

   Exit status: 0 when no row failed (and, with MLTORCH_COVERAGE_STRICT, none is
   pending), 1 otherwise. A pending row is never counted as passed. *)

module M = Inference_compare_coverage.Manifest

let contains hay needle =
  let n = String.length needle and h = String.length hay in
  let rec go i = i + n <= h && (String.sub hay i n = needle || go (i + 1)) in
  n = 0 || go 0

let read path = In_channel.with_open_bin path In_channel.input_all

let first_line_with text needle =
  List.find_opt (fun l -> contains l needle) (String.split_on_char '\n' text)

(* The counts of the "comparisons: P pass, M mismatch, O other" line. *)
let comparisons text =
  match first_line_with text "comparisons: " with
  | None -> None
  | Some l -> (
      try
        Scanf.sscanf l "comparisons: %d pass, %d mismatch, %d other"
          (fun p m o -> Some (p, m, o))
      with Scanf.Scan_failure _ | End_of_file | Failure _ -> None)

type outcome = Ok_ | Pending of string | Fail of string

let run_row ~exe ~model ~inputs (r : M.row) =
  let out = Filename.temp_file "coverage" ".out" in
  let env =
    String.concat " "
      (List.map
         (fun (k, v) -> Printf.sprintf "%s=%s" k (Filename.quote v))
         r.env)
  in
  let cmd =
    Printf.sprintf "env %s %s safetensors %s %s 0 %s %s > %s 2>&1" env
      (Filename.quote exe) (Filename.quote model) (Filename.quote inputs)
      (Filename.quote r.left) (Filename.quote r.right) (Filename.quote out)
  in
  let code = Sys.command cmd in
  let text = read out in
  Sys.remove out;
  let expected_refusal =
    match r.expect with M.Refused _ -> true | _ -> false
  in
  match first_line_with text "is not available" with
  | Some l when code = 3 && not expected_refusal -> Pending (String.trim l)
  | _ -> (
      let verdict =
        match r.expect with
        | M.Agree -> (
            if code <> 0 then Error (Printf.sprintf "exit %d, wanted 0" code)
            else
              match comparisons text with
              | None -> Error "no comparison line"
              | Some (0, _, _) -> Error "no comparison was made"
              | Some (_, m, o) when m <> 0 || o <> 0 ->
                  Error "a mismatch or an incomparable pair"
              | Some _ -> Ok ())
        | M.Disagree ->
            if code <> 1 then Error (Printf.sprintf "exit %d, wanted 1" code)
            else Ok ()
        | M.Invalid ->
            if code <> 2 then Error (Printf.sprintf "exit %d, wanted 2" code)
            else Ok ()
        | M.Refused t ->
            if code <> 3 then Error (Printf.sprintf "exit %d, wanted 3" code)
            else if not (contains text t) then
              Error (Printf.sprintf "the refusal does not say %S" t)
            else Ok ()
      in
      match verdict with
      | Error m -> Fail m
      | Ok () -> (
          let failed =
            List.find_map
              (function
                | M.Has t ->
                    if contains text t then None
                    else Some (Printf.sprintf "output lacks %S" t)
                | M.Lacks t ->
                    if contains text t then
                      Some (Printf.sprintf "output has %S" t)
                    else None)
              r.checks
          in
          match failed with Some m -> Fail m | None -> Ok_))

let () =
  match Array.to_list Sys.argv with
  | [ _; manifest; exe; model; inputs ] -> (
      match M.parse (read manifest) with
      | Error m ->
          prerr_endline ("manifest: " ^ m);
          exit 2
      | Ok rows ->
          let strict = Sys.getenv_opt "MLTORCH_COVERAGE_STRICT" <> None in
          let ok = ref 0 and pending = ref 0 and failed = ref 0 in
          List.iter
            (fun (r : M.row) ->
              match run_row ~exe ~model ~inputs r with
              | Ok_ ->
                  incr ok;
                  Printf.printf "%s: ok\n%!" r.id
              | Pending why ->
                  incr pending;
                  Printf.printf "%s: pending (%s)\n%!" r.id why
              | Fail why ->
                  incr failed;
                  Printf.printf "%s: FAIL (%s)\n%!" r.id why)
            rows;
          Printf.printf "%d ok, %d pending, %d failed\n" !ok !pending !failed;
          exit (if !failed > 0 || (strict && !pending > 0) then 1 else 0))
  | _ ->
      prerr_endline
        "usage: coverage_walk MANIFEST FIXTURE_PAIR_RUN MODEL_DIR INPUTS_PT";
      exit 2
