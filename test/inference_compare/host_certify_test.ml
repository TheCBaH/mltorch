(* Stage gating of the two-pass certification under injected faults: a fault
   at any stage ends the workflow, later stages never run and no golden record
   appears. Faults are injected through the workflow's `before` hook, which
   touches the filesystem just before a stage starts. *)

module G = Experiment_gen
module H = Inference_compare_host

let counter = ref 0

let fresh () =
  incr counter;
  let dir =
    Filename.concat
      (Filename.get_temp_dir_name ())
      (Printf.sprintf "certify-%d-%d" (Unix.getpid ()) !counter)
  in
  dir

let generate ?(scenario = "agree") () =
  let dir = fresh () in
  G.generate ~dir ~scenario
    { G.default_options with certify = Some G.default_certification };
  dir

let exists dir rel = Sys.file_exists (Filename.concat dir rel)

let report ~dir = function
  | Ok (c : H.Certify.certified) ->
      Printf.printf "certified (%s)\n"
        (match c.record.qualification with
        | Exact -> "exact"
        | Within_declared_contract -> "declared")
  | Error (f : H.Certify.failure) ->
      let text = Format.asprintf "%a" H.Certify.pp_failure f in
      Printf.printf "failed: %s, exit %d\n"
        (Str.global_replace (Str.regexp_string dir) "DIR" text)
        (H.Certify.exit_code f)

let state ~dir =
  let root = Filename.concat dir "evidence" in
  List.iter
    (fun rel -> Printf.printf "%s: %b\n" rel (exists root rel))
    [
      "run-exp-p1";
      "run-exp-p2";
      "run-exp-full";
      "cert-exp/golden";
      "cert-exp/golden/golden.json";
    ]

let certify ?hooks dir =
  H.Certify.run ?hooks
    ~root:(Filename.concat dir "evidence")
    ~workflow_path:(Filename.concat dir "validation.json")
    ()

let replace_in file a b =
  let ic = open_in_bin file in
  let s = really_input_string ic (in_channel_length ic) in
  close_in ic;
  let i = Str.search_forward (Str.regexp_string a) s 0 in
  let s' =
    String.sub s 0 i ^ b
    ^ String.sub s (i + String.length a) (String.length s - i - String.length a)
  in
  let oc = open_out_bin file in
  output_string oc s';
  close_out oc

let%expect_test "a clean certification writes its golden record" =
  let dir = generate () in
  report ~dir (certify dir);
  state ~dir;
  [%expect
    {|
    certified (exact)
    run-exp-p1: true
    run-exp-p2: true
    run-exp-full: true
    cert-exp/golden: true
    cert-exp/golden/golden.json: true |}]

let%expect_test "a changed engine between the passes cannot join the stages" =
  let dir = generate () in
  let hooks =
    {
      H.Certify.before =
        (function
        | H.Certify.Tested_trajectory ->
            replace_in
              (Filename.concat dir "experiment.json")
              "\"name\":\"right\"" "\"name\":\"right2\""
        | _ -> ());
    }
  in
  report ~dir (certify ~hooks dir);
  state ~dir;
  [%expect
    {|
    failed: identity_continuity: the plan, the engines or the inputs changed between the passes, exit 2
    run-exp-p1: true
    run-exp-p2: false
    run-exp-full: false
    cert-exp/golden: false
    cert-exp/golden/golden.json: false |}]

let%expect_test "a second pass that cannot start ends the workflow" =
  let dir = generate () in
  let hooks =
    {
      H.Certify.before =
        (function
        | H.Certify.Tested_trajectory ->
            Unix.mkdir (Filename.concat dir "evidence/run-exp-p2") 0o755
        | _ -> ());
    }
  in
  report ~dir (certify ~hooks dir);
  state ~dir;
  [%expect
    {|
    failed: tested_trajectory: run directory DIR/evidence/run-exp-p2 already exists, exit 3
    run-exp-p1: true
    run-exp-p2: true
    run-exp-full: false
    cert-exp/golden: false
    cert-exp/golden/golden.json: false |}]

let%expect_test "changed inputs before the complete-graph run are refused" =
  let dir = generate () in
  let hooks =
    {
      H.Certify.before =
        (function
        | H.Certify.Full_graph_confirmation ->
            let oc = open_out_bin (Filename.concat dir "x.bin") in
            output_string oc (String.make 24 '\001');
            close_out oc
        | _ -> ());
    }
  in
  report ~dir (certify ~hooks dir);
  state ~dir;
  [%expect
    {|
    failed: identity_continuity: the engines or the inputs of the complete-graph run differ from the passes', exit 2
    run-exp-p1: true
    run-exp-p2: true
    run-exp-full: false
    cert-exp/golden: false
    cert-exp/golden/golden.json: false |}]

let%expect_test "a golden that cannot be written fails the workflow" =
  let dir = generate () in
  let hooks =
    {
      H.Certify.before =
        (function
        | H.Certify.Golden ->
            let oc =
              open_out_bin (Filename.concat dir "evidence/cert-exp/golden")
            in
            close_out oc
        | _ -> ());
    }
  in
  report ~dir (certify ~hooks dir);
  state ~dir;
  [%expect
    {|
    failed: golden_materialization: DIR/evidence/cert-exp/golden: File exists, exit 3
    run-exp-p1: true
    run-exp-p2: true
    run-exp-full: true
    cert-exp/golden: true
    cert-exp/golden/golden.json: false |}]
