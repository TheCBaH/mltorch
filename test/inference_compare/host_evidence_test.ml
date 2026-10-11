open Inference_compare_schema
module H = Inference_compare_host
module Core = Inference_compare_core
module G = Experiment_gen

let rec rm_rf path =
  match (Unix.lstat path).st_kind with
  | exception _ -> ()
  | S_DIR ->
      Array.iter (fun f -> rm_rf (Filename.concat path f)) (Sys.readdir path);
      Unix.rmdir path
  | _ -> Unix.unlink path

let with_dir f =
  let base =
    Filename.concat
      (Filename.get_temp_dir_name ())
      (Printf.sprintf "ic-test-%d-%d" (Unix.getpid ()) (Random.bits ()))
  in
  Unix.mkdir base 0o755;
  Fun.protect ~finally:(fun () -> rm_rf base) (fun () -> f base)

let prepare ?(opts = G.default_options) ~scenario dir =
  let exp = Filename.concat dir "exp" in
  G.generate ~dir:exp ~scenario opts;
  let ( let* ) = Result.bind in
  let* loaded = H.Session.load (Filename.concat exp "experiment.json") in
  let* v = H.Session.validate loaded in
  H.Session.prepare v

let get = function
  | Ok x -> x
  | Error e -> Format.kasprintf failwith "unexpected: %a" H.Session.pp_error e

let run_id s = Result.get_ok (Id.Run.of_string s)
let hashers = H.Session.hashers

let%expect_test
    "a failed event write publishes nothing but the run is still finalized" =
  with_dir (fun dir ->
      let p = get (prepare ~scenario:"agree" dir) in
      let ev =
        Result.get_ok
          (H.Evidence.create ~root:(Filename.concat dir "ev") ~hashers p
             ~run_id:(run_id "r"))
      in
      (* let four events through; the fifth (the commit of unit A) fails *)
      H.Evidence.Fault.fail_events_after ev 4;
      let cfg =
        H.Run.config p ~emit:(H.Evidence.emit ev) ~retain:(H.Evidence.retain ev)
          ~mode:Full ~ledger_limit:1_000_000L
      in
      let r =
        Core.Coordinator.run cfg p.sealed ~left:p.left_endpoint
          ~right:p.right_endpoint
      in
      Format.printf "failure: %s@."
        (match r.failure with
        | Some f -> (
            match Failure.encode_code f.code with
            | Jsont.String (s, _) -> s
            | _ -> "?")
        | None -> "none");
      Format.printf "published units: %d, finals: %d@." (List.length r.executed)
        (List.length r.finals);
      (match H.Evidence.finalize ev r with
      | Ok () -> Format.printf "finalized@."
      | Error e -> Format.printf "finalize: %a@." H.Evidence.pp_error e);
      let opened =
        Result.get_ok (H.Evidence_read.open_ ~hashers (H.Evidence.directory ev))
      in
      Format.printf "recorded completion: %s@."
        (match opened.completion.execution with
        | Complete -> "complete"
        | Failed -> "failed"
        | Refused -> "refused"
        | Interrupted -> "interrupted");
      let events = Result.get_ok (H.Evidence_read.events opened) in
      Format.printf "events on disk: %d (the refused one is absent)@."
        (List.length events));
  [%expect
    {|
    failure: evidence_failed
    published units: 0, finals: 0
    finalized
    recorded completion: failed
    events on disk: 4 (the refused one is absent) |}]

let%expect_test
    "an interrupted run is incomplete, whatever else is in the directory" =
  with_dir (fun dir ->
      let p = get (prepare ~scenario:"agree" dir) in
      let ev =
        Result.get_ok
          (H.Evidence.create ~root:(Filename.concat dir "ev") ~hashers p
             ~run_id:(run_id "r"))
      in
      let cfg =
        H.Run.config p ~emit:(H.Evidence.emit ev) ~retain:(H.Evidence.retain ev)
          ~mode:Full ~ledger_limit:1_000_000L
      in
      let r =
        Core.Coordinator.run cfg p.sealed ~left:p.left_endpoint
          ~right:p.right_endpoint
      in
      List.iter
        (fun (_, v) -> Inference_compare_codecs.Logical_value.drop v)
        r.finals;
      (* the process "dies" before finalization *)
      H.Evidence.abandon ev;
      let dir = H.Evidence.directory ev in
      Format.printf "events written: %b, summary present: %b@."
        (Sys.file_exists (Filename.concat dir "events.jsonl"))
        (Sys.file_exists (Filename.concat dir "summary.json"));
      (match H.Evidence_read.open_ ~hashers dir with
      | Error e ->
          Format.printf "open: %a@."
            (fun ppf e ->
              Fmt.string ppf
                (match e with
                | H.Evidence_read.Incomplete _ ->
                    "incomplete (no completion record)"
                | _ -> "other"))
            e
      | Ok _ -> Format.printf "open: accepted@.");
      match H.Replay.full ~run_dir:dir with
      | Error (H.Replay.Evidence (Incomplete _)) ->
          Format.printf "replay: refused, incomplete@."
      | _ -> Format.printf "replay: not refused@.");
  [%expect
    {|
    events written: true, summary present: false
    open: incomplete (no completion record)
    replay: refused, incomplete |}]

let%expect_test "run directories are exclusive" =
  with_dir (fun dir ->
      let p = get (prepare ~scenario:"agree" dir) in
      let root = Filename.concat dir "ev" in
      let first = H.Evidence.create ~root ~hashers p ~run_id:(run_id "r") in
      Format.printf "first: %s@."
        (match first with Ok _ -> "created" | Error _ -> "error");
      (match H.Evidence.create ~root ~hashers p ~run_id:(run_id "r") with
      | Error (H.Evidence.Duplicate_run _) ->
          Format.printf "second: duplicate refused@."
      | Error _ -> Format.printf "second: other error@."
      | Ok _ -> Format.printf "second: created@.");
      match first with Ok e -> H.Evidence.abandon e | Error _ -> ());
  [%expect {|
    first: created
    second: duplicate refused |}]

let%expect_test
    "inputs that cannot be kept inside the run directory are refused" =
  with_dir (fun dir ->
      let exp = Filename.concat dir "exp" in
      G.generate ~dir:exp ~scenario:"agree" G.default_options;
      (* point the left graph at an absolute path *)
      let path = Filename.concat exp "experiment.json" in
      let text = Result.get_ok (H.File_io.read path) in
      let abs = Filename.concat exp "left.graph.json" in
      let patched =
        Str.global_replace
          (Str.regexp_string "\"left.graph.json\"")
          (Printf.sprintf "%S" abs) text
      in
      ignore (H.File_io.write_atomic path patched);
      let p =
        get
          (Result.bind (H.Session.load path) (fun l ->
               Result.bind (H.Session.validate l) H.Session.prepare))
      in
      match
        H.Evidence.create ~root:(Filename.concat dir "ev") ~hashers p
          ~run_id:(run_id "r")
      with
      | Error (H.Evidence.Unportable_path _) ->
          Format.printf "refused: unportable path@."
      | Error _ -> Format.printf "other error@."
      | Ok e ->
          H.Evidence.abandon e;
          Format.printf "created@.");
  [%expect {| refused: unportable path |}]

let%expect_test "invalid experiments fail before any engine is touched" =
  with_dir (fun dir ->
      (match prepare ~scenario:"unmapped" dir with
      | Error (H.Session.Structure _) -> Format.printf "structure error@."
      | Error _ -> Format.printf "other error@."
      | Ok _ -> Format.printf "prepared@.");
      let exp = Filename.concat dir "exp2" in
      G.generate ~dir:exp ~scenario:"agree" G.default_options;
      ignore (H.File_io.write_atomic (Filename.concat exp "x.bin") "short");
      (match H.Session.load (Filename.concat exp "experiment.json") with
      | Error (H.Session.Case _) ->
          Format.printf "case payload rejected: wrong length@."
      | Error _ -> Format.printf "other error@."
      | Ok _ -> Format.printf "loaded@.");
      ignore
        (H.File_io.write_atomic
           (Filename.concat exp "mapping.json")
           "{\"schema_version\":\"2\"}");
      match H.Session.load (Filename.concat exp "experiment.json") with
      | Error (H.Session.Decode _) ->
          Format.printf "mapping rejected: decode error@."
      | Error _ -> Format.printf "other error@."
      | Ok _ -> Format.printf "loaded@.");
  [%expect
    {|
    structure error
    case payload rejected: wrong length
    mapping rejected: decode error |}]

let%expect_test "a replay is judged by the events it produces" =
  with_dir (fun dir ->
      let p = get (prepare ~scenario:"deviate" dir) in
      let ev =
        Result.get_ok
          (H.Evidence.create ~root:(Filename.concat dir "ev") ~hashers p
             ~run_id:(run_id "r"))
      in
      let cfg =
        H.Run.config p ~emit:(H.Evidence.emit ev) ~retain:(H.Evidence.retain ev)
          ~mode:Full ~ledger_limit:1_000_000L
      in
      let r =
        Core.Coordinator.run cfg p.sealed ~left:p.left_endpoint
          ~right:p.right_endpoint
      in
      List.iter
        (fun (_, v) -> Inference_compare_codecs.Logical_value.drop v)
        r.finals;
      ignore (H.Evidence.finalize ev r);
      let opened =
        Result.get_ok (H.Evidence_read.open_ ~hashers (H.Evidence.directory ev))
      in
      let events = Result.get_ok (H.Evidence_read.events opened) in
      Format.printf "identical streams: %d difference(s)@."
        (List.length (H.Replay.differences ~recorded:events ~replayed:events));
      (* a different instance numbering is not a difference *)
      let renumbered =
        List.map
          (fun (e : Event.t) -> { e with sequence = Int64.add e.sequence 100L })
          events
      in
      Format.printf "renumbered: %d@."
        (List.length
           (H.Replay.differences ~recorded:events ~replayed:renumbered));
      (* a computed output with another digest is *)
      let other =
        Result.get_ok
          (Digest.Logical_value.make Inference_compare_sha256.algorithm
             ~hex:(String.make 64 '0'))
      in
      let altered =
        List.map
          (fun (e : Event.t) ->
            match e.payload with
            | Graph_completed { side = Right; outputs }
              when e.unit = Some (Plan_fixtures.uid "A") ->
                {
                  e with
                  payload =
                    Graph_completed
                      {
                        side = Right;
                        outputs =
                          List.map
                            (fun (o : Computed_value.t) ->
                              { o with logical_digest = other })
                            outputs;
                      };
                }
            | _ -> e)
          events
      in
      let d = H.Replay.differences ~recorded:events ~replayed:altered in
      Format.printf "one output digest changed: %d difference(s)@."
        (List.length d);
      (* a verdict flipped from mismatch to pass *)
      let flipped =
        List.map
          (fun (e : Event.t) ->
            match e.payload with
            | Outputs_compared rs when e.unit = Some (Plan_fixtures.uid "A") ->
                {
                  e with
                  payload =
                    Outputs_compared
                      (List.map
                         (fun (r : Comparison_result.t) ->
                           { r with verdict = Pass })
                         rs);
                }
            | _ -> e)
          events
      in
      Format.printf "a verdict flipped: %d difference(s)@."
        (List.length (H.Replay.differences ~recorded:events ~replayed:flipped));
      Format.printf "a missing event: %d difference(s)@."
        (List.length
           (H.Replay.differences ~recorded:events ~replayed:(List.tl events))));
  [%expect
    {|
    identical streams: 0 difference(s)
    renumbered: 0
    one output digest changed: 1 difference(s)
    a verdict flipped: 1 difference(s)
    a missing event: 15 difference(s) |}]
