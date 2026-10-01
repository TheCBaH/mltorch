(* The canonical-transform performance benchmark. Decodes and lowers each
   model once outside the timed interval, then times [Pipeline.canonical]
   (via [Native_transform_bench_run.run], which reruns from a fresh origin
   every measured sample), with origin/canonical/pack/lens as separate
   boundaries plus an end-to-end total.  No payload preload, no verification,
   for the primary timing run — matching production's payload-free path
   (`native_graph`, the model-support sweep, the Model Explorer export, the
   arena evaluation) that this benchmark exists to speed up.

   JSONL output, one row per sample plus one summary row per model; an
   optional per-model text artifact for comparison. Never a
   golden: this tool's timings are not asserted against in any test. *)

module Run = Native_transform_bench_run
module Artifact = Native_transform_bench_artifact
module Manifest = Native_transform_bench_manifest
module J = Jsont.Json

(* --- corpus -----------------------------------------------------------
   Same shape as [bin/arena_alloc_eval.ml]'s manifest/validate_manifest: the
   pinned producer submodule's tracked model.json files when in a git work
   tree, a directory scan otherwise (a fixture tree is not one). Duplicated
   rather than shared: each tool's corpus check is a few lines and the two
   binaries have no other reason to depend on each other. *)

module Entry = struct
  type t = { model : string; path : string; md5 : string }
end

let command_output prog args =
  let out = Filename.temp_file "native_transform_bench" ".out" in
  Fun.protect
    ~finally:(fun () -> Sys.remove out)
    (fun () ->
      let cmd =
        Filename.quote_command prog args ~stdout:out ~stderr:"/dev/null"
      in
      if Sys.command cmd = 0 then
        Some (In_channel.with_open_bin out In_channel.input_all)
      else None)

let lines s = String.split_on_char '\n' s |> List.filter (( <> ) "")

let manifest models_dir =
  let model_of path =
    match String.split_on_char '/' path with
    | [ model; "models"; "model.json" ] -> Some model
    | _ -> None
  in
  let models =
    match
      command_output "git"
        [ "-C"; models_dir; "ls-files"; "--"; "*/models/model.json" ]
    with
    | Some out when lines out <> [] -> List.filter_map model_of (lines out)
    | _ ->
        Sys.readdir models_dir |> Array.to_list
        |> List.filter (fun m ->
            Sys.file_exists
              (Filename.concat models_dir
                 (Filename.concat m "models/model.json")))
  in
  let models = List.sort String.compare models in
  List.map
    (fun model ->
      let path =
        Filename.concat models_dir (Filename.concat model "models/model.json")
      in
      let md5 =
        if Sys.file_exists path then Digest.to_hex (Digest.file path) else ""
      in
      { Entry.model; path; md5 })
    models

let validate_manifest ~expected (entries : Entry.t list) =
  let models = List.map (fun (e : Entry.t) -> e.model) entries in
  let rec dup = function
    | a :: (b :: _ as rest) -> if String.equal a b then Some a else dup rest
    | _ -> None
  in
  match
    (dup models, List.find_opt (fun (e : Entry.t) -> e.md5 = "") entries)
  with
  | Some m, _ -> Error (Fmt.str "duplicate model %s" m)
  | None, Some e -> Error (Fmt.str "missing %s" e.Entry.path)
  | None, None ->
      if List.length entries <> expected then
        Error
          (Fmt.str "expected %d models, found %d" expected (List.length entries))
      else Ok ()

(* --- per-model measurement ---------------------------------------------- *)

let words () =
  let s = Gc.quick_stat () in
  s.Gc.minor_words +. s.Gc.major_words

let median xs =
  let ys = Array.of_list xs in
  Array.sort Float.compare ys;
  ys.(Array.length ys / 2)

let min_ l = List.fold_left Float.min infinity l
let max_ l = List.fold_left Float.max neg_infinity l

type sample = { times : Run.times; words : float }

let lower_or_raise bytes =
  match
    Jsont_bytesrw.decode_string Pytorch_types.ExportedProgram.jsont bytes
  with
  | Error msg -> failwith (Fmt.str "model.json decode: %s" msg)
  | Ok program -> (
      match Native_interp.lower program with
      | Error e ->
          failwith
            (Fmt.str "lower: %a" Native_interp.pp_error (Err.Error.kind e))
      | Ok lowered -> lowered)

let measure ~warmup ~repeats lowered =
  for _ = 1 to warmup do
    ignore (Run.run ~now:Unix.gettimeofday lowered)
  done;
  List.init repeats (fun _ ->
      Gc.full_major ();
      let before = words () in
      let r = Run.run ~now:Unix.gettimeofday lowered in
      let after = words () in
      ({ times = r.times; words = after -. before }, r))

(* --- JSONL rows ----------------------------------------------------------- *)

let encode json =
  match Jsont_bytesrw.encode_string ~format:Jsont.Minify Jsont.json json with
  | Ok line -> line
  | Error msg -> failwith msg

let sample_row ~model ~md5 ~nodes ~repeat (s : sample) =
  J.object'
    [
      J.mem (J.name "kind") (J.string "sample");
      J.mem (J.name "model") (J.string model);
      J.mem (J.name "model_md5") (J.string md5);
      J.mem (J.name "repeat") (J.number (float_of_int repeat));
      J.mem (J.name "nodes") (J.number (float_of_int nodes));
      J.mem (J.name "origin_s") (J.number s.times.origin_s);
      J.mem (J.name "canonical_s") (J.number s.times.canonical_s);
      J.mem (J.name "pack_s") (J.number s.times.pack_s);
      J.mem (J.name "lens_s") (J.number s.times.lens_s);
      J.mem (J.name "total_s") (J.number s.times.total_s);
      J.mem (J.name "gc_words") (J.number s.words);
    ]

let stage_row ~model ~md5 ~index (s : Run.stage_sample) =
  J.object'
    [
      J.mem (J.name "kind") (J.string "stage");
      J.mem (J.name "model") (J.string model);
      J.mem (J.name "model_md5") (J.string md5);
      J.mem (J.name "stage_index") (J.number (float_of_int index));
      J.mem (J.name "stage_name") (J.string s.Run.name);
      J.mem (J.name "seconds") (J.number s.Run.seconds);
    ]

let stage_summary_row ~model ~md5 (r : Run.staged_result) =
  J.object'
    [
      J.mem (J.name "kind") (J.string "stage_summary");
      J.mem (J.name "model") (J.string model);
      J.mem (J.name "model_md5") (J.string md5);
      J.mem (J.name "total_canonical_s") (J.number r.Run.total_canonical_s);
      J.mem
        (J.name "agrees_with_composite")
        (J.bool r.Run.agrees_with_composite);
    ]

let summary_row ~model ~md5 ~nodes (samples : sample list) =
  let field f =
    let xs = List.map f samples in
    J.object'
      [
        J.mem (J.name "median") (J.number (median xs));
        J.mem (J.name "min") (J.number (min_ xs));
        J.mem (J.name "max") (J.number (max_ xs));
      ]
  in
  J.object'
    [
      J.mem (J.name "kind") (J.string "summary");
      J.mem (J.name "model") (J.string model);
      J.mem (J.name "model_md5") (J.string md5);
      J.mem (J.name "nodes") (J.number (float_of_int nodes));
      J.mem (J.name "repeats") (J.number (float_of_int (List.length samples)));
      J.mem (J.name "origin_s") (field (fun s -> s.times.origin_s));
      J.mem (J.name "canonical_s") (field (fun s -> s.times.canonical_s));
      J.mem (J.name "pack_s") (field (fun s -> s.times.pack_s));
      J.mem (J.name "lens_s") (field (fun s -> s.times.lens_s));
      J.mem (J.name "total_s") (field (fun s -> s.times.total_s));
    ]

(* --- driver --------------------------------------------------------------- *)

let run ~models_dir ~models ~expected ~warmup ~repeats ~output ~artifacts_dir
    ~dune_profile ~stages =
  let entries = manifest models_dir in
  (* [~models], when given, is an iteration subset: validate the FULL
     corpus first regardless, so a subset run still fails loudly on a moved
     or missing model rather than silently narrowing the cohort by typo. *)
  (match validate_manifest ~expected entries with
  | Ok () -> ()
  | Error msg ->
      Fmt.epr "native_transform_bench: %s@." msg;
      exit 1);
  let entries =
    match models with
    | None -> entries
    | Some names -> (
        let by_name =
          List.filter
            (fun (e : Entry.t) -> List.mem e.Entry.model names)
            entries
        in
        match
          List.find_opt
            (fun name ->
              not
                (List.exists
                   (fun (e : Entry.t) -> String.equal e.Entry.model name)
                   entries))
            names
        with
        | Some missing ->
            Fmt.epr "native_transform_bench: unknown model %s@." missing;
            exit 1
        | None -> by_name)
  in
  Option.iter
    (fun d -> if not (Sys.file_exists d) then Unix.mkdir d 0o755)
    artifacts_dir;
  let repo_root =
    match command_output "git" [ "rev-parse"; "--show-toplevel" ] with
    | Some out -> ( match lines out with r :: _ -> r | [] -> ".")
    | None -> "."
  in
  let manifest_env =
    Manifest.create ~repo_root ~producer_path:"pytorch-image-models"
      ~dune_profile ~verify_policy:"none (primary timing run)"
      ~payload_policy:"payload-free (no archive, no preload)" ~warmup ~repeats
  in
  Out_channel.with_open_bin output (fun oc ->
      List.iter
        (fun (e : Entry.t) ->
          let bytes =
            In_channel.with_open_bin e.Entry.path In_channel.input_all
          in
          let lowered = lower_or_raise bytes in
          let samples_and_results = measure ~warmup ~repeats lowered in
          let samples = List.map fst samples_and_results in
          let last_result = snd (List.nth samples_and_results (repeats - 1)) in
          let nodes =
            List.length
              (Graph_ir.nodes (Run.packed_state last_result).Run.graph)
          in
          List.iteri
            (fun i s ->
              Out_channel.output_string oc
                (encode
                   (sample_row ~model:e.Entry.model ~md5:e.Entry.md5 ~nodes
                      ~repeat:i s));
              Out_channel.output_char oc '\n')
            samples;
          Out_channel.output_string oc
            (encode
               (summary_row ~model:e.Entry.model ~md5:e.Entry.md5 ~nodes samples));
          Out_channel.output_char oc '\n';
          if stages then (
            (* A single staged run per model — attribution, not statistical
                timing, is the point here; [measure] above already gives the
                repeated end-to-end numbers. Fails loudly if the staged
                execution disagrees with the ordinary composite: that would
                mean this benchmark's own stage breakdown cannot be trusted,
                which is worse than not having one. *)
            let staged = Run.stage_run ~now:Unix.gettimeofday lowered in
            if not staged.Run.agrees_with_composite then
              failwith
                (Fmt.str
                   "%s: staged execution disagrees with Pipeline.canonical"
                   e.Entry.model);
            List.iteri
              (fun i s ->
                Out_channel.output_string oc
                  (encode
                     (stage_row ~model:e.Entry.model ~md5:e.Entry.md5 ~index:i s));
                Out_channel.output_char oc '\n')
              staged.Run.stages;
            Out_channel.output_string oc
              (encode
                 (stage_summary_row ~model:e.Entry.model ~md5:e.Entry.md5 staged));
            Out_channel.output_char oc '\n');
          Out_channel.flush oc;
          Option.iter
            (fun dir ->
              let path = Filename.concat dir (e.Entry.model ^ ".txt") in
              Out_channel.with_open_bin path (fun oc ->
                  Out_channel.output_string oc (Artifact.to_string last_result)))
            artifacts_dir;
          let m = median (List.map (fun s -> s.times.Run.total_s) samples) in
          Fmt.pr "%-28s nodes=%-6d median_total_s=%.3f@." e.Entry.model nodes m)
        entries);
  let manifest_path = output ^ ".manifest.json" in
  Out_channel.with_open_bin manifest_path (fun oc ->
      Out_channel.output_string oc (Manifest.to_string manifest_env));
  Fmt.pr "wrote %s (%d models) and %s@." output (List.length entries)
    manifest_path

(* --- command line --------------------------------------------------------- *)

open Cmdliner

let term =
  let models_dir =
    Arg.(
      required
      & opt (some dir) None
      & info [ "models-dir" ] ~docv:"DIR"
          ~doc:"The directory of <model>/models/model.json files.")
  and expected =
    Arg.(
      required
      & opt (some int) None
      & info [ "expected-models" ] ~docv:"N"
          ~doc:"The number of models the corpus must hold.")
  and models =
    Arg.(
      value
      & opt (some (list string)) None
      & info [ "models" ] ~docv:"NAME,..."
          ~doc:
            "Iteration subset, by model name; default is the full corpus (the \
             full-corpus mode). The corpus is still validated against \
             $(b,--expected-models) regardless.")
  and warmup = Arg.(value & opt int 1 & info [ "warmup" ] ~docv:"N")
  and repeats = Arg.(value & opt int 5 & info [ "repeats" ] ~docv:"N")
  and output =
    Arg.(
      required
      & opt (some string) None
      & info [ "output" ] ~docv:"FILE" ~doc:"JSONL samples + summaries.")
  and artifacts_dir =
    Arg.(
      value
      & opt (some string) None
      & info [ "artifacts" ] ~docv:"DIR"
          ~doc:
            "Write one deterministic comparison-artifact text file per model.")
  and dune_profile =
    Arg.(
      value & opt string "release"
      & info [ "dune-profile" ] ~docv:"PROFILE"
          ~doc:"Recorded in the manifest only; this tool does not select it.")
  and stages =
    Arg.(
      value & flag
      & info [ "stages" ]
          ~doc:
            "Also time each canonical stage separately, checked against the \
             ordinary composite run.")
  in
  let make models_dir models expected warmup repeats output artifacts_dir
      dune_profile stages =
    if warmup < 0 || repeats < 1 then begin
      Fmt.epr "native_transform_bench: --repeats >= 1, --warmup >= 0@.";
      2
    end
    else begin
      run ~models_dir ~models ~expected ~warmup ~repeats ~output ~artifacts_dir
        ~dune_profile ~stages;
      0
    end
  in
  Term.(
    const make $ models_dir $ models $ expected $ warmup $ repeats $ output
    $ artifacts_dir $ dune_profile $ stages)

(* Self-gated exactly like bin/aten_spec_verify.ml: a harmless no-op under
   the plain profile (nothing is instrumented to report), function-level
   attribution under `dune build --profile landmarks LANDMARKS=1`. Always uninstrumented for the timings this tool actually
   reports — a landmarks build is ~2.3x slower even idle (lib/native/dune),
   so accept performance conclusions only from a plain-profile run. *)
let () =
  if Sys.getenv_opt "LANDMARKS" <> None then Landmark.start_profiling ();
  (match Err_host.install_from_env () with
  | Ok (_ : Err.Config.t option) -> ()
  | Error e ->
      Fmt.epr "native_transform_bench: %a@." Err_host.pp_error
        (Err.Error.kind e);
      exit 124);
  exit
    (Cmd.eval'
       (Cmd.v
          (Cmd.info "native_transform_bench"
             ~doc:"Benchmark the canonical Native transform pipeline.")
          term))
