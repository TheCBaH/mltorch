(* CLI: verify or evaluate JSON op-specs against the ATen-vs-native harness.

   Each non-flag argument is a path to a spec file; "-", or no paths at all,
   reads one spec from stdin. A spec ({ target, seed, args }) is decoded, its
   tensor inputs are synthesized from the seed, and the op is run.

   Default mode compares the ATen and native outputs, printing the input
   tensors' shapes/distributions and the ATen output's shape + min/max/mean
   before the status line (Aten_spec_run.compare_report), and exits non-zero
   on any mismatch, error, or parse failure. With --print, the resulting
   output tensor(s) are printed from both paths instead of compared. With
   --eval, the op is run through ATen only (no native comparison at all) and
   its pretty-printed call, output shape(s), and status are printed — used
   for the per-node fixtures under test/data/ (see
   .ai/pt2_node_spec_design.md), whose promoted cram output is itself the
   golden reference.

   --walk N (only with --eval, for now) re-synthesizes the same spec from N
   independent seeds instead of just one, printing each step's configuration
   and a compact summary of the output tensor(s) — see
   Aten_spec_run.walk_eval. *)

type mode = Verify | Print | Eval

let read_source path =
  if path = "-" then In_channel.input_all stdin
  else In_channel.with_open_bin path In_channel.input_all

let handle ~mode ~walk path =
  Format.printf "=== %s ===@." path;
  match Jsont_bytesrw.decode_string Aten_op_spec.jsont (read_source path) with
  | Error e ->
      Format.printf "[spec] %s: parse error: %s@." path e;
      false
  | Ok spec -> (
      match (walk, mode) with
      | Some steps, Eval ->
          Aten_spec_run.walk_eval ~steps spec;
          true
      | Some _, (Print | Verify) ->
          prerr_endline "aten_spec_verify: --walk currently requires --eval";
          false
      | None, Print ->
          Aten_spec_run.eval_print spec;
          true
      | None, Eval ->
          Aten_spec_run.eval_report spec;
          true
      | None, Verify -> Aten_spec_run.compare_report spec)

(* Both self-gate on an env var and are safe to leave linked in unconditionally:
   [Memtrace.trace_if_requested] is a no-op unless MEMTRACE is set, and
   [Landmark.start_profiling] only matters if something was actually
   instrumented — which only happens under `dune build --profile landmarks`
   (see lib/native/dune and .ai/pt2_inference_perf.md's landmarks section);
   under the plain profile this is a harmless no-op (nothing to report; the
   ~2.3x instrumentation cost measured there lives entirely in the ppx, not
   in linking the landmarks runtime or calling start_profiling on
   uninstrumented code). *)
let () =
  Memtrace.trace_if_requested ();
  if Sys.getenv_opt "LANDMARKS" <> None then Landmark.start_profiling ();
  let mode = ref Verify and walk = ref None and paths = ref [] in
  let argv = Sys.argv in
  let rec parse i =
    if i < Array.length argv then
      let next =
        match argv.(i) with
        | "--print" ->
            mode := Print;
            i + 1
        | "--eval" ->
            mode := Eval;
            i + 1
        | "--walk" ->
            walk := Some (int_of_string argv.(i + 1));
            i + 2
        | a ->
            paths := a :: !paths;
            i + 1
      in
      parse next
  in
  parse 1;
  (* No file arguments means read a single spec from stdin. *)
  let sources = match List.rev !paths with [] -> [ "-" ] | ps -> ps in
  let ok =
    List.fold_left
      (fun acc p -> handle ~mode:!mode ~walk:!walk p && acc)
      true sources
  in
  if not ok then exit 1
