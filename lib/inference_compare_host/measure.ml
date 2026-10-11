(* What planning, preparing, running and replaying one experiment cost: the
   sizes the planner worked over, the time each phase took, the high-water mark
   of live payload bytes and the evidence left on disk. Measurement runs the
   experiment into a directory of its own; it claims nothing about the
   numbers' meaning beyond this host and this build. *)

open Inference_compare_schema
module Core = Inference_compare_core

type t = {
  left_nodes : int;
  right_nodes : int;
  left_tensors : int;
  right_tensors : int;
  units : int;
  comparison_sites : int;
  policy_bindings : int;
  rules : int;
  extracted_graphs : int;
  validate_seconds : float;
  prepare_seconds : float;
  run_seconds : float;
  replay_seconds : float;
  comparisons : int;
  elements_checked : int64;
  peak_live_bytes : int64;
  evidence_bytes : int64;
  evidence_files : int;
}

let ( let* ) = Result.bind

let rec du path =
  match (Unix.lstat path).Unix.st_kind with
  | Unix.S_DIR ->
      Array.fold_left
        (fun (b, n) f ->
          let b', n' = du (Filename.concat path f) in
          (Int64.add b b', n + n'))
        (0L, 0) (Sys.readdir path)
  | _ -> (Int64.of_int (Unix.lstat path).Unix.st_size, 1)

let timed f =
  let t0 = Unix.gettimeofday () in
  let r = f () in
  (r, Unix.gettimeofday () -. t0)

let error_text pp e = Format.asprintf "%a" pp e

let measure ~root ~experiment : (t, string) result =
  let session r = Result.map_error (error_text Session.pp_error) r in
  let* loaded = session (Session.load experiment) in
  let v, validate_seconds = timed (fun () -> Session.validate loaded) in
  let* v = session v in
  let p, prepare_seconds = timed (fun () -> Session.prepare v) in
  let* p = session p in
  let plan = Core.Sealed_plan.plan p.sealed in
  let* outcome, run_seconds =
    let r, s = timed (fun () -> Run.run ~root p) in
    match r with
    | Ok o -> Ok (o, s)
    | Error e -> Error (error_text Evidence.pp_error e)
  in
  let* () = Result.map_error (error_text Evidence.pp_error) outcome.finalized in
  let replay, replay_seconds =
    timed (fun () -> Replay.full ~run_dir:outcome.directory)
  in
  let* () =
    match replay with
    | Ok _ -> Ok ()
    | Error e -> Error (error_text Replay.pp_error e)
  in
  let evidence_bytes, evidence_files = du outcome.directory in
  let r = outcome.result in
  Ok
    {
      left_nodes = List.length loaded.left_graph.nodes;
      right_nodes = List.length loaded.right_graph.nodes;
      left_tensors = List.length loaded.left_graph.tensor_specs;
      right_tensors = List.length loaded.right_graph.tensor_specs;
      units = List.length loaded.mapping.units;
      comparison_sites = List.length v.sites;
      policy_bindings = List.length v.resolved.bindings;
      rules = List.length loaded.selection.rules;
      extracted_graphs = 2 * List.length plan.order;
      validate_seconds;
      prepare_seconds;
      run_seconds;
      replay_seconds;
      comparisons = List.length r.comparisons;
      elements_checked =
        List.fold_left
          (fun a (c : Comparison_result.t) -> Int64.add a c.elements_checked)
          0L r.comparisons;
      peak_live_bytes = outcome.peak_live_bytes;
      evidence_bytes;
      evidence_files;
    }

let pp ppf m =
  Fmt.pf ppf
    "@[<v>graphs: %d/%d nodes, %d/%d tensors (left/right)@,\
     plan: %d units, %d comparison sites, %d policy bindings, %d topology \
     rules, %d extracted graphs@,\
     validate %.3fs, prepare %.3fs (engines, extraction, sealing), run %.3fs, \
     replay %.3fs@,\
     compared: %d comparisons, %Ld elements@,\
     peak live payload %Ld bytes@,\
     evidence: %Ld bytes in %d files@]"
    m.left_nodes m.right_nodes m.left_tensors m.right_tensors m.units
    m.comparison_sites m.policy_bindings m.rules m.extracted_graphs
    m.validate_seconds m.prepare_seconds m.run_seconds m.replay_seconds
    m.comparisons m.elements_checked m.peak_live_bytes m.evidence_bytes
    m.evidence_files
