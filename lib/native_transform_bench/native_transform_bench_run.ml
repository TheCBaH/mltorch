(* Replicates the plumbing [Native_interp_exec.transform_lowered] hides behind
   one call — building the captured-constant store, running the canonical
   pipeline, packing, and building the PT2 lens — as four separately timed
   steps over intermediate states the public function does not expose. The
   PASS ORDER itself is never duplicated: [Pipeline.canonical] is the one
   source of that.

   Every measured run creates a fresh origin, per the plan: an [Id_supply]
   watermark and a [Constant_store] are cheap to rebuild and doing so is what
   lets a later run's allocation start from the same origin watermark as the
   first, rather than drifting with whichever run happened to go first. *)

module Tensor_id = Graph_ir.Tensor_id

type times = {
  origin_s : float;
  canonical_s : float;
  pack_s : float;
  lens_s : float;
  total_s : float;
}

(* The three states a comparison artifact must cover: before any pass
   ran, after the canonical pipeline but before packing, and the final packed
   graph — each paired with the allocator watermark at that point. *)
type state_dump = {
  label : string;
  graph : Graph_ir.graph;
  constants : Tensor.packed Tensor_id.Map.t;
  constant_store : Constant_store.t;
  allocator_pp : string;
}

type result = {
  states : state_dump list; (* origin, pre_pack, packed, in that order *)
  composed_map_pp : string;
  derived : (Tensor_id.t * string list) list;
  times : times;
}

let packed_state r = List.nth r.states 2

(* Small tensors inlined directly in model.json (not read from an archive) —
   the same lookup [transform_lowered] performs, over the same
   [captured_targets] map, before any pass runs. *)
let constant_store_of (lowered : Pt2_native_graph.t) =
  let open Err.Syntax in
  let source = lowered.Pt2_native_graph.graph in
  Tensor_id.Map.fold
    (fun id target acc ->
      let* store = acc in
      match Tensor_id.Map.find_opt id source.Graph_ir.Graph.tensors with
      | None -> Err.return store
      | Some tensor ->
          Constant_store.bind_captured store ~tensor
            (Const_ssa.Capture.of_string target))
    lowered.Pt2_native_graph.captured_targets
    (Err.return Constant_store.empty)
  |> Err.or_raise ~pp_error:Constant_store.pp_error

let dump_state label (state : 'v Rewrite.t) =
  {
    label;
    graph = Rewrite.graph state;
    constants = Rewrite.constants state;
    constant_store = Rewrite.constant_store state;
    allocator_pp =
      Fmt.to_to_string Rewrite.pp_allocator (Rewrite.allocator state);
  }

(* One fresh, fully timed pass over [lowered]: origin, canonical pipeline,
   pack, lens — in that order, matching [Native_interp_exec.transform_lowered]
   exactly. Raises on any step's error: a benchmark run must fail loudly
   rather than silently skip a model.

   [now] is supplied by the caller rather than hardcoded to
   [Unix.gettimeofday]: this keeps [unix] out of this library's dependency
   closure, so a test can link it (the coverage tests) without dragging in
   a library the JS inline-test mode of test/native cannot build against. *)
let run ~now (lowered : Pt2_native_graph.t) =
  let source = lowered.Pt2_native_graph.graph in
  let t0 = now () in
  let constant_store = constant_store_of lowered in
  let (Rewrite.Origin origin) =
    Rewrite.origin ~constant_store source
    |> Err.or_raise ~pp_error:Rewrite.pp_error
  in
  let t1 = now () in
  let (Rewrite.Step (rewritten, rewrite_map)) =
    Pass.run_all origin [ Pipeline.canonical ~fold:false ]
    |> Err.or_raise ~pp_error:Pass.pp_error
  in
  let t2 = now () in
  let (Rewrite.Step (packed, pack_map)) =
    Rewrite.pack rewritten |> Err.or_raise ~pp_error:Rewrite.pp_error
  in
  let t3 = now () in
  let composed_map = Graph_map.compose rewrite_map pack_map in
  let lens =
    Pt2_native_graph.lens lowered ~src:origin composed_map ~dst:packed
    |> Err.or_raise ~pp_error:Pt2_native_graph.pp_lens_error
  in
  let t4 = now () in
  let packed_graph = Rewrite.graph packed in
  (* Constants with no archive path of their own — a folded weight — named by
     the PT2 tensors they derive from. Mirrors
     [Native_interp_exec.derivations] exactly; that function is private to
     the library, so the artifact needs its own copy to see the intermediate
     [packed] graph and lens this driver (not [transform_lowered]) built. *)
  let derived =
    List.fold_left
      (fun acc id ->
        if Graph_ir.input_kind packed_graph id <> Graph_ir.Input.Constant then
          acc
        else
          match
            Pt2_native_graph.captured_target lens id
            |> Err.or_raise ~pp_error:Pt2_native_graph.pp_lens_error
          with
          | Some _ -> acc
          | None -> (
              let names =
                List.filter_map
                  (fun src ->
                    match
                      Tensor_id.Map.find_opt src
                        lowered.Pt2_native_graph.tensor_origins
                    with
                    | Some (Pt2_native_graph.Source o) ->
                        Some o.Pt2_native_graph.Tensor_origin.ssa_name
                    | Some Pt2_native_graph.Derived | None -> None)
                  (Pt2_native_graph.provenance_sources lens id)
              in
              match names with [] -> acc | _ -> (id, names) :: acc))
      [] packed_graph.Graph_ir.Graph.inputs
    |> List.rev
  in
  {
    states =
      [
        dump_state "origin" origin;
        dump_state "pre_pack" rewritten;
        dump_state "packed" packed;
      ];
    composed_map_pp = Fmt.to_to_string Graph_map.pp composed_map;
    derived;
    times =
      {
        origin_s = t1 -. t0;
        canonical_s = t2 -. t1;
        pack_s = t3 -. t2;
        lens_s = t4 -. t3;
        total_s = t4 -. t0;
      };
  }

(* --- per-stage timing within the canonical pipeline ---------------------- *)

type stage_sample = { name : string; seconds : float }

type staged_result = {
  stages : stage_sample list; (* [Pipeline.canonical_stages]' order *)
  total_canonical_s : float; (* sum of [stages]; excludes the agreement check *)
  agrees_with_composite : bool;
      (* the staged graph and composed map equal a fresh run of
         [Pipeline.canonical] itself, checked by printed comparison *)
}

(* Runs [Pipeline.canonical_stages] one stage at a time from a fresh origin,
   timing each; then runs the ordinary [Pipeline.canonical] composite from
   that SAME origin (untimed here — that measurement is [run]'s job) purely
   to check agreement — [Rewrite.t] is immutable, so replaying from the same
   origin twice is safe and makes the comparison apples-to-apples.
   [Pipeline.canonical_stages] is the same list [canonical_with_trace]
   composes into one [Pass.sequence], so this never duplicates the pass
   order; it only duplicates EXECUTING it, which is the point — a
   [Pass.sequence]'s own overhead is not assumed identical to running its
   members one at a time (verification/trace scope can differ), which is
   exactly what [agrees_with_composite] checks for this graph. *)
let stage_run ~now (lowered : Pt2_native_graph.t) =
  let source = lowered.Pt2_native_graph.graph in
  let constant_store = constant_store_of lowered in
  let (Rewrite.Origin origin) =
    Rewrite.origin ~constant_store source
    |> Err.or_raise ~pp_error:Rewrite.pp_error
  in
  let stages = Pipeline.canonical_stages ~on_materialized_fold:(fun _ -> ()) in
  let Rewrite.Step (staged_state, staged_map), stage_samples =
    List.fold_left
      (fun (Rewrite.Step (state, map), acc) (stage : Pass.t) ->
        let s0 = now () in
        let (Rewrite.Step (next, step_map)) =
          Pass.run_all state [ stage ] |> Err.or_raise ~pp_error:Pass.pp_error
        in
        let s1 = now () in
        ( Rewrite.Step (next, Graph_map.compose map step_map),
          { name = stage.Pass.name; seconds = s1 -. s0 } :: acc ))
      (Rewrite.Step (origin, Graph_map.identity), [])
      stages
  in
  let stage_samples = List.rev stage_samples in
  let (Rewrite.Step (composite_state, composite_map)) =
    Pass.run_all origin [ Pipeline.canonical ~fold:false ]
    |> Err.or_raise ~pp_error:Pass.pp_error
  in
  let agrees_with_composite =
    String.equal
      (Fmt.to_to_string Graph_ir.pp (Rewrite.graph staged_state))
      (Fmt.to_to_string Graph_ir.pp (Rewrite.graph composite_state))
    && String.equal
         (Fmt.to_to_string Graph_map.pp staged_map)
         (Fmt.to_to_string Graph_map.pp composite_map)
  in
  {
    stages = stage_samples;
    total_canonical_s =
      List.fold_left (fun acc s -> acc +. s.seconds) 0. stage_samples;
    agrees_with_composite;
  }
