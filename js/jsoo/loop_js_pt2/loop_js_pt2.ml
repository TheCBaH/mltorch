(* The whole-model verification this plan exists for: a real downloaded
   model's Region-authored nodes run through Loop_js_exec-compiled
   JavaScript, in-process, under node, checked against the release's own
   top-5 rankings. No native counterpart -- js_of_ocaml-only by
   construction, since [Loop_region_executor] is (see
   js/jsoo/loop_js_exec_js/dune). MANUAL, like `jsoo.pt2.run`: a single
   fastvit_sa12 sample measured at ~103s natively makes even one sample,
   let alone the full ten `results.json` holds, unsuitable for a blocking
   CI step.

   argv: <model.pt2> <inputs.pt> <expected.json> <outputs.pt> [--cram]
         [--strict] [--nodes] [--shadow] [--direct] [--arena] [--bundle] [--baseline] [--warm]
   Same four positional paths as js/run/pt2_run.ml -- deliberately runs only
   the FIRST of the (many) samples they hold ([~max_samples:(Some 1)]),
   never the whole map: see [Infer_report.report]'s own doc for why that
   parameter exists.

   [--nodes]/[--shadow]/[--arena] are this runner's own, stripped from
   [Sys.argv] before [Infer_report.parse_argv] ever sees it: that parser's
   [take_flags] rejects any flag it does not recognize, and it is shared with
   every other [Infer_report] caller, none of which knows [Node_executor]
   exists. [--nodes] installs [Loop_node_executor] alongside the
   Region-authored one, unconditionally still present; [--shadow] turns on
   its bitwise shadow check. One executor for the whole process, matching
   [region_executor]'s own top-level scope: safe because every sample here
   shares the SAME graph (compile-table ids are unique only within one
   graph, design §4.2: sharing one table across graphs
   can collide).

   [--arena] forces [Native_interp]'s own [?arena:Admission.Best_effort] for
   this run, the jsoo/node counterpart of native_graph eval's own [--arena]
   (see the tensor arena design in .ai/). Outputs are bit-identical either
   way, so a decline would otherwise be invisible here: it is reported (one
   line, "arena: used ..." or "arena: declined: ...") and fails the run, same
   as a coverage or ranking failure. *)

type eval =
  [ Native_interp.error
  | Native_predict.error
  | Loop_bundle.error
  | Loop_bundle_exec.error
  | `Bundle_mismatch of int ]

let pp_eval ppf : eval -> unit = function
  | `Bundle_mismatch n ->
      Format.fprintf ppf "bundle: %d output(s) differ bitwise from per-node" n
  | #Loop_bundle_exec.error as e -> Loop_bundle_exec.pp_error ppf e
  | #Loop_bundle.error as e -> Loop_bundle.pp_error ppf e
  | #Native_predict.error as e -> Native_predict.pp_error ppf e
  | #Native_interp.error as e -> Native_interp.pp_error ppf e

let region_coverage = Loop_region_executor.Coverage.create ()
let region_executor = Loop_region_executor.make region_coverage

(* A separate coverage counter from [region_coverage]: [Lstm] is the only
   Region-authored multi-output op, and neither target model this runner
   downloads has one (T7.2's own finding), so this stays at 0/0 on every
   real run here -- the walk-scale executor test
   (region_group_executor_test.ml) is where the group path actually gets
   exercised. Wired anyway, matching [node_executor]/[region_executor]'s
   own unconditional presence, so a future model with an Lstm needs no
   change here. *)
let region_group_coverage = Loop_region_executor.Coverage.create ()

let region_group_executor =
  Loop_region_executor.make_group region_group_coverage

(* Canonical is the DEFAULT (2026-09-23, follow-up to the 2026-09-22
   [--canonical] flag this replaces): runs the same generated-JS path over
   [Pipeline.canonical]'s output instead of the raw imported graph -- DCE
   plus permute-cancellation/layout-normalization plus constant/batch-norm
   folding (see [Pipeline.canonical_with_trace]'s own doc), the same pass
   list [to4d]/[transform] use, not fusion. Neither
   [Node_executor]/[Loop_node_executor] nor [Loop_node_program] needed any
   change for this: both take a bare [Graph_ir.graph], indifferent to how
   it was produced, so [Native_interp.evaluate] on a [transformed] graph
   reuses this file's SAME [node_executor]/[region_executor]/
   [region_group_executor] values unmodified. Two timings, printed to
   stderr (never stdout, which the cram tests compare exactly): the
   transform pass alone, and evaluation alone on its result -- so
   "canonicalize once, run many times" and "one-shot, transform included"
   are both readable from one run rather than conflated into a single
   number.

   [--direct] opts back into the raw, untransformed graph -- kept for the
   smaller subset of models/targets that still exercise the direct
   PT2-to-Native conversion path itself, not only its canonicalized
   result (see the Makefile's own loop_js.pt2.run). *)
(* Set by [on_arena] when [--arena] asked for one and it was declined, so
   [main] can fail the run for that reason too, not only a wrong ranking --
   see native_graph_eval.ml's own [--arena] for the same reasoning on the
   native side. *)
let arena_declined = ref false

let pp_arena_outcome ppf (outcome : Arena_run.Outcome.t) =
  match outcome with
  | Used report ->
      Format.fprintf ppf "arena: used %a" Arena_run.Report.pp report
  | Declined e ->
      arena_declined := true;
      Format.fprintf ppf "arena: declined: %a" Arena_run.pp_error e

let on_arena outcome = Format.printf "%a@." pp_arena_outcome outcome

let infer_canonical ~arena ~node_executor archive image =
  let open Err.Syntax in
  let t0 = Sys.time () in
  let* (Native_interp.Transformed t as transformed) =
    Native_interp.transform archive ~preload:true
      ~passes:[ Pipeline.canonical ~fold:true ]
    |> Err.map_error ~pos:__POS__ (fun e -> (e :> eval))
  in
  let t1 = Sys.time () in
  let* outputs, _loaded =
    Native_interp.evaluate
      ?arena:(if arena then Some Arena.Admission.Best_effort else None)
      ?on_arena:(if arena then Some on_arena else None)
      ~region_executor ~region_group_executor ?node_executor archive transformed
      ~input:image
    |> Err.map_error ~pos:__POS__ (fun e -> (e :> eval))
  in
  let t2 = Sys.time () in
  Printf.eprintf
    "loop_js_pt2 canonical: nodes %d -> %d, canonicalize %.1f ms, evaluate \
     (inference only) %.1f ms, total %.1f ms\n\
     %!"
    t.nodes_before
    (List.length t.graph.Graph_ir.Graph.nodes)
    ((t1 -. t0) *. 1000.)
    ((t2 -. t1) *. 1000.)
    ((t2 -. t0) *. 1000.);
  let* top =
    Native_predict.top_predictions outputs 5
    |> Err.map_error ~pos:__POS__ (fun e -> (e :> eval))
  in
  Err.return (List.map (fun ((c : Dim.index Dim.t), p) -> ((c :> int), p)) top)

let infer ~direct ~arena ~warm ~node_executor archive image =
  if not direct then infer_canonical ~arena ~node_executor archive image
  else
    let open Err.Syntax in
    let run_once () =
      Native_interp.run
        ?arena:(if arena then Some Arena.Admission.Best_effort else None)
        ?on_arena:(if arena then Some on_arena else None)
        ~region_executor ~region_group_executor ?node_executor archive
        ~input:image
      |> Err.map_error ~pos:__POS__ (fun e -> (e :> eval))
    in
    let* outputs = run_once () in
    (* [--warm]: five further runs on the same process (kernels already
       compiled), the per-node counterpart of [--bundle]'s warm repeats. *)
    let* () =
      if not warm then Err.return ()
      else
        let+ times =
          Err.List.map
            (fun _ ->
              let w0 = Sys.time () in
              let+ _ = run_once () in
              (Sys.time () -. w0) *. 1000.)
            [ 1; 2; 3; 4; 5 ]
        in
        Printf.eprintf "loop_js_pt2 per-node: warm runs ms [%s]\n%!"
          (String.concat "; "
             (List.map (Printf.sprintf "%.1f") (List.sort compare times)))
    in
    let* top =
      Native_predict.top_predictions outputs 5
      |> Err.map_error ~pos:__POS__ (fun e -> (e :> eval))
    in
    Err.return
      (List.map (fun ((c : Dim.index Dim.t), p) -> ((c :> int), p)) top)

(* [--bundle]: the whole model as ONE generated-JS entry call over prepared
   arena pools ([Loop_bundle_exec]), on the direct graph. [--shadow] also runs
   [Eval_direct.run] on the same graph and inputs and requires every output to
   be bitwise equal. Preparation and the run are timed separately, to stderr. *)
let output_values t =
  let (Tensor.Tensor tt) = t in
  let acc = ref [] in
  Vec6.iter tt.Tensor.shape (fun c ->
      acc := Int32.bits_of_float (Tensor.read_at t (Vec6.get c)) :: !acc);
  !acc

(* Lowered, preloaded and prepared once: every sample shares the one graph, as
   the per-node executors' tables do. [first] marks the call that paid for it. *)
let bundle_cache = ref None

let prepared_bundle archive =
  let open Err.Syntax in
  let map e = (e :> eval) in
  match !bundle_cache with
  | Some cached -> Err.return (false, cached)
  | None ->
      let* lowered = Native_interp.lower_archive archive |> Err.map_error map in
      let g = lowered.Pt2_native_graph.graph in
      let* constants =
        Native_interp.preload archive lowered |> Err.map_error map
      in
      let* b = Loop_bundle.build g |> Err.map_error map in
      let t0 = Sys.time () in
      let* p =
        Loop_bundle_exec.prepare b ~constants:(fun id ->
            Graph_ir.Tensor_id.Map.find_opt id constants)
        |> Err.map_error map
      in
      let cached = (g, constants, b, p, (Sys.time () -. t0) *. 1000.) in
      bundle_cache := Some cached;
      Err.return (true, cached)

let infer_bundle ~shadow ~baseline archive image =
  let open Err.Syntax in
  let map e = (e :> eval) in
  let* first, (g, constants, b, p, prepare_ms) = prepared_bundle archive in
  let* input = Native_interp.tensor_of_pt2 image |> Err.map_error map in
  let t1 = Sys.time () in
  let input_id = List.hd b.Loop_bundle.inputs in
  let* out =
    Loop_bundle_exec.run p ~bind:(fun id ->
        if Graph_ir.Tensor_id.equal id input_id then Some input else None)
    |> Err.map_error map
  in
  let t2 = Sys.time () in
  (* Warm repeats on the same prepared bundle, the run above being the first
     (cold) one; reported as the sorted list so the spread is visible. *)
  let st = Loop_bundle_exec.stats p in
  if first then
    Printf.eprintf
      "loop_js_pt2 bundle: source %d bytes, %d distinct kernels, %d execution \
       set(s), pools: %s\n\
       %!"
      st.Loop_bundle_exec.source_bytes st.Loop_bundle_exec.distinct_kernels
      st.Loop_bundle_exec.execution_sets
      (String.concat ", "
         (List.map
            (fun (a, k, n) ->
              Format.asprintf "%a/%a=%d" Storage_script.Arena_id.pp a
                Alloc_script.Kind.pp k n)
            st.Loop_bundle_exec.pools));
  let* warm =
    Err.List.map
      (fun _ ->
        let w0 = Sys.time () in
        let+ _ =
          Loop_bundle_exec.run p ~bind:(fun id ->
              if Graph_ir.Tensor_id.equal id input_id then Some input else None)
          |> Err.map_error map
        in
        (Sys.time () -. w0) *. 1000.)
      (if first then [ 1; 2; 3; 4; 5 ] else [])
  in
  Printf.eprintf
    "loop_js_pt2 bundle: %d invocations, prepare %.1f ms, first run %.1f ms, \
     warm runs ms [%s]\n\
     %!"
    (List.length b.Loop_bundle.invocations)
    prepare_ms
    ((t2 -. t1) *. 1000.)
    (String.concat "; "
       (List.map (Printf.sprintf "%.1f") (List.sort compare warm)));
  (* [--baseline]: the matched per-node comparison -- the same graph, constants
     and input through [Eval_direct.run] with every node in its own generated-JS
     kernel, constants passed in (no re-binding from the archive) -- one cold run
     then five warm ones, so the two figures differ only in per-node dispatch
     against one entry call. *)
  let* () =
    if not (baseline && first) then Err.return ()
    else
      let node_state = Loop_node_executor.create () in
      let node_executor = Loop_node_executor.node_executor node_state in
      let run_once () =
        let t = Sys.time () in
        let+ _ =
          Eval_direct.run ~region_executor ~region_group_executor ~node_executor
            g
            ~constants:(Graph_ir.Tensor_id.Map.bindings constants)
            ~inputs:[ (input_id, input) ]
          |> Err.map_error map
        in
        (Sys.time () -. t) *. 1000.
      in
      let* cold = run_once () in
      let+ warm = Err.List.map (fun _ -> run_once ()) [ 1; 2; 3; 4; 5 ] in
      Printf.eprintf
        "loop_js_pt2 per-node baseline (constants passed in): first run %.1f \
         ms, warm runs ms [%s]\n\
         %!"
        cold
        (String.concat "; "
           (List.map (Printf.sprintf "%.1f") (List.sort compare warm)))
  in
  let outputs =
    List.map
      (fun id -> Graph_ir.Tensor_id.Map.find id out)
      g.Graph_ir.Graph.outputs
  in
  let* () =
    if not shadow then Err.return ()
    else
      let* reference =
        Eval_direct.run g
          ~constants:(Graph_ir.Tensor_id.Map.bindings constants)
          ~inputs:[ (input_id, input) ]
        |> Err.map_error map
      in
      let bad =
        List.length
          (List.filter
             (fun id ->
               output_values (Graph_ir.Tensor_id.Map.find id out)
               <> output_values (Graph_ir.Tensor_id.Map.find id reference))
             g.Graph_ir.Graph.outputs)
      in
      Printf.eprintf "loop_js_pt2 bundle: shadow %d/%d outputs differ\n%!" bad
        (List.length g.Graph_ir.Graph.outputs);
      if bad = 0 then Err.return () else Err.fail (`Bundle_mismatch bad)
  in
  let* top = Native_predict.top_predictions outputs 5 |> Err.map_error map in
  Err.return (List.map (fun ((c : Dim.index Dim.t), p) -> ((c :> int), p)) top)

(* T3.3/T5.1: a run that silently skipped the generated-JS path is a worse
   defect than one that fails loudly, since nothing else about a passing
   ranking check would tell them apart. Checked BEFORE the ranking result
   is even inspected, so a coverage failure is never masked by (or
   mistaken for) a ranking one. *)
let min_generated_js = 1

let strip_flag flag argv =
  ( Array.exists (String.equal flag) argv,
    Array.of_list
      (List.filter (fun a -> not (String.equal a flag)) (Array.to_list argv)) )

(* One row per op kind seen in any of the three buckets, sorted, so the
   table's order does not depend on hashtable iteration. *)
let print_node_coverage (coverage : Loop_node_executor.Coverage.t) =
  let keys tbl = Hashtbl.fold (fun k _ acc -> k :: acc) tbl [] in
  let op_kinds =
    keys coverage.generated_js @ keys coverage.fallback @ keys coverage.pending
    |> List.sort_uniq String.compare
  in
  let count tbl k = Option.value ~default:0 (Hashtbl.find_opt tbl k) in
  Format.printf "loop_js_pt2: node coverage@.";
  List.iter
    (fun k ->
      Format.printf "  %-28s generated_js=%d fallback=%d pending=%d@." k
        (count coverage.generated_js k)
        (count coverage.fallback k)
        (count coverage.pending k))
    op_kinds

let () =
  let nodes, argv = strip_flag "--nodes" Sys.argv in
  let shadow, argv = strip_flag "--shadow" argv in
  let direct, argv = strip_flag "--direct" argv in
  let arena, argv = strip_flag "--arena" argv in
  let bundle, argv = strip_flag "--bundle" argv in
  let baseline, argv = strip_flag "--baseline" argv in
  let warm, argv = strip_flag "--warm" argv in
  (* [--samples=N]: [--bundle]'s one prepared graph serves N samples (default
     1, as every other mode here). *)
  let samples, argv =
    let is_flag a = String.length a > 10 && String.sub a 0 10 = "--samples=" in
    let n =
      Array.fold_left
        (fun acc a ->
          if is_flag a then
            int_of_string (String.sub a 10 (String.length a - 10))
          else acc)
        1 argv
    in
    ( n,
      Array.of_list
        (List.filter (fun a -> not (is_flag a)) (Array.to_list argv)) )
  in
  match Infer_report.parse_argv argv with
  | Error usage ->
      prerr_endline usage;
      exit 2
  | Ok (paths, options) -> (
      let node_state =
        if nodes then Some (Loop_node_executor.create ~shadow ()) else None
      in
      let node_executor =
        Option.map Loop_node_executor.node_executor node_state
      in
      let report_result =
        Infer_report.run ~max_samples:samples ~now:Sys.time
          ~infer:
            (if bundle then infer_bundle ~shadow ~baseline
             else infer ~direct ~arena ~warm ~node_executor)
          paths options
      in
      (match node_state with
      | Some { Loop_node_executor.coverage; _ } -> print_node_coverage coverage
      | None -> ());
      Format.printf "loop_js_pt2: region coverage generated_js=%d fallback=%d@."
        region_coverage.Loop_region_executor.Coverage.generated_js
        region_coverage.Loop_region_executor.Coverage.fallback;
      Format.printf
        "loop_js_pt2: region group coverage generated_js=%d fallback=%d@."
        region_group_coverage.Loop_region_executor.Coverage.generated_js
        region_group_coverage.Loop_region_executor.Coverage.fallback;
      let node_check =
        match node_state with
        | None -> Ok ()
        | Some { Loop_node_executor.coverage; _ } ->
            Loop_node_executor.Coverage.check ~min_generated_js coverage
      in
      (* The Region floor is 0 under [--nodes]: that flag's whole point is
         making a model with no Region-authored op (mobilenetv2_050, T0.5)
         meaningful for the first time, and requiring it to also clear the
         Region floor would fail every such model regardless of how well
         [--nodes] itself did. Unconditional (no [--nodes]) behavior is
         unchanged: fastvit_sa12's own SDPA nodes still must clear it.

         [region_group_coverage]'s own floor is unconditionally 0 (T7.2):
         neither downloaded model has an Lstm, so requiring it to clear the
         same floor as the solo Region path would fail every run here
         regardless of how well everything else did -- the group path's
         own real exercise is region_group_executor_test.ml's walk-scale
         subject, not this runner. Checked anyway (D7's "one report"), so a
         future model with an Lstm needs no change here to start gating on
         it too. *)
      (* [--bundle] compiles every scheduled invocation before running, or refuses at
         preparation, so completeness is checked by construction and the Region
         floor has nothing to count. *)
      let region_min_generated_js =
        if nodes || bundle then 0 else min_generated_js
      in
      let arena_check =
        if arena && !arena_declined then
          Error "--arena asked for a Best_effort arena and it was declined"
        else Ok ()
      in
      match
        ( Loop_region_executor.Coverage.check
            ~min_generated_js:region_min_generated_js region_coverage,
          Loop_region_executor.Coverage.check ~min_generated_js:0
            region_group_coverage,
          node_check,
          arena_check )
      with
      | Error msg, _, _, _
      | _, Error msg, _, _
      | _, _, Error msg, _
      | _, _, _, Error msg ->
          Format.eprintf "loop_js_pt2: %s@." msg;
          exit 1
      | Ok (), Ok (), Ok (), Ok () -> (
          match report_result with
          | Ok () -> ()
          | Error e ->
              Format.eprintf "%a@."
                (Err.Error.pp (Infer_report.pp_error pp_eval))
                e;
              exit 1))
