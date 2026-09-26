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
         [--strict] [--nodes] [--shadow] [--direct]
   Same four positional paths as js/run/pt2_run.ml -- deliberately runs only
   the FIRST of the (many) samples they hold ([~max_samples:(Some 1)]),
   never the whole map: see [Infer_report.report]'s own doc for why that
   parameter exists.

   [--nodes]/[--shadow] are this runner's own, stripped from
   [Sys.argv] before [Infer_report.parse_argv] ever sees it: that parser's
   [take_flags] rejects any flag it does not recognize, and it is shared with
   every other [Infer_report] caller, none of which knows [Node_executor]
   exists. [--nodes] installs [Loop_node_executor] alongside the
   Region-authored one, unconditionally still present; [--shadow] turns on
   its bitwise shadow check. One executor for the whole process, matching
   [region_executor]'s own top-level scope: safe because every sample here
   shares the SAME graph (compile-table ids are unique only within one
   graph, design §4.2: sharing one table across graphs
   can collide). *)

type eval = [ Native_interp.error | Native_predict.error ]

let pp_eval ppf : eval -> unit = function
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
let infer_canonical ~node_executor archive image =
  let open Err.Syntax in
  let t0 = Sys.time () in
  let* (Native_interp.Transformed t as transformed) =
    Native_interp.transform archive ~preload:true
      ~passes:[ Pipeline.canonical ~fold:true ]
    |> Err.map_error ~pos:__POS__ (fun e -> (e :> eval))
  in
  let t1 = Sys.time () in
  let* outputs, _loaded =
    Native_interp.evaluate ~region_executor ~region_group_executor
      ?node_executor archive transformed ~input:image
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

let infer ~direct ~node_executor archive image =
  if not direct then infer_canonical ~node_executor archive image
  else
    let open Err.Syntax in
    let* outputs =
      Native_interp.run ~region_executor ~region_group_executor ?node_executor
        archive ~input:image
      |> Err.map_error ~pos:__POS__ (fun e -> (e :> eval))
    in
    let* top =
      Native_predict.top_predictions outputs 5
      |> Err.map_error ~pos:__POS__ (fun e -> (e :> eval))
    in
    Err.return
      (List.map (fun ((c : Dim.index Dim.t), p) -> ((c :> int), p)) top)

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
        Infer_report.run ~max_samples:1 ~now:Sys.time
          ~infer:(infer ~direct ~node_executor)
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
      let region_min_generated_js = if nodes then 0 else min_generated_js in
      match
        ( Loop_region_executor.Coverage.check
            ~min_generated_js:region_min_generated_js region_coverage,
          Loop_region_executor.Coverage.check ~min_generated_js:0
            region_group_coverage,
          node_check )
      with
      | Error msg, _, _ | _, Error msg, _ | _, _, Error msg ->
          Format.eprintf "loop_js_pt2: %s@." msg;
          exit 1
      | Ok (), Ok (), Ok () -> (
          match report_result with
          | Ok () -> ()
          | Error e ->
              Format.eprintf "%a@."
                (Err.Error.pp (Infer_report.pp_error pp_eval))
                e;
              exit 1))
