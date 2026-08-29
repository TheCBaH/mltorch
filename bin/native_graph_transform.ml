(* Split out of native_graph.ml, see native_graph_common.ml, and
   native_graph_args.ml. *)

open Cmdliner
open Native_graph_common
open Native_graph_args

let pp_audits fmt (audits : Pass.Audit_log.t) =
  let reports =
    List.map
      (fun (a : Pass.Audit.t) ->
        (* The execution id, not the bare pass name: a fixpoint contributes one
           audit per iteration and a sequence may hold the same leaf twice, so a
           name alone does not say which run a report describes. *)
        (Core.Pretty.to_string Pass.Exec_id.pp a.id, a.report))
      audits.reports
  in
  match reports with
  | [] -> Format.fprintf fmt "symbolic verification: nothing to check@."
  | _ ->
      List.iter
        (fun (pass, report) ->
          Format.fprintf fmt "@[<v 2>symbolic verification: %s@,%a@]@." pass
            Map_verify.Report.pp_groups report)
        reports;
      (* An omitted report's clusters are still counted, in the aggregate the
         log folded them into — so the total covers the whole run even when the
         retention budget dropped individual reports. *)
      (match audits.overflow with
      | None -> ()
      | Some s ->
          Format.fprintf fmt
            "@[<v 2>symbolic verification: %Ld further report(s), retained as \
             counts@,\
             %a@]@."
            s.Pass.Audit_summary.omitted_reports
            (Fmt.list (fun fmt (l, n) -> Fmt.pf fmt "%s: %Ld" l n))
            (Pass.Outcome_counts.bindings s.Pass.Audit_summary.counts));
      Format.fprintf fmt "@[<v 2>symbolic verification: total@,%a@]@."
        Map_verify.Tally.pp
        (Map_verify.Tally.of_entries
           (List.concat_map
              (fun (_, (r : Map_verify.Report.t)) ->
                r.Map_verify.Report.entries)
              reports))

(* Annotates the TRANSFORMED graph with provenance recovered through the lens,
   which is the whole claim of §10 made visible: the sidecar still describes the
   imported graph, and every name here was recovered by walking the composed map
   backwards. A folded constant has no PT2 name and no archive path of its own,
   so it shows what it was computed from instead. *)
(* A node's claim is the WEAKEST over its outputs, since the node is only as
   verified as its least-verified result. Nodes are what a reader scans for, so
   they carry the roll-up and the edges carry the detail. *)
let verdicts_by_node graph verdicts =
  List.fold_left
    (fun acc (n : Graph_ir.node) ->
      match
        List.filter_map
          (fun out ->
            Option.map
              (fun ((o : Map_verify.Outcome.t), _) -> o)
              (Graph_ir.Tensor_id.Map.find_opt out verdicts))
          n.Graph_ir.Node.outputs
      with
      | [] -> acc
      | first :: rest ->
          (* [Outcome.join], so the surviving verdict keeps ITS OWN coverage.
             Joining the two fields separately let an [Unproved Too_large] —
             which examined nothing, hence [Not_applicable] — come out marked
             [sampled n] borrowed from a sibling output. *)
          Graph_ir.Node_id.Map.add n.Graph_ir.Node.id
            (List.fold_left Map_verify.Outcome.join first rest)
            acc)
    Graph_ir.Node_id.Map.empty (Graph_ir.nodes graph)

(* No braces of its own: [Graph_ir.pp_with] already wraps the annotation, and
   nesting reads as structure that is not there. *)
(* Coverage is printed, not dropped. Under an effort that samples, a
   [Proved Structural] is a proof about four coordinates, and rendering it as
   plain "proved (structural)" is exactly the overstatement [Coverage] exists to
   prevent. *)
let pp_outcome ppf (o : Map_verify.Outcome.t) =
  Fmt.pf ppf "verify=%s" (Map_verify.Verdict.label o.verdict);
  match o.coverage with
  | Map_verify.Coverage.Exhaustive | Map_verify.Coverage.Not_applicable -> ()
  | Map_verify.Coverage.Sampled n -> Fmt.pf ppf " [sampled %d]" n

let pp_verdict_annotation verdicts ppf id =
  Option.iter
    (fun (outcome, sources) ->
      Fmt.pf ppf " %a" pp_outcome outcome;
      if sources <> 1 then Fmt.pf ppf " origins=%d" sources)
    (Graph_ir.Tensor_id.Map.find_opt id verdicts)

let pp_lens_printer ?(verdicts = Graph_ir.Tensor_id.Map.empty)
    ?(node_verdicts = Graph_ir.Node_id.Map.empty) lens derived :
    Graph_ir.Printer.t =
  {
    tensor =
      (fun ppf id ->
        (match
           ( Pt2_native_graph.tensor_origins lens id,
             Pt2_native_graph.captured_target lens id )
         with
        | Error e, _ | _, Error e ->
            Fmt.pf ppf "provenance error: %a" Pt2_native_graph.pp_lens_error
              (Err.Error.kind e)
        | Ok origins, Ok target -> (
            match (origins, target, List.assoc_opt id derived) with
            | [], _, Some names ->
                Fmt.pf ppf "folded from=[%a]"
                  (Fmt.list ~sep:(Fmt.any ",") Fmt.string)
                  names
            | [], _, None -> Fmt.string ppf "derived"
            | origins, target, _ ->
                Fmt.pf ppf "pt2=%a"
                  (Fmt.list ~sep:(Fmt.any ";")
                     (fun ppf (o : Pt2_native_graph.Tensor_origin.t) ->
                       Fmt.pf ppf "%a:%s" Pt2_native_graph.Graph_path.pp
                         o.graph_path o.ssa_name))
                  origins;
                Option.iter (Fmt.pf ppf " target=%s") target));
        pp_verdict_annotation verdicts ppf id);
    node =
      (fun ppf id ->
        (match Pt2_native_graph.node_origins lens id with
        | Error e ->
            Fmt.pf ppf "provenance error: %a" Pt2_native_graph.pp_lens_error
              (Err.Error.kind e)
        | Ok [] -> Fmt.string ppf "derived"
        | Ok origins ->
            List.iteri
              (fun i (o : Pt2_native_graph.Node_origin.t) ->
                if i > 0 then Fmt.string ppf "; ";
                Fmt.pf ppf "pt2=%a[%d] %s" Pt2_native_graph.Graph_path.pp
                  o.graph_path o.index o.target)
              origins);
        Option.iter
          (fun o -> Fmt.pf ppf " %a" pp_outcome o)
          (Graph_ir.Node_id.Map.find_opt id node_verdicts));
  }

let pp_summary ppf (Native_interp.Transformed t) =
  let constants =
    List.length
      (List.filter
         (fun id -> Graph_ir.input_kind t.graph id = Graph_ir.Input.Constant)
         t.graph.Graph_ir.Graph.inputs)
  in
  Format.fprintf ppf "nodes: %d -> %d@." t.nodes_before
    (List.length t.graph.Graph_ir.Graph.nodes);
  Format.fprintf ppf "constants: %d, of which %d folded@." constants
    (List.length t.derived)

let transform model input expect fold verify verify_symbolic :
    (unit, string) result =
  with_archive model (fun archive ->
      let* transformed =
        to_cli Native_interp.pp_error
          (Native_interp.transform ~preload:fold
             ?verify:
               (Option.map
                  (fun _ -> Map_verify.Policy.Reject_refuted)
                  verify_symbolic)
             ?verify_budget:
               (Option.map Map_verify.Effort.budget verify_symbolic)
             ?verify_probe:(Option.map Map_verify.Effort.probe verify_symbolic)
             archive ~passes:(passes ~fold))
      in
      let (Native_interp.Transformed t) = transformed in
      pp_summary Format.std_formatter transformed;
      if Option.is_some verify_symbolic then
        pp_audits Format.std_formatter t.audits;
      match input with
      | None ->
          (* Structure only: deterministic, and no inference to wait for. *)
          Format.printf "%a@."
            (let verdicts = verdicts_by_edge t.composed in
             Graph_ir.pp_with
               ~printer:
                 (pp_lens_printer ~verdicts
                    ~node_verdicts:(verdicts_by_node t.graph verdicts)
                    t.lens t.derived))
            t.graph;
          Ok ()
      | Some input -> (
          let* input =
            to_cli Pt2_archive.pp_error (Pt2_archive.load_first_pt_tensor input)
          in
          let* outputs, loaded =
            to_cli Native_interp.pp_error
              (Native_interp.evaluate archive transformed ~input)
          in
          Format.printf "payloads: %d computed, %d from the archive@."
            loaded.from_state loaded.from_archive;
          let verified output =
            if not verify then Ok ()
            else
              let* reference =
                to_cli Native_interp.pp_error (Native_interp.run archive ~input)
              in
              match reference with
              | [ reference ] -> (
                  match compare_tensor ~atol:1e-4 reference output with
                  | Ok distance ->
                      Format.printf "vs untransformed: %a@." pp_distance
                        distance;
                      Ok ()
                  | Error (_, message) -> Error message)
              | _ -> Error "untransformed graph produced unexpected outputs"
          in
          match (outputs, expect) with
          | [ output ], None ->
              Format.printf "output[0] = %a@." Tensor.pp output;
              verified output
          | [ output ], Some path -> (
              match Graph_json.decode_tensor (read_source path) with
              | Error e ->
                  Error
                    (Fmt.str "cannot decode reference: %a" Graph_json.pp_error
                       (Err.Error.kind e))
              | Ok expected -> (
                  match compare_tensor ~atol:1e-4 expected output with
                  | Ok distance ->
                      Format.printf "output[0]: %a@." pp_distance distance;
                      Format.printf "output[0]: matches %s@." path;
                      verified output
                  | Error (distance, message) ->
                      Option.iter
                        (fun distance ->
                          Format.printf "output[0]: %a@." pp_distance distance)
                        distance;
                      Error message))
          | _ -> Error "expected exactly one native graph output"))

let transform_cmd =
  let doc =
    "Import and transform a PT2 graph, printing the result or evaluating it, \
     with provenance resolved through the transformation map."
  in
  Cmd.v
    (Cmd.info "transform" ~doc)
    Term.(
      const transform $ pt2_arg $ optional_input_arg $ expect_arg $ fold_arg
      $ verify_arg $ verify_symbolic_arg)

(* ---- to4d ----------------------------------------------------------------

   The Native4D conversion report .ai/native4d_plan.md stage 8 asks for. It
   answers, per model: does the canonical graph lie in the four-axis dialect, and
   if not which node puts it outside; what the destination is made of; and what
   the conversion map is worth symbolically.

   Correctness and cost are reported SEPARATELY, per §11 stage 6, and cost is
   left out of the golden entirely — a timing in a cram is a flaky test, not a
   measurement. *)
