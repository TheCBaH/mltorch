(* CLI for a whole native graph imported from PT2.  [print] and [eval] share
   the exact same [Native_interp] import: PT2 names live only in the sidecar,
   while the native graph itself is addressed by deterministic ids. *)

open Cmdliner

(* An option (not a bare positional) since [print] is meant to grow other
   graph sources (e.g. a serialized native graph) as sibling options
   alongside this one. *)
let pt2_arg =
  let doc = "Path to the exported .pt2 model archive to convert." in
  Arg.(required & opt (some file) None & info [ "pt2" ] ~doc ~docv:"MODEL.pt2")

let input_arg =
  let doc = "Path to the single user-input .pt tensor." in
  Arg.(required & opt (some file) None & info [ "input" ] ~docv:"INPUT.pt" ~doc)

(* [transform] prints the rewritten graph when given no input, so that seeing
   what a pipeline produced does not require running inference over it. *)
let optional_input_arg =
  let doc =
    "Path to the single user-input .pt tensor. Omit to print the transformed \
     graph instead of evaluating it."
  in
  Arg.(value & opt (some file) None & info [ "input" ] ~docv:"INPUT.pt" ~doc)

let expect_arg =
  let doc = "Native JSON tensor written by aten_graph_ref for comparison." in
  Arg.(
    value & opt (some file) None & info [ "expect" ] ~docv:"REFERENCE.json" ~doc)

let verbose_arg =
  let doc =
    "Print each evaluated native node and its CPU compute time to stderr."
  in
  Arg.(value & flag & info [ "verbose" ] ~doc)

let read_source path = In_channel.with_open_bin path In_channel.input_all

type distance = {
  count : int;
  max_abs : float;
  bias : float;
  mean_abs : float;
  rmse : float;
  stddev : float;
  relative_l2 : float option;
  cosine_similarity : float option;
  cosine_distance : float option;
}

type distance_acc = {
  mismatches : int;
  max_abs : float;
  sum : float;
  sum_abs : float;
  sum_sq : float;
  expected_sq : float;
  actual_sq : float;
  dot : float;
  first : (Vec6.coord * float * float) option;
}

let pp_optional_float =
  Core.Pretty.option_or ~none:"n/a" (fun ppf x -> Fmt.pf ppf "%g" x)

let pp_distance ppf d =
  Fmt.pf ppf
    "count=%d max_abs=%g bias=%g mean_abs=%g rmse=%g stddev=%g relative_l2=%a \
     cosine_similarity=%a cosine_distance=%a"
    d.count d.max_abs d.bias d.mean_abs d.rmse d.stddev pp_optional_float
    d.relative_l2 pp_optional_float d.cosine_similarity pp_optional_float
    d.cosine_distance

let compare_tensor ~atol expected actual =
  let shape_of (Tensor.Tensor tensor) = tensor.shape in
  let fmt_of (Tensor.Tensor tensor) = Payload.fmt_name tensor.payload.fmt in
  let expected_shape = shape_of expected and actual_shape = shape_of actual in
  let expected_fmt = fmt_of expected and actual_fmt = fmt_of actual in
  if expected_shape <> actual_shape then
    Error
      ( None,
        Format.asprintf "shape mismatch: expected %a got %a" Vec6.pp_shape
          expected_shape Vec6.pp_shape actual_shape )
  else if expected_fmt <> actual_fmt then
    Error
      ( None,
        Format.asprintf "format mismatch: expected %s got %s" expected_fmt
          actual_fmt )
  else
    let zero =
      {
        mismatches = 0;
        max_abs = 0.;
        sum = 0.;
        sum_abs = 0.;
        sum_sq = 0.;
        expected_sq = 0.;
        actual_sq = 0.;
        dot = 0.;
        first = None;
      }
    in
    let stats =
      Vec6.fold_coords actual_shape ~init:zero ~f:(fun stats coord ->
          let expected_value = Tensor.read expected coord
          and actual_value = Tensor.read actual coord in
          let diff = actual_value -. expected_value in
          let abs_diff = Float.abs diff in
          {
            mismatches = (stats.mismatches + if abs_diff > atol then 1 else 0);
            max_abs = max stats.max_abs abs_diff;
            sum = stats.sum +. diff;
            sum_abs = stats.sum_abs +. abs_diff;
            sum_sq = stats.sum_sq +. (diff *. diff);
            expected_sq = stats.expected_sq +. (expected_value *. expected_value);
            actual_sq = stats.actual_sq +. (actual_value *. actual_value);
            dot = stats.dot +. (expected_value *. actual_value);
            first =
              (if abs_diff > atol && stats.first = None then
                 Some (coord, expected_value, actual_value)
               else stats.first);
          })
    in
    let count = (Vec6.numel actual_shape :> int) in
    let divisor = float_of_int count in
    let bias = stats.sum /. divisor in
    let mean_square = stats.sum_sq /. divisor in
    let cosine_similarity =
      let denominator = Float.sqrt (stats.expected_sq *. stats.actual_sq) in
      if denominator = 0. then None else Some (stats.dot /. denominator)
    in
    let distance =
      {
        count;
        max_abs = stats.max_abs;
        bias;
        mean_abs = stats.sum_abs /. divisor;
        rmse = Float.sqrt mean_square;
        stddev = Float.sqrt (max 0. (mean_square -. (bias *. bias)));
        relative_l2 =
          (if stats.expected_sq = 0. then None
           else Some (Float.sqrt (stats.sum_sq /. stats.expected_sq)));
        cosine_similarity;
        cosine_distance = Option.map (fun x -> 1. -. x) cosine_similarity;
      }
    in
    match stats.first with
    | None -> Ok distance
    | Some (coord, expected_value, actual_value) ->
        Error
          ( Some distance,
            Format.asprintf
              "payload mismatch: %d elements differ; first %a expected=%g \
               got=%g"
              stats.mismatches Vec6.pp_coord coord expected_value actual_value
          )

(* [Cmd.eval_result] below maps [Ok ()] to exit 0 and [Error msg] to exit 123
   (`Exit.some_error`), printing [msg] itself — so failures here are reported
   as plain text, with no OCaml backtrace, matching a normal CLI's UX (unlike
   [Core.Error.pp], which is meant for developer-facing diagnostics). *)
let pp_tensor_origin ppf = function
  | Pt2_native_graph.Source { graph_path; ssa_name; _ } ->
      Fmt.pf ppf "%a:%s" Pt2_native_graph.Graph_path.pp graph_path ssa_name
  | Derived -> Fmt.string ppf "derived"

let pp_inline_printer (lowered : Pt2_native_graph.t) : Graph_ir.Printer.t =
  {
    tensor =
      (fun ppf id ->
        let pp_origin ppf origin =
          Fmt.pf ppf "pt2=%a" pp_tensor_origin origin;
          Option.iter
            (fun target -> Fmt.pf ppf " target=%s" target)
            (Graph_ir.Tensor_id.Map.find_opt id lowered.captured_targets)
        in
        Fmt.option ~none:(Fmt.any "derived") pp_origin ppf
          (Graph_ir.Tensor_id.Map.find_opt id lowered.tensor_origins));
    node =
      (fun ppf id ->
        let pp_origins ppf origins =
          List.iteri
            (fun i (origin : Pt2_native_graph.Node_origin.t) ->
              if i > 0 then Fmt.string ppf "; ";
              Fmt.pf ppf "pt2=%a[%d] %s%s" Pt2_native_graph.Graph_path.pp
                origin.graph_path origin.index origin.target
                (match origin.name with
                | None -> ""
                | Some name -> " (" ^ name ^ ")"))
            origins
        in
        Fmt.option ~none:(Fmt.any "derived") pp_origins ppf
          (Graph_ir.Node_id.Map.find_opt id lowered.node_origins));
  }

let pp_provenance ppf (lowered : Pt2_native_graph.t) =
  let graph = lowered.graph in
  Format.fprintf ppf
    "native graph: inputs=%d constants=%d nodes=%d outputs=%d@,\
     PT2 provenance: tensor-origins=%d captured-targets=%d node-origins=%d"
    (List.length graph.Graph_ir.Graph.inputs)
    (List.length
       (List.filter
          (fun id -> Graph_ir.input_kind graph id = Graph_ir.Input.Constant)
          graph.Graph_ir.Graph.inputs))
    (List.length graph.Graph_ir.Graph.nodes)
    (List.length graph.Graph_ir.Graph.outputs)
    (List.length (Graph_ir.Tensor_id.Map.bindings lowered.tensor_origins))
    (List.length (Graph_ir.Tensor_id.Map.bindings lowered.captured_targets))
    (List.length (Graph_ir.Node_id.Map.bindings lowered.node_origins));
  Format.fprintf ppf "@,%a"
    (Graph_ir.pp_with ~printer:(pp_inline_printer lowered))
    graph;
  Format.pp_print_flush ppf ()

let with_archive model f =
  match Pt2_archive.open_pt2 model with
  | Error e ->
      Error (Format.asprintf "%a" Pt2_archive.pp_error e.Core.Error.kind)
  | Ok archive -> f archive

let print_graph model : (unit, string) result =
  with_archive model (fun archive ->
      match Native_interp.lower_archive archive with
      | Ok lowered ->
          pp_provenance Format.std_formatter lowered;
          Ok ()
      | Error e ->
          Error (Format.asprintf "%a" Native_interp.pp_error e.Core.Error.kind))

let print_cmd =
  let doc =
    "Import a PT2 graph into one native graph and print its structure and PT2 \
     provenance sidecar."
  in
  Cmd.v (Cmd.info "print" ~doc) Term.(const print_graph $ pt2_arg)

let eval model input expect verbose : (unit, string) result =
  with_archive model (fun archive ->
      match Pt2_archive.load_pt input with
      | Error e ->
          Error (Format.asprintf "%a" Pt2_archive.pp_error e.Core.Error.kind)
      | Ok input -> (
          let hooks =
            if verbose then
              (* [on_end] fires once per evaluated node against the same
                 [lowered], so build the pp index once (it's an O(nodes)
                 rescan otherwise) and reuse it across the whole run. *)
              let index = ref None in
              let index_for (lowered : Pt2_native_graph.t) =
                match !index with
                | Some (graph, idx) when graph == lowered.graph -> idx
                | _ ->
                    let idx = Graph_ir.Index.make lowered.graph in
                    index := Some (lowered.graph, idx);
                    idx
              in
              Some
                (Native_interp.Hooks
                   {
                     on_start = (fun _ _ -> Sys.time ());
                     on_end =
                       (fun lowered node started ->
                         Format.eprintf "%a@,[eval] n%d compute: %.3f ms@."
                           (Graph_ir.pp_node
                              ~printer:(pp_inline_printer lowered)
                              ~index:(index_for lowered) lowered.graph)
                           node
                           (Graph_ir.Node_id.to_int node.Graph_ir.Node.id)
                           ((Sys.time () -. started) *. 1000.));
                   })
            else None
          in
          match Native_interp.run ?hooks archive ~input with
          | Error e ->
              Error
                (Format.asprintf "%a" Native_interp.pp_error e.Core.Error.kind)
          | Ok outputs -> (
              match expect with
              | None ->
                  List.iteri
                    (fun i output ->
                      Format.printf "output[%d] = %a@." i Tensor.pp output)
                    outputs;
                  Ok ()
              | Some path -> (
                  match
                    (outputs, Graph_json.decode_tensor (read_source path))
                  with
                  | [ output ], Ok expected -> (
                      match compare_tensor ~atol:1e-4 expected output with
                      | Ok distance ->
                          Format.printf "output[0]: %a@." pp_distance distance;
                          Format.printf "output[0]: matches %s@." path;
                          Ok ()
                      | Error (distance, message) ->
                          Option.iter
                            (fun distance ->
                              Format.printf "output[0]: %a@." pp_distance
                                distance)
                            distance;
                          Error message)
                  | _ :: _ :: _, _ ->
                      Error "expected exactly one native graph output"
                  | [], _ -> Error "native graph produced no outputs"
                  | _, Error message ->
                      Error ("cannot decode reference: " ^ message)))))

let eval_cmd =
  let doc =
    "Import and evaluate a static single-input PT2 graph with the native \
     interpreter."
  in
  Cmd.v (Cmd.info "eval" ~doc)
    Term.(const eval $ pt2_arg $ input_arg $ expect_arg $ verbose_arg)

(* The pipeline the transform subcommand runs. Structural only: no weight is
   materialised, so [Fold_const] is deliberately absent — with no payload bound
   it would decline every node, and including it would suggest otherwise. The
   permute passes are here for the case .ai/native_transform_design.md §1 names
   directly: the relayout lowering emits an inverse permute pair at every op
   boundary, and cancelling them is the whole point. [Sink_permute] (§12d)
   catches the case where an elementwise op (Relu, Add, ...) sits between the
   pair — Chain/Trim only cancel ADJACENT permutes, so sinking has to make
   them adjacent first; a second Chain/Trim round then collapses what it
   exposes. *)
(* [Sink_permute_mean] transports a permutation through a [keepdim=true]
   [Mean] (§12f) rather than sinking it unchanged — [Mean] is intentionally
   absent from [Sink_permute]'s allowlist, since reducing the same axis names
   after removing its input permutation would reduce the wrong dimensions.
   It runs right after the initial chain/trim cleanup: transport can expose
   an adjacent permute pair on either side, which [Sink_permute] and a later
   [Chain_permute]/[Trim_permute] round then pick up. *)
(* [Reuse_permute] and [Bypass_permute] complement [Sink_permute]: see
   .ai/native_layout_reuse_plan.md. [Reuse_permute] turns a mixed elementwise
   operand set uniform by reusing an alternate-layout edge the graph already
   computes, which [Sink_permute] can then move downstream; [Bypass_permute]
   then removes the inverse consumers that move exposes, without requiring
   the whole run to be interior the way [Trim_permute] does. These unlock one
   another — bypassing one residual block's inverse permutes can make the
   next block's skip edge interior, exposing another sink/trim opportunity —
   so the whole group runs under one outer fixed point rather than a single
   pass over each. *)
let relayout_pass =
  Pass.fixpoint
    (Pass.sequence ~name:"relayout"
       [
         Pass.fixpoint Chain_permute.pass;
         Pass.fixpoint Trim_permute.pass;
         Pass.fixpoint Sink_permute_mean.pass;
         Pass.fixpoint Sink_permute.pass;
         Pass.fixpoint Reuse_permute.pass;
         Pass.fixpoint Sink_permute.pass;
         Pass.fixpoint Bypass_permute.pass;
         Pass.fixpoint Chain_permute.pass;
         Pass.fixpoint Trim_permute.pass;
       ])

(* Order is load-bearing. The importer emits every conv weight behind a relayout
   permute, so the weight is a NODE OUTPUT until folding materialises it — and
   batch-norm folding requires constant parameters. So constant folding runs
   first to make the weights constant, then the batch-norm fold, then constant
   folding again to collapse the parameter arithmetic that fold emits. Without
   the first pass the batch-norm fold matches nothing at all. *)
let passes ~fold =
  [ Reshape_to_permute.pass; relayout_pass ]
  @
  if fold then
    [
      Pass.fixpoint Fold_const.pass;
      Fold_batch_norm.pass;
      Pass.fixpoint Fold_const.pass;
    ]
  else [ Fold_batch_norm.pass ]

let fold_arg =
  let doc =
    "Load every captured weight up front so constant folding can hoist \
     parameter arithmetic to load time. Reads the whole archive."
  in
  Arg.(value & flag & info [ "fold" ] ~doc)

let verify_arg =
  let doc =
    "Also evaluate the untransformed graph and report the distance between the \
     two outputs. Requires --input."
  in
  Arg.(value & flag & info [ "verify" ] ~doc)

(* A different check from --verify, and complementary. That one runs the whole
   model twice on real weights and compares the two OUTPUTS; this one checks
   every corresponding edge of every pass's mapping symbolically, without
   payloads for the graph inputs, so what it says holds for every input rather
   than for the one tensor supplied. It is budget-capped, so a real model's
   activation-shaped clusters come back "too large" and the useful coverage is
   the constant-shaped ones — folded weights and biases, exactly where
   fold_const and fold_batch_norm act. See .ai/native_transform_verify.md. *)
let effort_conv =
  let parse s =
    match Map_verify.Effort.of_string s with
    | Ok e -> Ok e
    | Error (`Unknown_effort other) ->
        Error (`Msg (Printf.sprintf "unknown effort %S" other))
  in
  Arg.conv (parse, fun fmt e -> Map_verify.Effort.pp fmt e)

let verify_symbolic_arg =
  let doc =
    Printf.sprintf
      "Symbolically verify each pass's mapping as it is applied, at EFFORT \
       (%s). Needs no --input, and fails only on an actual counterexample."
      (String.concat ", "
         (List.map Map_verify.Effort.to_string Map_verify.Effort.all))
  in
  Arg.(
    value
    & opt (some effort_conv) None
    & info [ "verify-symbolic" ] ~docv:"EFFORT" ~doc)

(* Per group, then the roll-up. Groups are what a reader recognises in a real
   model — "layer1.0", "features.3" — so a report over 170 clusters is only
   legible attributed to them. The roll-up counts by outcome AND reason, since
   "40 unproved" says nothing without "because too large". *)
let pp_audits fmt audits =
  let reports =
    List.map (fun (a : Pass.Audit.t) -> (a.Pass.Audit.pass, a.report)) audits
  in
  match reports with
  | [] -> Format.fprintf fmt "symbolic verification: nothing to check@."
  | _ ->
      List.iter
        (fun (pass, report) ->
          Format.fprintf fmt "@[<v 2>symbolic verification: %s@,%a@]@." pass
            Map_verify.Report.pp_groups report)
        reports;
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
let pp_lens_printer lens derived : Graph_ir.Printer.t =
  {
    tensor =
      (fun ppf id ->
        match
          ( Pt2_native_graph.tensor_origins lens id,
            Pt2_native_graph.captured_target lens id )
        with
        | Error e, _ | _, Error e ->
            Fmt.pf ppf "provenance error: %a" Pt2_native_graph.pp_lens_error
              e.Core.Error.kind
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
    node =
      (fun ppf id ->
        match Pt2_native_graph.node_origins lens id with
        | Error e ->
            Fmt.pf ppf "provenance error: %a" Pt2_native_graph.pp_lens_error
              e.Core.Error.kind
        | Ok [] -> Fmt.string ppf "derived"
        | Ok origins ->
            List.iteri
              (fun i (o : Pt2_native_graph.Node_origin.t) ->
                if i > 0 then Fmt.string ppf "; ";
                Fmt.pf ppf "pt2=%a[%d] %s" Pt2_native_graph.Graph_path.pp
                  o.graph_path o.index o.target)
              origins);
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
      match
        Native_interp.transform ~preload:fold
          ?verify:
            (Option.map
               (fun _ -> Map_verify.Policy.Reject_refuted)
               verify_symbolic)
          ?verify_budget:(Option.map Map_verify.Effort.budget verify_symbolic)
          archive ~passes:(passes ~fold)
      with
      | Error e ->
          Error (Format.asprintf "%a" Native_interp.pp_error e.Core.Error.kind)
      | Ok (Native_interp.Transformed t as transformed) -> (
          pp_summary Format.std_formatter transformed;
          if Option.is_some verify_symbolic then
            pp_audits Format.std_formatter t.audits;
          match input with
          | None ->
              (* Structure only: deterministic, and no inference to wait for. *)
              Format.printf "%a@."
                (Graph_ir.pp_with ~printer:(pp_lens_printer t.lens t.derived))
                t.graph;
              Ok ()
          | Some input -> (
              match Pt2_archive.load_pt input with
              | Error e ->
                  Error
                    (Format.asprintf "%a" Pt2_archive.pp_error e.Core.Error.kind)
              | Ok input -> (
                  match Native_interp.evaluate archive transformed ~input with
                  | Error e ->
                      Error
                        (Format.asprintf "%a" Native_interp.pp_error
                           e.Core.Error.kind)
                  | Ok (outputs, loaded) -> (
                      Format.printf
                        "payloads: %d computed, %d from the archive@."
                        loaded.from_state loaded.from_archive;
                      let verified output =
                        if not verify then Ok ()
                        else
                          match Native_interp.run archive ~input with
                          | Error e ->
                              Error
                                (Format.asprintf "%a" Native_interp.pp_error
                                   e.Core.Error.kind)
                          | Ok [ reference ] -> (
                              match
                                compare_tensor ~atol:1e-4 reference output
                              with
                              | Ok distance ->
                                  Format.printf "vs untransformed: %a@."
                                    pp_distance distance;
                                  Ok ()
                              | Error (_, message) -> Error message)
                          | Ok _ ->
                              Error
                                "untransformed graph produced unexpected \
                                 outputs"
                      in
                      match (outputs, expect) with
                      | [ output ], None ->
                          Format.printf "output[0] = %a@." Tensor.pp output;
                          verified output
                      | [ output ], Some path -> (
                          match Graph_json.decode_tensor (read_source path) with
                          | Error message ->
                              Error ("cannot decode reference: " ^ message)
                          | Ok expected -> (
                              match
                                compare_tensor ~atol:1e-4 expected output
                              with
                              | Ok distance ->
                                  Format.printf "output[0]: %a@." pp_distance
                                    distance;
                                  Format.printf "output[0]: matches %s@." path;
                                  verified output
                              | Error (distance, message) ->
                                  Option.iter
                                    (fun distance ->
                                      Format.printf "output[0]: %a@."
                                        pp_distance distance)
                                    distance;
                                  Error message))
                      | _ -> Error "expected exactly one native graph output")))
          ))

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

let cmd =
  let doc = "Tools for the native inference engine's graph representation." in
  Cmd.group
    (Cmd.info "native_graph" ~doc)
    [ print_cmd; eval_cmd; transform_cmd ]

let () = exit (Cmd.eval_result cmd)
