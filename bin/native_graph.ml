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

let pp_optional_float ppf = function
  | Some x -> Format.fprintf ppf "%g" x
  | None -> Format.pp_print_string ppf "n/a"

let pp_distance ppf d =
  Format.fprintf ppf
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
      Format.fprintf ppf "%a:%s" Pt2_native_graph.Graph_path.pp graph_path
        ssa_name
  | Derived -> Format.pp_print_string ppf "derived"

let pp_inline_printer (lowered : Pt2_native_graph.t) : Graph_ir.Printer.t =
  {
    tensor =
      (fun ppf id ->
        match Graph_ir.Tensor_id.Map.find_opt id lowered.tensor_origins with
        | None -> Format.pp_print_string ppf "derived"
        | Some origin ->
            Format.fprintf ppf "pt2=%a" pp_tensor_origin origin;
            Option.iter
              (fun target -> Format.fprintf ppf " target=%s" target)
              (Graph_ir.Tensor_id.Map.find_opt id lowered.captured_targets));
    node =
      (fun ppf id ->
        match Graph_ir.Node_id.Map.find_opt id lowered.node_origins with
        | None -> Format.pp_print_string ppf "derived"
        | Some origins ->
            List.iteri
              (fun i (origin : Pt2_native_graph.Node_origin.t) ->
                if i > 0 then Format.pp_print_string ppf "; ";
                Format.fprintf ppf "pt2=%a[%d] %s%s"
                  Pt2_native_graph.Graph_path.pp origin.graph_path origin.index
                  origin.target
                  (match origin.name with
                  | None -> ""
                  | Some name -> " (" ^ name ^ ")"))
              origins);
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
              Some
                (Native_interp.Hooks
                   {
                     on_start = (fun _ _ -> Sys.time ());
                     on_end =
                       (fun lowered node started ->
                         Format.eprintf "%a@,[eval] n%d compute: %.3f ms@."
                           (Graph_ir.pp_node
                              ~printer:(pp_inline_printer lowered)
                              lowered.graph)
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

let cmd =
  let doc = "Tools for the native inference engine's graph representation." in
  Cmd.group (Cmd.info "native_graph" ~doc) [ print_cmd; eval_cmd ]

let () = exit (Cmd.eval_result cmd)
