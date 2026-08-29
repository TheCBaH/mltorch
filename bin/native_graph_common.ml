(* CLI for a whole native graph imported from PT2.  [print] and [eval] share
   the exact same [Native_interp] import: PT2 names live only in the sidecar,
   while the native graph itself is addressed by deterministic ids.

   This file holds the helpers shared across more than one subcommand file
   (native_graph_print.ml, _eval.ml, _transform.ml, _to4d.ml, _const_ssa.ml,
   _visualize.ml): the Err/Cmdliner boundary, tensor-distance comparison,
   provenance/inline printing, [with_archive], [passes], and
   [verdicts_by_edge] (shared by _transform.ml and _to4d.ml). Cmdliner
   argument terms live in native_graph_args.ml instead. Split out of
   native_graph.ml. *)

(* Every command below returns Cmdliner's [(unit, string) result], so each step
   has to leave the [Err.t] framework. [to_cli] is that boundary, named
   once instead of open-coded at each of the ten crossings.

   The detection provenance is DELIBERATELY dropped: this string is a
   diagnostic for whoever ran the command, not for whoever wrote the code. That
   is also why [Err.map_error] is wrong here — it would keep the [Err.Error.t]
   wrapper, which is precisely what Cmdliner cannot take.

   [Err.export] is what makes the drop visible rather than merely commented: it
   marks [Export] and then unwraps, so under a policy that records boundaries an
   error leaving through here carries an [Export] event, and the marking is part
   of the operation rather than something this helper has to remember. What is
   left is the typed payload; rendering it is this boundary's own decision. *)
let to_cli pp r =
  Err.export ~pos:__POS__ r |> Result.map_error (Core.Pretty.to_string pp)

(* Plain [Result.bind]: these are Cmdliner's bare results, already lowered by
   [to_cli], so [Err.Syntax]'s operators do not apply. *)
let ( let* ) = Result.bind
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
   [Err.Error.pp], which is meant for developer-facing diagnostics). *)
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
                (Option.fold ~none:""
                   ~some:(fun name -> " (" ^ name ^ ")")
                   origin.name))
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
  let* archive = to_cli Pt2_archive.pp_error (Pt2_archive.open_pt2 model) in
  f archive

let passes ~fold = [ Pipeline.canonical ~fold ]

(* Destination edge -> the one claim the whole pipeline makes about it, and how
   many origin edges collapsed into it. A node several passes rewrote reads as
   the claim they add up to rather than as a pile of intermediate ones, and the
   origin count is what shows the rewriting happened at all: [origins=3] is
   three source edges that became this one, [origins=0] an edge a pass created.
   Shared with native_graph_to4d.ml. *)
let verdicts_by_edge (report : Map_verify.Report.t option) =
  (* No report and an empty report agree: the fold's seed IS the absent case. *)
  report
  |> Option.fold ~none:[] ~some:(fun (r : Map_verify.Report.t) ->
      r.Map_verify.Report.entries)
  |> List.fold_left
       (fun acc (e : Map_verify.Entry.t) ->
         Graph_ir.Tensor_id.Set.fold
           (fun id acc ->
             Graph_ir.Tensor_id.Map.add id
               (e.outcome, Graph_ir.Tensor_id.Set.cardinal e.cluster.src)
               acc)
           e.cluster.dst acc)
       Graph_ir.Tensor_id.Map.empty
