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

let eval model input : (unit, string) result =
  with_archive model (fun archive ->
      match Pt2_archive.load_pt input with
      | Error e ->
          Error (Format.asprintf "%a" Pt2_archive.pp_error e.Core.Error.kind)
      | Ok input -> (
          match Native_interp.run archive ~input with
          | Error e ->
              Error
                (Format.asprintf "%a" Native_interp.pp_error e.Core.Error.kind)
          | Ok outputs ->
              List.iteri
                (fun i output ->
                  Format.printf "output[%d] = %a@." i Tensor.pp output)
                outputs;
              Ok ()))

let eval_cmd =
  let doc =
    "Import and evaluate a static single-input PT2 graph with the native \
     interpreter."
  in
  Cmd.v (Cmd.info "eval" ~doc) Term.(const eval $ pt2_arg $ input_arg)

let cmd =
  let doc = "Tools for the native inference engine's graph representation." in
  Cmd.group (Cmd.info "native_graph" ~doc) [ print_cmd; eval_cmd ]

let () = exit (Cmd.eval_result cmd)
