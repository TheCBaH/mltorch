(* Split out of native_graph.ml, see native_graph_common.ml, and
   native_graph_args.ml. *)

open Cmdliner
open Native_graph_common
open Native_graph_args

let eval model input expect verbose : (unit, string) result =
  with_archive model (fun archive ->
      let* input =
        to_cli Pt2_archive.pp_error (Pt2_archive.load_first_pt_tensor input)
      in
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
      let* outputs =
        to_cli Native_interp.pp_error (Native_interp.run ?hooks archive ~input)
      in
      match expect with
      | None ->
          List.iteri
            (fun i output ->
              Format.printf "output[%d] = %a@." i Tensor.pp output)
            outputs;
          Ok ()
      | Some path -> (
          match (outputs, Graph_json.decode_tensor (read_source path)) with
          | [ output ], Ok expected -> (
              match compare_tensor ~atol:1e-4 expected output with
              | Ok distance ->
                  Format.printf "output[0]: %a@." pp_distance distance;
                  Format.printf "output[0]: matches %s@." path;
                  Ok ()
              | Error (distance, message) ->
                  Option.iter
                    (fun distance ->
                      Format.printf "output[0]: %a@." pp_distance distance)
                    distance;
                  Error message)
          | _ :: _ :: _, _ -> Error "expected exactly one native graph output"
          | [], _ -> Error "native graph produced no outputs"
          | _, Error e ->
              Error
                (Fmt.str "cannot decode reference: %a" Graph_json.pp_error
                   (Err.Error.kind e))))

let eval_cmd =
  let doc =
    "Import and evaluate a static single-input PT2 graph with the native \
     interpreter."
  in
  Cmd.v (Cmd.info "eval" ~doc)
    Term.(const eval $ pt2_arg $ input_arg $ expect_arg $ verbose_arg)

(* The canonical pipeline now lives in [Pipeline], because Native4D needs the
   same definition of "canonical" and two callers agreeing by coincidence is not
   a definition. The ordering rationale moved with it. *)
