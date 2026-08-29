(* Split out of native_graph.ml, see native_graph_common.ml, and
   native_graph_args.ml. *)

open Cmdliner
open Native_graph_common
open Native_graph_args

let op_counts (g : Native4d.Graph.graph) =
  List.fold_left
    (fun acc (n : Native4d.Graph.node) ->
      let name = Native4d.Op.name n.Graph_common.Node.op in
      let seen = Option.value (List.assoc_opt name acc) ~default:0 in
      (name, seen + 1) :: List.remove_assoc name acc)
    [] g.Graph_common.Graph.nodes
  |> List.sort (fun (a, _) (b, _) -> String.compare a b)

let to4d model fold verify_symbolic input : (unit, string) result =
  with_archive model (fun archive ->
      let* transformed =
        to_cli Native_interp.pp_error
          (Native_interp.transform ~preload:fold archive ~passes:(passes ~fold))
      in
      let (Native_interp.Transformed t) = transformed in
      Format.printf "canonical native: nodes=%d@."
        (List.length t.graph.Graph_common.Graph.nodes);
      let* snapshot = to_cli Graph_view.pp_error (Snapshot.create t.graph) in
      let (Snapshot.Pack src) = snapshot in
      match
        Native4d.Lower.convert ~constants:t.constants
          ~constant_store:t.constant_store src
      with
      | Error e ->
          (* Not a failure of the tool: a graph outside the dialect's
                     domain is the partiality the design is explicit about, so
                     the first unsupported node is the ANSWER, printed and
                     exited zero. *)
          Format.printf "outside the dialect: %a@." Native4d.Error.pp
            (Err.Error.kind e);
          Ok ()
      | Ok (Native4d.Lower.Pack r) -> (
          let dst = Native4d.Lower.graph r in
          Format.printf "native4d: nodes=%d inputs=%d@."
            (List.length dst.Graph_common.Graph.nodes)
            (List.length dst.Graph_common.Graph.inputs);
          List.iter
            (fun (name, n) -> Format.printf "  %-18s %d@." name n)
            (op_counts dst);
          (* THE STRUCTURE, and each edge's verdict beside it. Counts
                     say what the graph is made of; only the graph says how it is
                     wired, and only a per-edge verdict says which parts of the
                     conversion are actually justified. *)
          let verdicts =
            match verify_symbolic with
            | None -> Graph_ir.Tensor_id.Map.empty
            | Some effort -> (
                match
                  Native4d.Framework.Verify_from_native.run
                    ~budget:(Map_verify.Effort.budget effort)
                    ~probe:(Map_verify.Effort.probe effort)
                    ~src_constants:t.constants
                    ~dst_constants:r.Native4d.Lower.constants
                    r.Native4d.Lower.map ~src ~dst:r.Native4d.Lower.dst
                with
                | Error e ->
                    Format.printf "map verification: %a@." Map_verify.pp_error
                      (Err.Error.kind e);
                    Graph_ir.Tensor_id.Map.empty
                | Ok report ->
                    Format.printf "map verification: %s@."
                      (Map_verify.Report.summary report);
                    verdicts_by_edge (Some report))
          in
          let annot id =
            Option.map
              (fun (outcome, _) ->
                Format.asprintf "%a" Map_verify.Outcome.pp outcome)
              (Graph_ir.Tensor_id.Map.find_opt id verdicts)
          in
          Format.printf "%a@." (Native4d.Graph.pp_with ~annot) dst;
          match input with
          | None -> Ok ()
          | Some path -> (
              let* raw =
                to_cli Pt2_archive.pp_error
                  (Pt2_archive.load_first_pt_tensor path)
              in
              let* native_outputs, _ =
                to_cli Native_interp.pp_error
                  (Native_interp.evaluate archive transformed ~input:raw)
              in
              let* input =
                to_cli Native_interp.pp_error (Native_interp.tensor_of_pt2 raw)
              in
              let user_inputs =
                List.filter
                  (fun id ->
                    Graph_common.input_kind dst id = Graph_ir.Input.Input)
                  dst.Graph_common.Graph.inputs
              in
              let* env, report =
                match user_inputs with
                | [ id ] ->
                    to_cli Native4d.Lower.pp_eval_error
                      (Native4d.Lower.evaluate
                         (Native_interp.capture_resolver archive)
                         r
                         ~inputs:[ (id, input) ])
                | ids ->
                    Error
                      (Printf.sprintf
                         "Native4D graph has %d user inputs, expected one"
                         (List.length ids))
              in
              match (native_outputs, dst.Graph_common.Graph.outputs) with
              | [ native ], [ output ] -> (
                  match Tensor_id.Map.find_opt output env with
                  | None -> Error "Native4D graph did not evaluate its output"
                  | Some native4d -> (
                      match compare_tensor ~atol:1e-4 native native4d with
                      | Ok distance ->
                          Format.printf
                            "materialized Native4D: applies=%d, native \
                             agreement: %a@."
                            report.applies pp_distance distance;
                          Ok ()
                      | Error (_, message) -> Error message))
              | _ -> Error "expected exactly one Native and Native4D output")))

let to4d_cmd =
  let doc =
    "Convert a canonical PT2-imported graph to the Native4D four-axis dialect, \
     reporting what it converted to or which node put it outside the domain."
  in
  Cmd.v (Cmd.info "to4d" ~doc)
    Term.(
      const to4d $ pt2_arg $ fold_arg $ verify_symbolic_arg $ optional_input_arg)

(* The operation manifest for Gate 6 is deliberately emitted from a real,
   payload-backed run. [Fold_const.Trace] records only successful direct folds,
   and its canonical order erases incidental graph ids and pass sweep order, so
   redirecting this command is a stable corpus measurement rather than a debug
   log. *)
