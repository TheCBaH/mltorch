(* Split out of native_graph.ml, see native_graph_common.ml, and
   native_graph_args.ml. *)

open Cmdliner
open Native_graph_common
open Native_graph_args

let print_graph model : (unit, string) result =
  with_archive model (fun archive ->
      let* lowered =
        to_cli Native_interp.pp_error (Native_interp.lower_archive archive)
      in
      pp_provenance Format.std_formatter lowered;
      Ok ())

let print_cmd =
  let doc =
    "Import a PT2 graph into one native graph and print its structure and PT2 \
     provenance sidecar."
  in
  Cmd.v (Cmd.info "print" ~doc) Term.(const print_graph $ pt2_arg)
