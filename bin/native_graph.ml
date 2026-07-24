(* CLI for the native engine's graph representation. Currently one
   subcommand ([print]); see lib/native_graph/native_graph.ml for the walk
   itself. Structured as a [Cmd.group] from the start so further operations
   (e.g. diffing two native graphs, summarising op coverage) can land as
   sibling subcommands without changing [print]'s own invocation. *)

open Cmdliner
open Core.Syntax

(* An option (not a bare positional) since [print] is meant to grow other
   graph sources (e.g. a serialized native graph) as sibling options
   alongside this one. *)
let pt2_arg =
  let doc = "Path to the exported .pt2 model archive to convert." in
  Arg.(required & opt (some file) None & info [ "pt2" ] ~doc ~docv:"MODEL.pt2")

let verbose_arg =
  let doc =
    "Also print each source ATen node (target and name) before its translated \
     native graph."
  in
  Arg.(value & flag & info [ "v"; "verbose" ] ~doc)

(* [Cmd.eval_result] below maps [Ok ()] to exit 0 and [Error msg] to exit 123
   (`Exit.some_error`), printing [msg] itself — so failures here are reported
   as plain text, with no OCaml backtrace, matching a normal CLI's UX (unlike
   [Core.Error.pp], which is meant for developer-facing diagnostics). *)
let print_graph verbose model : (unit, string) result =
  let result =
    let* archive =
      Pt2_archive.open_pt2 model
      |> Core.map_error (fun e -> (e :> Native_graph.error))
    in
    Native_graph.run ~verbose Format.std_formatter archive
  in
  match result with
  | Ok () -> Ok ()
  | Error e ->
      Error (Format.asprintf "%a" Native_graph.pp_error e.Core.Error.kind)

let print_cmd =
  let doc =
    "Convert a real .pt2 model's exported graph to the native engine's \
     Graph_ir, node by node, printing each node's translation. Stops at the \
     first node with no native implementation (see \
     .ai/native_aten_bridge_layout.md)."
  in
  Cmd.v (Cmd.info "print" ~doc) Term.(const print_graph $ verbose_arg $ pt2_arg)

let cmd =
  let doc = "Tools for the native inference engine's graph representation." in
  Cmd.group (Cmd.info "native_graph" ~doc) [ print_cmd ]

let () = exit (Cmd.eval_result cmd)
