(* CLI for a whole native graph imported from PT2. See native_graph_common.ml for why this file is now just the subcommand
   group and entry point: print/eval/transform/to4d/const_ssa_trace/
   visualize/detail each live in their own native_graph_<name>.ml. *)

open Cmdliner
open Native_graph_print
open Native_graph_eval
open Native_graph_transform
open Native_graph_to4d
open Native_graph_const_ssa
open Native_graph_visualize

let cmd =
  let doc = "Tools for the native inference engine's graph representation." in
  Cmd.group
    (Cmd.info "native_graph" ~doc)
    [
      print_cmd;
      eval_cmd;
      transform_cmd;
      to4d_cmd;
      const_ssa_trace_cmd;
      visualize_cmd;
      detail_cmd;
    ]

(* The trace policy is chosen HERE, before any command runs, because this is
   the host: [Err] reads no environment of its own, and nothing under lib/ is
   allowed to. See lib/err_host for the variables.

   A bad setting exits rather than being ignored. Silently falling back to the
   default would leave the operator debugging with a policy they did not ask
   for, which is the one situation this knob exists to avoid. Exit 124
   ([Cmd.Exit.cli_error]) because a malformed MLTORCH_ERROR_* value is a usage
   error, the same class as a malformed flag. *)
let () =
  match Err_host.install_from_env () with
  | Ok (_ : Err.Config.t option) -> ()
  | Error e ->
      Fmt.epr "native_graph: %a@." Err_host.pp_error (Err.Error.kind e);
      exit Cmd.Exit.cli_error

let () = exit (Cmd.eval_result cmd)
