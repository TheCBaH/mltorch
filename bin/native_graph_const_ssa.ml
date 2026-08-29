(* Split out of native_graph.ml, see native_graph_common.ml, and
   native_graph_args.ml. *)

open Cmdliner
open Native_graph_common
open Native_graph_args

let const_ssa_trace model : (unit, string) result =
  with_archive model (fun archive ->
      let events = ref [] in
      (* This intentionally omits [transform_lowered]'s captured bindings. The
         command measures the legacy materialized path against which the
         Const-SSA language is being closed; seeding captures here would make
         every fold symbolic and emit an empty, misleading manifest. *)
      let* lowered =
        to_cli Native_interp.pp_error (Native_interp.lower_archive archive)
      in
      let* constants =
        to_cli Native_interp.pp_error (Native_interp.preload archive lowered)
      in
      let* (Rewrite.Origin origin) =
        to_cli Rewrite.pp_error
          (Rewrite.origin
             ~constants:(Tensor_id.Map.bindings constants)
             lowered.Pt2_native_graph.graph)
      in
      let* _ =
        to_cli Pass.pp_error
          (Pass.run_all origin
             [
               Pipeline.canonical_with_trace ~fold:true
                 ~on_materialized_fold:(fun event -> events := event :: !events);
             ])
      in
      let events = Fold_const.Trace.canonical !events in
      match
        List.find_opt
          (fun event -> not (Const_ssa.allows (Fold_const.Trace.op event)))
          events
      with
      | Some event ->
          Error
            (Format.asprintf
               "Const-SSA registry is missing materialized fold: %a"
               Fold_const.Trace.pp event)
      | None ->
          List.iter
            (fun event -> Format.printf "%a@." Fold_const.Trace.pp event)
            events;
          Ok ())

let const_ssa_trace_cmd =
  let doc =
    "Trace successful payload-backed Native constant folds in deterministic \
     Const-SSA manifest order."
  in
  Cmd.v (Cmd.info "const-ssa-trace" ~doc) Term.(const const_ssa_trace $ pt2_arg)

(* --- visualize: export a Model Explorer session ------------------------- *)

(* The shell owns the filesystem, and the library takes bytes -- so the size
   check belongs here too. Peeks the first 4 bytes to pick the applicable
   ceiling -- [Me_export.detect]'s own magic-byte rule, reused rather than
   re-derived so the two cannot disagree about what a file is -- and the size
   check runs before the rest is read, so an oversized file of either format
   never becomes an OCaml string. Shared by [visualize] and [detail]: two
   copies of a filesystem size check is exactly the kind of drift that let
   [detail] skip it once already. *)
