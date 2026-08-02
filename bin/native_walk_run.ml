(* CLI: run one of the generated ATen-vs-native walk modules (lib/aten_op_walk,
   see .ai/native_walk_design.md) by target name, for a fixed number of steps.
   Walks Walk_core.Walk.run directly (rather than through Op_walk.run) with
   Aten_spec_run.compare_report as the verify function, so each step prints
   the input tensors' shapes/distributions and the ATen output's shape +
   min/max/mean, not just a bare matched/skipped/mismatched line. *)

let target_of
    (m :
      (module Aten_walk_recipes.Walk.Op with type subject = Aten_spec.Op_spec.t))
    =
  let module M = (val m) in
  M.target

let find target =
  List.find_opt
    (fun m -> String.equal (target_of m) target)
    Aten_op_walk.all_walks

let () =
  match Sys.argv with
  | [| _; target; steps |] -> (
      match find target with
      | None -> Printf.printf "no walk recipe for %s (needs_meta)\n" target
      | Some m ->
          (* Exit non-zero when a step fails to verify. Previously the walk's
             verdict was discarded and this exited 0 on a mismatch, so anything
             driving it by exit status -- as opposed to reading the printed
             output -- saw a clean run. *)
          if
            not
              (Aten_walk_recipes.Walk.run m
                 ~verify:(fun ppf spec ->
                   Aten_spec_run.compare_report ~ppf spec)
                 ~ppf:Format.std_formatter
                 ~pcg:(Aten_spec.Pcg.seed ~seed:42L ~seq:1L)
                 ~steps:(int_of_string steps))
          then exit 1)
  | _ ->
      prerr_endline "usage: native_walk_run <target> <steps>";
      exit 1
