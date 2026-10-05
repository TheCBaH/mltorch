open Ssa_bridge
open Ssa_ir

(* The optimizer's differential sweep: every native op's random walk, both
   placements, lowered, optimized, and run against [Kernel_eval]. The same
   subjects as the direct sweep, so a pass that changes a result, a failure row
   or the logical work shows as a disagreement where the unoptimized program
   agreed. The runner allocates each buffer on its own, which is the guarantee
   [Distinct_buffers] asks a caller to have established. *)

let alias = Ssa_effects.Distinct_buffers
let optimize p = fst (Ssa_opt.run ~alias p)

type tally = {
  mutable agree : int;
  mutable agree_on_failure : int;
  mutable disagreements : string list;
  mutable refusals : int;
  mutable work_differs : int;
  mutable more_loads : int;
  mutable before : Ssa_stats.t;
  mutable after : Ssa_stats.t;
}

let tallies : (string, tally) Hashtbl.t = Hashtbl.create 64

let tally target =
  match Hashtbl.find_opt tallies target with
  | Some t -> t
  | None ->
      let t =
        {
          agree = 0;
          agree_on_failure = 0;
          disagreements = [];
          refusals = 0;
          work_differs = 0;
          more_loads = 0;
          before = Ssa_stats.zero;
          after = Ssa_stats.zero;
        }
      in
      Hashtbl.add tallies target t;
      t

let counters plan program ~bind =
  let counters = Ssa_interp.Counters.create () in
  ignore (Ssa_lower.Ssa_exec.run ~counters plan program ~bind);
  counters

let marks c = List.map (Ssa_interp.Counters.mark c) Ssa_mark.all

let record t plan ~bind =
  match Err.payload (Ssa_lower.Ssa_lower_plan.lower plan) with
  | Error _ -> t.refusals <- t.refusals + 1
  | Ok program ->
      let optimized = optimize program in
      t.before <- Ssa_stats.add t.before (Ssa_stats.of_program program);
      t.after <- Ssa_stats.add t.after (Ssa_stats.of_program optimized);
      (match Ssa_check.run ~prepare:optimize plan ~bind with
      | Ssa_check.Agree -> t.agree <- t.agree + 1
      | Ssa_check.Agree_on_failure _ ->
          t.agree_on_failure <- t.agree_on_failure + 1
      | Ssa_check.Disagree d ->
          t.disagreements <-
            Fmt.str "%a" Ssa_check.Disagreement.pp d :: t.disagreements
      | Ssa_check.Refused _ -> t.refusals <- t.refusals + 1);
      (* the logical work, up to a failure as well as at the end *)
      let before = counters plan program ~bind
      and after = counters plan optimized ~bind in
      if marks before <> marks after then t.work_differs <- t.work_differs + 1;
      (* a pass may remove reads, never add one *)
      if Ssa_interp.Counters.loads after > Ssa_interp.Counters.loads before then
        t.more_loads <- t.more_loads + 1

let verify _ppf (s : Native_op_walk.Subject.t) =
  let t = tally s.Native_op_walk.Subject.target in
  let prog = Eval_symbolic.run s.Native_op_walk.Subject.graph in
  (match Kernel_adapt.of_stage_program prog with
  | Error _ -> ()
  | Ok kernel ->
      let bind id = List.assoc_opt id s.Native_op_walk.Subject.inputs in
      List.iter
        (fun plan -> record t plan ~bind)
        [ Fusion_plan.default kernel; fst (Fusion_plan.plan kernel) ]);
  true

let silent = Format.make_formatter (fun _ _ _ -> ()) (fun () -> ())

let sweep ~shard ~shards =
  List.iteri
    (fun index (m : Native_op_walk.op) ->
      if index mod shards = shard then
        ignore
          (Walk_core.Walk.run m ~verify ~ppf:silent
             ~pcg:(Walk_core.Pcg.seed ~seed:(Int64.of_int index) ~seq:1L)
             ~steps:5))
    Native_op_walk.all_walks

let report () =
  let rows =
    Hashtbl.fold (fun target t acc -> (target, t) :: acc) tallies []
    |> List.sort (fun (a, _) (b, _) -> String.compare a b)
  in
  List.iter
    (fun (target, t) ->
      List.iter (fun d -> Fmt.pr "%s DISAGREES: %s@." target d) t.disagreements)
    rows;
  Fmt.pr "disagreements: %d, work differs: %d, more loads: %d@."
    (List.fold_left (fun n (_, t) -> n + List.length t.disagreements) 0 rows)
    (List.fold_left (fun n (_, t) -> n + t.work_differs) 0 rows)
    (List.fold_left (fun n (_, t) -> n + t.more_loads) 0 rows);
  List.iter
    (fun (target, t) ->
      Fmt.pr
        "%-24s agree=%d failed-alike=%d instrs %d -> %d, checked %d -> %d, \
         loops %d -> %d@."
        target t.agree t.agree_on_failure t.before.Ssa_stats.instrs
        t.after.Ssa_stats.instrs t.before.Ssa_stats.checked
        t.after.Ssa_stats.checked t.before.Ssa_stats.loops
        t.after.Ssa_stats.loops)
    rows
