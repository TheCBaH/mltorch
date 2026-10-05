open Ssa_bridge

(* The op sweep, through direct SSA lowering: every native op's own random walk,
   each subject adapted to a Kernel and run under both placements (everything
   stored, and the planner's fused plan), against [Kernel_eval]. The same
   subjects the Loop IR sweep runs, so a refusal here is a measured gap against
   the Loop lowering's coverage.

   A refusal is recorded, not hidden: the report is the inventory of what the
   lowering does not yet cover. Nothing may DISAGREE. *)

type tally = {
  mutable agree : int;
  mutable agree_on_failure : int;
  mutable disagreements : string list;
  mutable refusals : string list;
      (** constructs, deduplicated, in first-seen order *)
  mutable not_a_kernel : int;
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
          refusals = [];
          not_a_kernel = 0;
        }
      in
      Hashtbl.add tallies target t;
      t

let record t (verdict : Ssa_check.verdict) =
  match verdict with
  | Ssa_check.Agree -> t.agree <- t.agree + 1
  | Ssa_check.Agree_on_failure _ -> t.agree_on_failure <- t.agree_on_failure + 1
  | Ssa_check.Disagree d ->
      t.disagreements <- Fmt.str "%a" Ssa_check.Disagreement.pp d :: t.disagreements
  | Ssa_check.Refused u ->
      let name =
        Ssa_lower.Ssa_unsupported.construct_name u.Ssa_lower.Ssa_unsupported.construct
      in
      if not (List.mem name t.refusals) then t.refusals <- t.refusals @ [ name ]

let verify _ppf (s : Native_op_walk.Subject.t) =
  let t = tally s.Native_op_walk.Subject.target in
  let prog = Eval_symbolic.run s.Native_op_walk.Subject.graph in
  (match Kernel_adapt.of_stage_program prog with
  | Error _ -> t.not_a_kernel <- t.not_a_kernel + 1
  | Ok kernel ->
      let bind id = List.assoc_opt id s.Native_op_walk.Subject.inputs in
      List.iter
        (fun plan -> record t (Ssa_check.run plan ~bind))
        [ Fusion_plan.default kernel; fst (Fusion_plan.plan kernel) ]);
  true

let silent = Format.make_formatter (fun _ _ _ -> ()) (fun () -> ())

(* The ops are dealt round-robin into [shards] slices, one test file each. The
   seed stays the op's index in [all_walks], so a slice walks exactly what the
   Loop sweep's does. *)
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
  Fmt.pr "disagreements: %d@."
    (List.fold_left (fun n (_, t) -> n + List.length t.disagreements) 0 rows);
  List.iter
    (fun (target, t) ->
      Fmt.pr "%-28s agree=%d failed-alike=%d%s%s@." target t.agree
        t.agree_on_failure
        (if t.refusals = [] then ""
         else " refused: " ^ String.concat ", " t.refusals)
        (if t.not_a_kernel > 0 then Fmt.str " not-a-kernel=%d" t.not_a_kernel
         else ""))
    rows
