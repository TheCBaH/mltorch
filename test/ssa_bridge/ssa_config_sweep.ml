open Ssa_bridge

(* The op sweep for one planning configuration: every native op's random walk,
   lowered, planned as the configuration says and checked against that
   configuration's own oracle. A configuration is a name and a function from a
   plan and its bindings to a verdict; the walks, seeds and slices are the direct
   sweep's, so a slice here covers exactly the ops its namesake does. *)

type tally = {
  mutable agree : int;
  mutable agree_on_failure : int;
  mutable disagreements : string list;
  mutable not_admitted : int;
  mutable not_a_kernel : int;
  mutable refusals : int;
}

let tallies : (string * string, tally) Hashtbl.t = Hashtbl.create 64

let tally config target =
  match Hashtbl.find_opt tallies (config, target) with
  | Some t -> t
  | None ->
      let t =
        {
          agree = 0;
          agree_on_failure = 0;
          disagreements = [];
          not_admitted = 0;
          not_a_kernel = 0;
          refusals = 0;
        }
      in
      Hashtbl.add tallies (config, target) t;
      t

let record t (verdict : Ssa_check.verdict) =
  match verdict with
  | Ssa_check.Agree -> t.agree <- t.agree + 1
  | Ssa_check.Agree_on_failure _ -> t.agree_on_failure <- t.agree_on_failure + 1
  | Ssa_check.Disagree d ->
      t.disagreements <-
        Fmt.str "%a" Ssa_check.Disagreement.pp d :: t.disagreements
  | Ssa_check.Not_admitted _ -> t.not_admitted <- t.not_admitted + 1
  | Ssa_check.Refused _ -> t.refusals <- t.refusals + 1

let verify ~config ~check _ppf (s : Native_op_walk.Subject.t) =
  let t = tally config s.Native_op_walk.Subject.target in
  let prog = Eval_symbolic.run s.Native_op_walk.Subject.graph in
  (match Kernel_adapt.of_stage_program prog with
  | Error _ -> t.not_a_kernel <- t.not_a_kernel + 1
  | Ok kernel ->
      let bind id = List.assoc_opt id s.Native_op_walk.Subject.inputs in
      List.iter
        (fun plan -> record t (check plan ~bind))
        [ Fusion_plan.default kernel; fst (Fusion_plan.plan kernel) ]);
  true

let silent = Format.make_formatter (fun _ _ _ -> ()) (fun () -> ())

let sweep ~config ~check ~shard ~shards =
  List.iteri
    (fun index (m : Native_op_walk.op) ->
      if index mod shards = shard then
        ignore
          (Walk_core.Walk.run m ~verify:(verify ~config ~check) ~ppf:silent
             ~pcg:(Walk_core.Pcg.seed ~seed:(Int64.of_int index) ~seq:1L)
             ~steps:5))
    Native_op_walk.all_walks

let report ~config =
  let rows =
    Hashtbl.fold
      (fun (c, target) t acc -> if c = config then (target, t) :: acc else acc)
      tallies []
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
      Fmt.pr "%-28s agree=%d failed-alike=%d%s%s%s@." target t.agree
        t.agree_on_failure
        (if t.not_admitted > 0 then Fmt.str " not-admitted=%d" t.not_admitted
         else "")
        (if t.refusals > 0 then Fmt.str " refused=%d" t.refusals else "")
        (if t.not_a_kernel > 0 then Fmt.str " not-a-kernel=%d" t.not_a_kernel
         else ""))
    rows
