open Ssa_bridge
open Ssa_ir

(* The op sweep through generated JavaScript: every native op's random walk,
   both placements, against [Kernel_eval]. A refusal is counted, never hidden;
   nothing may disagree. *)

type tally = {
  mutable agree : int;
  mutable agree_on_failure : int;
  mutable disagreements : string list;
  mutable refused : int;
}

let tallies : (string, tally) Hashtbl.t = Hashtbl.create 8

let tally config =
  match Hashtbl.find_opt tallies config with
  | Some t -> t
  | None ->
      let t =
        { agree = 0; agree_on_failure = 0; disagreements = []; refused = 0 }
      in
      Hashtbl.add tallies config t;
      t

let record t = function
  | Ssa_js_check.Refused _ -> t.refused <- t.refused + 1
  | Ssa_js_check.Verdict v -> (
      match v with
      | Ssa_check.Agree -> t.agree <- t.agree + 1
      | Ssa_check.Agree_on_failure _ ->
          t.agree_on_failure <- t.agree_on_failure + 1
      | Ssa_check.Disagree d ->
          t.disagreements <-
            Fmt.str "%a" Ssa_check.Disagreement.pp d :: t.disagreements
      | Ssa_check.Not_admitted _ | Ssa_check.Refused _ ->
          t.refused <- t.refused + 1)

let verify ~config ~prepare _ppf (s : Native_op_walk.Subject.t) =
  let t = tally config in
  let prog = Eval_symbolic.run s.Native_op_walk.Subject.graph in
  (match Kernel_adapt.of_stage_program prog with
  | Error _ -> ()
  | Ok kernel ->
      let bind id = List.assoc_opt id s.Native_op_walk.Subject.inputs in
      List.iter
        (fun plan -> record t (Ssa_js_check.run ~prepare plan ~bind))
        [ Fusion_plan.default kernel; fst (Fusion_plan.plan kernel) ]);
  true

let silent = Format.make_formatter (fun _ _ _ -> ()) (fun () -> ())

let sweep ~config ~prepare ~shard ~shards =
  List.iteri
    (fun index (m : Native_op_walk.op) ->
      if index mod shards = shard then
        ignore
          (Walk_core.Walk.run m ~verify:(verify ~config ~prepare) ~ppf:silent
             ~pcg:(Walk_core.Pcg.seed ~seed:(Int64.of_int index) ~seq:1L)
             ~steps:5))
    Native_op_walk.all_walks

let report ~config =
  let t = tally config in
  List.iter (fun d -> Fmt.pr "DISAGREES: %s@." d) t.disagreements;
  Fmt.pr "%s: disagreements %d, agree %d, failed alike %d, refused %d@." config
    (List.length t.disagreements)
    t.agree t.agree_on_failure t.refused

let alias = Ssa_effects.Distinct_buffers
let optimize p = fst (Ssa_opt.run ~alias p)
