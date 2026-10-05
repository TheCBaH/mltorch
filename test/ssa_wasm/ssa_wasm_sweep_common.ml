open Ssa_bridge
open Ssa_ir

(* The op sweep through generated WebAssembly: every native op's random walk, both
   placements, against [Kernel_eval]. A refusal (a quantized buffer) is counted,
   never hidden; nothing may disagree. *)

type tally = {
  mutable agree : int;
  mutable agree_on_failure : int;
  mutable disagreements : string list;
  mutable refused : int;
  mutable binary32 : int;
  mutable fused : int;
  mutable vectors : int;
}

let tallies : (string, tally) Hashtbl.t = Hashtbl.create 8

let tally config =
  match Hashtbl.find_opt tallies config with
  | Some t -> t
  | None ->
      let t =
        {
          agree = 0;
          agree_on_failure = 0;
          disagreements = [];
          refused = 0;
          binary32 = 0;
          fused = 0;
          vectors = 0;
        }
      in
      Hashtbl.add tallies config t;
      t

let record t = function
  | Ssa_wasm_check.Refused _ -> t.refused <- t.refused + 1
  | Ssa_wasm_check.Verdict (v, f) -> (
      if f.Ssa_wasm_check.relaxed then t.fused <- t.fused + 1;
      if f.Ssa_wasm_check.vectors then t.vectors <- t.vectors + 1;
      match v with
      | Ssa_check.Agree -> t.agree <- t.agree + 1
      | Ssa_check.Agree_on_failure _ ->
          t.agree_on_failure <- t.agree_on_failure + 1
      | Ssa_check.Disagree d ->
          t.disagreements <-
            Fmt.str "%a" Ssa_check.Disagreement.pp d :: t.disagreements
      | Ssa_check.Not_admitted _ | Ssa_check.Refused _ ->
          t.refused <- t.refused + 1)

type engine = Reference | Interpreter

let verify ~config ~prepare ~engine ~relaxed _ppf (s : Native_op_walk.Subject.t)
    =
  let t = tally config in
  let prog = Eval_symbolic.run s.Native_op_walk.Subject.graph in
  (match Kernel_adapt.of_stage_program prog with
  | Error _ -> ()
  | Ok kernel ->
      let bind id = List.assoc_opt id s.Native_op_walk.Subject.inputs in
      List.iter
        (fun plan ->
          record t
            (match engine with
            | Reference -> Ssa_wasm_check.run ~relaxed ~prepare plan ~bind
            | Interpreter ->
                Ssa_wasm_check.run_against_interpreter ~relaxed ~prepare plan
                  ~bind))
        [ Fusion_plan.default kernel; fst (Fusion_plan.plan kernel) ]);
  true

let silent = Format.make_formatter (fun _ _ _ -> ()) (fun () -> ())

let sweep ?(engine = Reference) ?(relaxed = false) ~config ~prepare ~shard
    ~shards () =
  List.iteri
    (fun index (m : Native_op_walk.op) ->
      if index mod shards = shard then
        ignore
          (Walk_core.Walk.run m
             ~verify:(verify ~config ~prepare ~engine ~relaxed)
             ~ppf:silent
             ~pcg:(Walk_core.Pcg.seed ~seed:(Int64.of_int index) ~seq:1L)
             ~steps:5))
    Native_op_walk.all_walks

let report ~config =
  let t = tally config in
  List.iter (fun d -> Fmt.pr "DISAGREES: %s@." d) t.disagreements;
  Fmt.pr
    "%s: disagreements %d, agree %d, failed alike %d, refused %d; with vectors \
     %d, relaxed %d@."
    config
    (List.length t.disagreements)
    t.agree t.agree_on_failure t.refused t.vectors t.fused

let alias = Ssa_effects.Distinct_buffers
let optimize p = fst (Ssa_opt.run ~alias p)

(* The programs a numerical plan makes for a target: the same pipeline the
   planner suites run, resolved per policy. *)
let planned ~numerics ~target p =
  (Ssa_plan.resolve ~target ~alias ~numerics p).Ssa_plan.program

(* Strict vectorization at binary64 for a target. *)
let vectorized ~target p =
  planned ~numerics:Ssa_numerics.Reference_f64 ~target p
