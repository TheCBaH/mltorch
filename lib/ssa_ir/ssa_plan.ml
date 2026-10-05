module P = Ssa_numerics.Precision

type t = {
  numerics : Ssa_numerics.t;
  precision : P.t;
  program : Ssa_program.t;
  refusal : Ssa_numerics.Refusal.t option;
  vectorized : Ssa_vectorize.report;
  sums : Ssa_vector_sum.report;
  blocked : int;
  contracted : int;
}

let count_vector_loops report =
  List.length
    (List.filter
       (fun (d : Ssa_vectorize.Decision.t) ->
         d.Ssa_vectorize.Decision.outcome = Ssa_vectorize.Decision.Vectorized)
       report)

let scheduled report =
  List.exists
    (fun (d : Ssa_vector_sum.Decision.t) ->
      match d.Ssa_vector_sum.Decision.outcome with
      | Ssa_vector_sum.Decision.Scheduled _ -> true
      | Ssa_vector_sum.Decision.Kept_sequential _ -> false)
    report

let run ~alias passes p = fst (Ssa_opt.run ~alias ~passes p)

(* The passes every plan runs before it looks for vectors: what the vectorizer
   relies on (checks proved away, invariants hoisted). *)
let before ~alias =
  [ Ssa_opt.simplify; Ssa_opt.guards; Ssa_opt.simplify; Ssa_opt.hoist ~alias ]

let after ~alias =
  [
    Ssa_opt.block ~alias ~group:Ssa_opt_block.Auto;
    Ssa_opt.hoist ~alias;
    Ssa_opt.share ~alias;
    Ssa_opt.simplify;
  ]

let resolve ?target ?(alias = Ssa_effects.Conservative) ~numerics
    (p : Ssa_program.t) =
  let scalar = run ~alias (before ~alias) p in
  let f64 refusal =
    match target with
    | None ->
        {
          numerics;
          precision = P.F64;
          program = run ~alias (after ~alias) scalar;
          refusal;
          vectorized = [];
          sums = [];
          blocked = 0;
          contracted = 0;
        }
    | Some target ->
        let q, vectorized = Ssa_vectorize.program ~alias ~target scalar in
        let q, blocked =
          Ssa_opt_rows.program ~rows:target.Ssa_target.row_block ~policy:alias q
        in
        {
          numerics;
          precision = P.F64;
          program = run ~alias (after ~alias) q;
          refusal;
          vectorized;
          sums = [];
          blocked;
          contracted = 0;
        }
  in
  match (numerics, target) with
  | Ssa_numerics.Reference_f64, _ | _, None -> f64 None
  | ( (Ssa_numerics.Simd_fp32_ordered | Ssa_numerics.Simd_fp32_relaxed),
      Some target ) -> (
      match Ssa_numerics.admit scalar with
      | Error r -> f64 (Some r)
      | Ok () ->
          let f32_target = Ssa_target.f32 target in
          let narrowed = Ssa_precision.to_f32 scalar in
          let q, vectorized =
            Ssa_vectorize.program ~alias ~target:f32_target narrowed
          in
          (* Contraction first, so a scheduled sum's own accumulate, which a
             fused operation would change, is never a candidate: fusing it is a
             separate permission the default plan does not take. *)
          let q =
            if
              Ssa_numerics.contraction_permitted numerics
              && (f32_target.Ssa_target.fma
                || f32_target.Ssa_target.relaxed_madd)
            then
              (* a relaxed multiply-add is vector-only: no scalar form *)
              fst (Ssa_opt_contract.pass ~scalar:f32_target.Ssa_target.fma q)
            else q
          in
          let q, sums =
            if Ssa_numerics.reassociation_permitted numerics then
              Ssa_vector_sum.program ~target:f32_target q
            else (q, [])
          in
          if count_vector_loops vectorized > 0 || scheduled sums then
            let q, blocked =
              Ssa_opt_rows.program ~rows:f32_target.Ssa_target.row_block
                ~policy:alias q
            in
            let q = run ~alias (after ~alias) q in
            {
              numerics;
              precision = P.F32;
              program = q;
              refusal = None;
              vectorized;
              sums;
              blocked;
              contracted = Ssa_opt_contract.contracted q;
            }
          else f64 None)

let oracle t = Ssa_vec_expand.program t.program
