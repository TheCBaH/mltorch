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
let before ?target ~alias () =
  [ Ssa_opt.simplify; Ssa_opt.guards; Ssa_opt.simplify ]
  @ (match target with
    | Some _ -> [ Ssa_opt.convert_ifs; Ssa_opt.simplify ]
    | None -> [])
  @ [ Ssa_opt.hoist ~alias ]

let after ~alias =
  [
    Ssa_opt.block ~alias ~group:Ssa_opt_block.Auto;
    Ssa_opt.hoist ~alias;
    Ssa_opt.share ~alias;
    Ssa_opt.simplify;
  ]

let vectorize_f32 ?(alias = Ssa_effects.Conservative) ?elements ~target ~exact
    (p : Ssa_program.t) =
  let scalar = run ~alias (before ~target ~alias ()) p in
  Result.map
    (fun () ->
      Ssa_vectorize.program ~alias ?elements ~target:(Ssa_target.f32 target)
        (Ssa_precision.to_f32 ~exact scalar))
    (Ssa_numerics.admit scalar)

(* Every kernel binary32 and nothing vectorized: the scalar program, its cells
   read and written as binary32. *)
let fp32_scalar ~alias ~vectorized scalar =
  {
    numerics = Ssa_numerics.Fp32_ordered;
    precision = P.F32;
    program =
      run ~alias (after ~alias) (Ssa_precision.to_f32 ~exact:true scalar);
    refusal = None;
    vectorized;
    sums = [];
    blocked = 0;
    contracted = 0;
  }

let fp32_vector ?(alias = Ssa_effects.Conservative) ~target (p : Ssa_program.t)
    =
  Result.map
    (fun (q, vectorized) ->
      if count_vector_loops vectorized = 0 then
        fp32_scalar ~alias ~vectorized (run ~alias (before ~alias ()) p)
      else
        let q, blocked =
          Ssa_opt_rows.program
            ~rows:(Ssa_target.f32 target).Ssa_target.row_block ~policy:alias q
        in
        let q = run ~alias (after ~alias) q in
        (match
           List.filter
             (fun ((v : Ssa_value.t), _) ->
               match v.Ssa_value.ty with
               | Ssa_type.Vec _ -> true
               | Ssa_type.Effect | Ssa_type.Local | Ssa_type.Mask _
               | Ssa_type.Scalar _ ->
                   false)
             (Ssa_numerics.wide_values q)
         with
        | [] -> ()
        | wide ->
            invalid_arg
              (Fmt.str "Ssa_plan: a binary32 vector plan holds %a"
                 Ssa_numerics.pp_wide wide));
        {
          numerics = Ssa_numerics.Fp32_ordered;
          precision = P.F32;
          program = q;
          refusal = None;
          vectorized;
          sums = [];
          blocked;
          contracted = 0;
        })
    (vectorize_f32 ~alias ~elements:[ Ssa_type.F32 ] ~target ~exact:true p)

(* The binary64 plan: every kernel under [Reference_f64], or one an fp32 policy
   leaves binary64, with the target's binary64 vectors when there is one. *)
let reference ~alias ~numerics ~refusal ?target (p : Ssa_program.t) =
  let scalar = run ~alias (before ?target ~alias ()) p in
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

(* The [Simd_fp32_*] plan: the exact binary32 form vectorized, every element
   admitted (a kernel that does not vectorize is planned binary64, so its
   reads stay binary64 decodes); binary32 only when a loop
   vectorizes or a sum is scheduled, else the binary64 plan. *)
let simd_fp32 ~alias ~numerics ~target (p : Ssa_program.t) =
  let scalar = run ~alias (before ~target ~alias ()) p in
  match Ssa_numerics.admit scalar with
  | Error r -> reference ~alias ~numerics ~refusal:(Some r) ~target p
  | Ok () ->
      let f32_target = Ssa_target.f32 target in
      let narrowed = Ssa_precision.to_f32 ~exact:true scalar in
      let q, vectorized =
        Ssa_vectorize.program ~alias ~target:f32_target narrowed
      in
      (* Contraction first, so a scheduled sum's own accumulate, which a fused
         operation would change, is never a candidate: fusing it is a separate
         permission the default plan does not take. *)
      let q =
        if
          Ssa_numerics.contraction_permitted numerics
          && (f32_target.Ssa_target.fma || f32_target.Ssa_target.relaxed_madd)
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
      else reference ~alias ~numerics ~refusal:None ~target p

let resolve ?target ?(alias = Ssa_effects.Conservative) ~numerics
    (p : Ssa_program.t) =
  match (numerics, target) with
  | Ssa_numerics.Reference_f64, _ ->
      reference ~alias ~numerics ~refusal:None ?target p
  | (Ssa_numerics.Simd_fp32_ordered | Ssa_numerics.Simd_fp32_relaxed), None ->
      reference ~alias ~numerics ~refusal:None p
  | ( (Ssa_numerics.Simd_fp32_ordered | Ssa_numerics.Simd_fp32_relaxed),
      Some target ) ->
      simd_fp32 ~alias ~numerics ~target p
  | (Ssa_numerics.Fp32_contracted | Ssa_numerics.Fp32_ordered), None -> (
      (* every kernel binary32, nothing vectorized; a contracted plan is the
         ordered one, its sums fused only when the Machine IR lowering forms
         their accumulations *)
      let scalar = run ~alias (before ~alias ()) p in
      match Ssa_numerics.admit scalar with
      | Error r -> reference ~alias ~numerics ~refusal:(Some r) p
      | Ok () -> { (fp32_scalar ~alias ~vectorized:[] scalar) with numerics })
  | (Ssa_numerics.Fp32_contracted | Ssa_numerics.Fp32_ordered), Some target -> (
      match fp32_vector ~alias ~target p with
      | Ok t -> { t with numerics }
      | Error r -> reference ~alias ~numerics ~refusal:(Some r) p)

let admitted t =
  match (t.numerics, t.refusal) with
  | (Ssa_numerics.Fp32_contracted | Ssa_numerics.Fp32_ordered), Some r ->
      Error r
  | ( ( Ssa_numerics.Fp32_contracted | Ssa_numerics.Fp32_ordered
      | Ssa_numerics.Reference_f64 | Ssa_numerics.Simd_fp32_ordered
      | Ssa_numerics.Simd_fp32_relaxed ),
      (Some _ | None) ) ->
      Ok t

let oracle t = Ssa_vec_expand.program t.program
