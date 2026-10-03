module P = Loop_numerics.Precision

type t = {
  numerics : Loop_numerics.t;
  precision : P.t;
  vector : Loop_vector.program option;
  refusal : Loop_numerics.Refusal.t option;
  report : Loop_vectorize.report;
  blocked : int;
}

let plan ?target p =
  Option.map (fun target -> Loop_vectorize.program ~target p) target

let split = function
  | Some (vp, report) -> (Some vp, report)
  | None -> (None, [])

let resolve ?target ?fuse_reductions ~numerics p =
  let f64 refusal =
    let vector, report = split (plan ?target p) in
    { numerics; precision = P.F64; vector; refusal; report; blocked = 0 }
  in
  match (numerics, target) with
  | Loop_numerics.Reference_f64, _ | _, None -> f64 None
  | (Loop_numerics.Simd_fp32_ordered | Loop_numerics.Simd_fp32_relaxed), Some t
    -> (
      match Loop_numerics.admit p with
      | Error r -> f64 (Some r)
      | Ok () ->
          let vp, report =
            Loop_vectorize.program ~target:(Loop_target.f32 t)
              ~reductions:(Loop_numerics.reassociation_permitted numerics)
              p
          in
          if Loop_vector.count_vector_loops vp.Loop_vector.body > 0 then
            (* Rows blocked first: contraction rewrites each copy alike. *)
            let vp, blocked =
              Loop_block.program ~target:(Loop_target.f32 t) vp
            in
            (* Contraction, where the policy permits it and the target has a
               fused operation: after the sums are scheduled. *)
            let vp =
              let f32 = Loop_target.f32 t in
              if
                Loop_numerics.contraction_permitted numerics
                && (f32.Loop_target.fma || f32.Loop_target.relaxed_madd)
              then
                (* A relaxed multiply-add is vector-only: no scalar form. *)
                Loop_contract.program ?fuse_reductions
                  ~scalar:f32.Loop_target.fma vp
              else vp
            in
            {
              numerics;
              precision = P.F32;
              vector = Some vp;
              refusal = None;
              report;
              blocked;
            }
          else f64 None)

let force ?target ~precision p =
  match precision with
  | P.F64 ->
      let vector, report = split (plan ?target p) in
      Ok
        {
          numerics = Loop_numerics.Reference_f64;
          precision;
          vector;
          refusal = None;
          report;
          blocked = 0;
        }
  | P.F32 -> (
      match Loop_numerics.admit p with
      | Error r -> Error r
      | Ok () ->
          let vector, report =
            split
              (Option.map
                 (fun t -> Loop_vectorize.program ~target:(Loop_target.f32 t) p)
                 target)
          in
          Ok
            {
              numerics = Loop_numerics.Simd_fp32_ordered;
              precision;
              vector;
              refusal = None;
              report;
              blocked = 0;
            })

let oracle t (p : Loop_program.t) =
  match t.vector with
  | Some vp -> Loop_vector_expand.expand vp
  | None -> Loop_sum.program p
