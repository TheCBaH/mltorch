open Loop_ir
module Refusal = Lowering_artifact.Refusal
module Metric = Lowering_metric

type t = {
  node : Graph_ir.Node_id.t;
  precision : Loop_numerics.Precision.t option;
  vectorized : bool;
  vector_loops : int;
  scalar_loops : int;
  ssa_refusal : Refusal.t option;
  coral_cycles : float option;
}

let metric s = function
  | Metric.Coral_cycles -> s.coral_cycles
  | Metric.Scalar_loops -> Some (float_of_int s.scalar_loops)
  | Metric.Vector_loops -> Some (float_of_int s.vector_loops)

let empty node =
  {
    node;
    precision = None;
    vectorized = false;
    vector_loops = 0;
    scalar_loops = 0;
    ssa_refusal = None;
    coral_cycles = None;
  }

let precision : Ssa_ir.Ssa_numerics.Precision.t -> Loop_numerics.Precision.t =
  function
  | Ssa_ir.Ssa_numerics.Precision.F32 -> Loop_numerics.Precision.F32
  | Ssa_ir.Ssa_numerics.Precision.F64 -> Loop_numerics.Precision.F64

let of_invocation ~granted (inv : Loop_bundle.invocation) =
  let node = inv.Loop_bundle.node in
  if granted node then
    { (empty node) with ssa_refusal = Some (Refusal.Granted node) }
  else
    let s =
      match Ssa_backends.lowered inv with
      | Error r ->
          { (empty node) with ssa_refusal = Some (Refusal.Ssa_lowering r) }
      | Ok _ -> (
          match
            Ssa_backends.plan ~numerics:Ssa_ir.Ssa_numerics.Simd_fp32_relaxed
              ~target:Ssa_ir.Ssa_target.neon128 inv
          with
          | Error e ->
              { (empty node) with ssa_refusal = Some (Refusal.Ssa_backends e) }
          | Ok plan ->
              let module D = Ssa_ir.Ssa_vectorize.Decision in
              let vec, sca =
                List.fold_left
                  (fun (v, s) (d : D.t) ->
                    match d.D.outcome with
                    | D.Vectorized -> (v + 1, s)
                    | D.Kept_scalar _ -> (v, s + 1))
                  (0, 0) plan.Ssa_ir.Ssa_plan.vectorized
              in
              {
                (empty node) with
                precision = Some (precision plan.Ssa_ir.Ssa_plan.precision);
                vectorized = vec > 0;
                vector_loops = vec;
                scalar_loops = sca;
              })
    in
    {
      s with
      coral_cycles =
        Result.to_option
          (Result.map Lowering_coral.estimate (Lowering_coral.census inv));
    }

let of_bundle ~granted (b : Loop_bundle.t) =
  List.map (of_invocation ~granted) b.Loop_bundle.invocations
