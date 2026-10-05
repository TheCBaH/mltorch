open Ssa_ir

module Disagreement = struct
  type t =
    | Error_kind of { reference : string; ssa : string }
    | Error_payload of string
    | Invalid_program of Ssa_verify.diagnostic
    | Missing_output of Tensor_id.t
    | Reference_only_failed of string
    | Ssa_only_failed of string
    | Unexpected_output of Tensor_id.t
    | Value_mismatch of Tensor_id.t

  let pp fmt = function
    | Error_kind { reference; ssa } ->
        Fmt.pf fmt "failure kinds differ: reference %s, ssa %s" reference ssa
    | Error_payload kind -> Fmt.pf fmt "%s payloads differ" kind
    | Invalid_program d ->
        Fmt.pf fmt "the lowered program does not verify: %a"
          Ssa_verify.pp_diagnostic d
    | Missing_output id ->
        Fmt.pf fmt "ssa produced no output for %a" Tensor_id.pp id
    | Reference_only_failed kind ->
        Fmt.pf fmt "only the reference failed: %s" kind
    | Ssa_only_failed kind -> Fmt.pf fmt "only ssa failed: %s" kind
    | Unexpected_output id ->
        Fmt.pf fmt "ssa produced an extra output %a" Tensor_id.pp id
    | Value_mismatch id -> Fmt.pf fmt "%a differs bitwise" Tensor_id.pp id
end

type verdict =
  | Agree
  | Agree_on_failure of string
  | Disagree of Disagreement.t
  | Not_admitted of Ssa_numerics.Refusal.t
  | Refused of Ssa_lower.Ssa_unsupported.t

let pp_verdict fmt = function
  | Agree -> Fmt.string fmt "agree"
  | Agree_on_failure kind -> Fmt.pf fmt "agree on failure: %s" kind
  | Disagree d -> Fmt.pf fmt "DISAGREE: %a" Disagreement.pp d
  | Not_admitted r -> Fmt.pf fmt "not admitted: %a" Ssa_numerics.Refusal.pp r
  | Refused u -> Fmt.pf fmt "refused: %a" Ssa_lower.Ssa_unsupported.pp u

(* The first output, in id order, the other side lacks or holds differently. *)
let compare_outputs reference ssa =
  let absent in_ other =
    Tensor_id.Map.fold
      (fun id _ acc ->
        match acc with
        | Some _ -> acc
        | None -> if Tensor_id.Map.mem id other then None else Some id)
      in_ None
  in
  match absent reference ssa with
  | Some id -> Disagree (Disagreement.Missing_output id)
  | None -> (
      match absent ssa reference with
      | Some id -> Disagree (Disagreement.Unexpected_output id)
      | None -> (
          match
            Tensor_id.Map.fold
              (fun id r acc ->
                match acc with
                | Some _ -> acc
                | None ->
                    if
                      Loop_ir.Loop_check.tensors_equal r
                        (Tensor_id.Map.find id ssa)
                    then None
                    else Some id)
              reference None
          with
          | Some id -> Disagree (Disagreement.Value_mismatch id)
          | None -> Agree))

let compare_results reference ssa =
  match (reference, ssa) with
  | Ok reference, Ok ssa -> compare_outputs reference ssa
  | _, Error (`Invalid_program d) -> Disagree (Disagreement.Invalid_program d)
  | Error r, Error (#Kernel_eval.error as s) ->
      let reference_kind = Loop_ir.Loop_check.kind r
      and ssa_kind = Loop_ir.Loop_check.kind s in
      if not (String.equal reference_kind ssa_kind) then
        Disagree
          (Disagreement.Error_kind
             { reference = reference_kind; ssa = ssa_kind })
      else if Stdlib.( = ) r s then Agree_on_failure reference_kind
      else Disagree (Disagreement.Error_payload reference_kind)
  | Ok _, Error (#Kernel_eval.error as s) ->
      Disagree (Disagreement.Ssa_only_failed (Loop_ir.Loop_check.kind s))
  | Error r, Ok _ ->
      Disagree (Disagreement.Reference_only_failed (Loop_ir.Loop_check.kind r))

let compare ~reference ~ssa =
  compare_results (Err.payload reference) (Err.payload ssa)

let run ?(prepare = Fun.id) (plan : Fusion_plan.t) ~bind =
  match Err.payload (Ssa_lower.Ssa_lower_plan.lower plan) with
  | Error (`Unsupported u) -> Refused u
  | Ok program ->
      compare
        ~reference:(Kernel_eval.run_plan plan ~bind)
        ~ssa:(Ssa_lower.Ssa_exec.run plan (prepare program) ~bind)

(* The binary32 oracle of a scalar kernel is the Loop interpreter at binary32:
   every float operation in binary64, rounded once. *)
let run_f32 ?(prepare = Fun.id) (plan : Fusion_plan.t) ~bind =
  match
    ( Err.payload (Ssa_lower.Ssa_lower_plan.lower plan),
      Err.payload (Loop_ir.Loop_lower.lower plan) )
  with
  | Error (`Unsupported u), _ -> Refused u
  | Ok _, Error (`Unsupported _) ->
      Disagree (Disagreement.Reference_only_failed "loop_unsupported")
  | Ok program, Ok loop -> (
      match Ssa_numerics.admit program with
      | Error r -> Not_admitted r
      | Ok () ->
          let program = prepare (Ssa_precision.to_f32 program) in
          let reference =
            Err.map_error
              (fun (e : Loop_ir.Loop_interp.error) -> (e :> Kernel_eval.error))
              (Loop_ir.Loop_interp.run
                 ~precision:Loop_ir.Loop_numerics.Precision.F32 loop ~bind)
          in
          compare ~reference ~ssa:(Ssa_lower.Ssa_exec.run plan program ~bind))

(* An outcome of the SSA executor as the reference's error vocabulary, for the
   comparison: an invalid program is a defect, never a failure row. *)
let as_reference (r : (_, Ssa_lower.Ssa_exec.error) Err.t) =
  Err.map_error
    (function
      | `Invalid_program _ ->
          invalid_arg "Ssa_check: the oracle does not verify"
      | (#Ssa_interp.failure | `Binding_mismatch _ | `Unbound_input _) as e ->
          (e :> Kernel_eval.error))
    r

(* A relaxed plan against the binary64 reference: every finite cell within the
   tolerance, every nonfinite cell the same kind. *)
let within_tolerance ~atol ~rtol reference ssa =
  match (Err.payload reference, Err.payload ssa) with
  | Ok reference, Ok ssa -> (
      let failing =
        Tensor_id.Map.fold
          (fun id r acc ->
            match (acc, Tensor_id.Map.find_opt id ssa) with
            | Some _, _ -> acc
            | None, None -> Some id
            | None, Some s ->
                let d =
                  Loop_ir.Loop_numeric_diff.compare ~atol ~rtol ~actual:s
                    ~reference:r
                in
                if d.Loop_ir.Loop_numeric_diff.failing = 0 then None
                else Some id)
          reference None
      in
      match failing with
      | Some id -> Disagree (Disagreement.Value_mismatch id)
      | None -> Agree)
  | _ -> compare_results (Err.payload reference) (Err.payload ssa)

let run_planned ?(alias = Ssa_effects.Distinct_buffers) ~numerics ~target
    (plan : Fusion_plan.t) ~bind =
  match Err.payload (Ssa_lower.Ssa_lower_plan.lower plan) with
  | Error (`Unsupported u) -> (Refused u, None)
  | Ok program ->
      let resolved = Ssa_plan.resolve ~target ~alias ~numerics program in
      let exec p = Ssa_lower.Ssa_exec.run plan p ~bind in
      let result = exec resolved.Ssa_plan.program in
      let reference = Kernel_eval.run_plan plan ~bind in
      (* the plan against its own oracle, bit for bit *)
      let own =
        compare_results
          (Err.payload (as_reference (exec (Ssa_plan.oracle resolved))))
          (Err.payload result)
      in
      let verdict =
        match own with
        | Agree | Agree_on_failure _ -> (
            match (resolved.Ssa_plan.precision, numerics) with
            | Ssa_numerics.Precision.F64, _ -> compare ~reference ~ssa:result
            | ( Ssa_numerics.Precision.F32,
                (Ssa_numerics.Reference_f64 | Ssa_numerics.Simd_fp32_ordered) )
              -> (
                match Err.payload (Loop_ir.Loop_lower.lower plan) with
                | Error (`Unsupported _) ->
                    Disagree
                      (Disagreement.Reference_only_failed "loop_unsupported")
                | Ok loop ->
                    let f32 =
                      Err.map_error
                        (fun (e : Loop_ir.Loop_interp.error) ->
                          (e :> Kernel_eval.error))
                        (Loop_ir.Loop_interp.run
                           ~precision:Loop_ir.Loop_numerics.Precision.F32 loop
                           ~bind)
                    in
                    compare ~reference:f32 ~ssa:result)
            | Ssa_numerics.Precision.F32, Ssa_numerics.Simd_fp32_relaxed ->
                within_tolerance ~atol:1e-4 ~rtol:1e-4 reference result)
        | Disagree _ | Not_admitted _ | Refused _ -> own
      in
      (verdict, Some resolved)

module Plan_comparison = struct
  type t = {
    ssa_f32 : bool;
    loop_f32 : bool;
    ssa_loops : int;
    loop_loops : int;
    ssa_blocked : int;
    loop_blocked : int;
  }
end

let compare_plans ?(alias = Ssa_effects.Distinct_buffers) ~numerics ~target
    (plan : Fusion_plan.t) =
  let loop_target =
    List.find_opt
      (fun (t : Loop_ir.Loop_target.t) ->
        String.equal t.Loop_ir.Loop_target.name target.Ssa_target.name)
      Loop_ir.Loop_target.all
  and loop_numerics =
    List.find_opt
      (fun n ->
        String.equal (Loop_ir.Loop_numerics.name n) (Ssa_numerics.name numerics))
      Loop_ir.Loop_numerics.all
  in
  match
    ( Err.payload (Ssa_lower.Ssa_lower_plan.lower plan),
      Err.payload (Loop_ir.Loop_lower.lower plan),
      loop_target,
      loop_numerics )
  with
  | Ok ssa, Ok loop, Some loop_target, Some loop_numerics ->
      let s = Ssa_plan.resolve ~target ~alias ~numerics ssa in
      let l =
        Loop_ir.Loop_plan.resolve ~target:loop_target ~numerics:loop_numerics
          loop
      in
      Some
        {
          Plan_comparison.ssa_f32 =
            s.Ssa_plan.precision = Ssa_numerics.Precision.F32;
          loop_f32 =
            l.Loop_ir.Loop_plan.precision = Loop_ir.Loop_numerics.Precision.F32;
          ssa_loops =
            List.length
              (List.filter
                 (fun (d : Ssa_vectorize.Decision.t) ->
                   d.Ssa_vectorize.Decision.outcome
                   = Ssa_vectorize.Decision.Vectorized)
                 s.Ssa_plan.vectorized);
          loop_loops =
            List.length
              (List.filter
                 (fun (d : Loop_ir.Loop_vectorize.Decision.t) ->
                   d.Loop_ir.Loop_vectorize.Decision.outcome
                   = Loop_ir.Loop_vectorize.Decision.Vectorized)
                 l.Loop_ir.Loop_plan.report);
          ssa_blocked = s.Ssa_plan.blocked;
          loop_blocked = l.Loop_ir.Loop_plan.blocked;
        }
  | _ -> None

type marks = {
  emitters : int;
  keys : int;
  locals : int;
  reductions : int;
  scan_updates : int;
  scans : int;
}

let ssa_marks c =
  let m = Ssa_interp.Counters.mark c in
  {
    emitters = m Ssa_mark.Emitter;
    keys = m Ssa_mark.Key;
    locals = m Ssa_mark.Local;
    reductions = m Ssa_mark.Reduction;
    scan_updates = m Ssa_mark.Scan_update;
    scans = m Ssa_mark.Scan;
  }

let loop_marks (c : Loop_ir.Loop_interp.counters) =
  {
    emitters = c.Loop_ir.Loop_interp.emitters;
    keys = c.Loop_ir.Loop_interp.keys;
    locals = c.Loop_ir.Loop_interp.locals;
    reductions = c.Loop_ir.Loop_interp.reductions;
    scan_updates = c.Loop_ir.Loop_interp.scan_updates;
    scans = c.Loop_ir.Loop_interp.scans;
  }

let marks (plan : Fusion_plan.t) ~bind =
  match
    ( Err.payload (Ssa_lower.Ssa_lower_plan.lower plan),
      Err.payload (Loop_ir.Loop_lower.lower plan) )
  with
  | Error (`Unsupported u), _ -> Error (`Refused u)
  | Ok _, Error (`Unsupported _) -> Error `Failed
  | Ok program, Ok loop -> (
      let ssa_counters = Ssa_interp.Counters.create () in
      let loop_counters = Loop_ir.Loop_interp.counters () in
      match
        ( Err.payload
            (Ssa_lower.Ssa_exec.run ~counters:ssa_counters plan program ~bind),
          Err.payload
            (Loop_ir.Loop_interp.run ~counters:loop_counters loop ~bind) )
      with
      | Ok _, Ok _ -> Ok (ssa_marks ssa_counters, loop_marks loop_counters)
      | _ -> Error `Failed)
