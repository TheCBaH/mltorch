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
  | Refused of Ssa_lower.Ssa_unsupported.t

let pp_verdict fmt = function
  | Agree -> Fmt.string fmt "agree"
  | Agree_on_failure kind -> Fmt.pf fmt "agree on failure: %s" kind
  | Disagree d -> Fmt.pf fmt "DISAGREE: %a" Disagreement.pp d
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
