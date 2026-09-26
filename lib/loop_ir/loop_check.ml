module Disagreement = struct
  type t =
    | Error_kind of { reference : string; loop : string }
    | Error_payload of string
    | Executor of t
    | Host_failure of string
    | Loop_only_failed of string
    | Missing_output of Tensor_id.t
    | Reference_only_failed of string
    | Unexpected_output of Tensor_id.t
    | Value_mismatch of Tensor_id.t

  let rec pp fmt = function
    | Error_kind { reference; loop } ->
        Fmt.pf fmt "failure kinds differ: reference %s, loop %s" reference loop
    | Error_payload kind -> Fmt.pf fmt "%s payloads differ" kind
    | Executor d -> Fmt.pf fmt "the in-process executor: %a" pp d
    | Host_failure m -> Fmt.pf fmt "the executor's host failed: %s" m
    | Loop_only_failed kind -> Fmt.pf fmt "only the loop failed: %s" kind
    | Missing_output id ->
        Fmt.pf fmt "loop produced no output for %a" Tensor_id.pp id
    | Reference_only_failed kind ->
        Fmt.pf fmt "only the reference failed: %s" kind
    | Unexpected_output id ->
        Fmt.pf fmt "loop produced an extra output %a" Tensor_id.pp id
    | Value_mismatch id -> Fmt.pf fmt "%a differs bitwise" Tensor_id.pp id
end

type verdict =
  | Agree
  | Agree_on_failure of string
  | Disagree of Disagreement.t
  | Refused of Loop_unsupported.t

let pp_verdict fmt = function
  | Agree -> Fmt.string fmt "agree"
  | Agree_on_failure kind -> Fmt.pf fmt "agree on failure: %s" kind
  | Disagree d -> Fmt.pf fmt "DISAGREE: %a" Disagreement.pp d
  | Refused u -> Fmt.pf fmt "refused: %a" Loop_unsupported.pp u

let kind : Kernel_eval.error -> string = function
  | `Binding_mismatch _ -> "binding_mismatch"
  | `Coord_out_of_range _ -> "coord_out_of_range"
  | `Data_index_unexpected_here -> "data_index_unexpected_here"
  | `Data_source_wrong_format _ -> "data_source_wrong_format"
  | `Gather_index_out_of_range _ -> "gather_index_out_of_range"
  | `I64_division_by_zero -> "i64_division_by_zero"
  | `I64_division_overflow -> "i64_division_overflow"
  | `I64_from_float_infinite -> "i64_from_float_infinite"
  | `I64_from_float_nan -> "i64_from_float_nan"
  | `I64_from_float_out_of_range _ -> "i64_from_float_out_of_range"
  | `Index_not_exact_in_float _ -> "index_not_exact_in_float"
  | `Index_overflow _ -> "index_overflow"
  | `Non_positive_divisor _ -> "non_positive_divisor"
  | `Scan_meter _ -> "scan_meter"
  | `Scan_meter_required -> "scan_meter_required"
  | `Scan_projection _ -> "scan_projection"
  | `Unbound_input _ -> "unbound_input"
  | `Unbound_local _ -> "unbound_local"
  | `Unbound_reducer _ -> "unbound_reducer"
  | `Unknown_source _ -> "unknown_source"
  | _ -> "other"

(* Cells, not decoded floats: an int64 past [2^53] and a signed zero must both
   stay distinguishable. A NaN is compared as a NaN, whatever its payload or sign:
   a JavaScript engine does not portably preserve either, so an identity that must
   hold under js_of_ocaml as well as natively cannot ask for them. *)
let tensors_equal (Tensor.Tensor a as ta) (Tensor.Tensor b as tb) =
  Stdlib.( = ) a.Tensor.shape b.Tensor.shape
  &&
  match (a.Tensor.payload.Payload.fmt, b.Tensor.payload.Payload.fmt) with
  | Payload.I64, Payload.I64 ->
      let n = Bigarray.Array1.dim a.Tensor.payload.Payload.data in
      n = Bigarray.Array1.dim b.Tensor.payload.Payload.data
      &&
      let rec go i =
        i >= n
        || Int64.equal
             a.Tensor.payload.Payload.data.{i}
             b.Tensor.payload.Payload.data.{i}
           && go (i + 1)
      in
      go 0
  | Payload.I64, _ | _, Payload.I64 -> false
  | _ ->
      String.equal
        (let (Payload.Fmt f) = Payload.Fmt a.Tensor.payload.Payload.fmt in
         Payload.fmt_name f)
        (let (Payload.Fmt f) = Payload.Fmt b.Tensor.payload.Payload.fmt in
         Payload.fmt_name f)
      && Vec6.fold_coords a.Tensor.shape ~init:true ~f:(fun ok coord ->
          ok
          && Core.Float_bits.equal_portable (Tensor.read ta coord)
               (Tensor.read tb coord))

let compare_outputs reference loop =
  let missing =
    Tensor_id.Map.fold
      (fun id _ acc ->
        match acc with
        | Some _ -> acc
        | None ->
            if Tensor_id.Map.mem id loop then None
            else Some (Disagreement.Missing_output id))
      reference None
  in
  match missing with
  | Some d -> Disagree d
  | None -> (
      let extra =
        Tensor_id.Map.fold
          (fun id _ acc ->
            match acc with
            | Some _ -> acc
            | None ->
                if Tensor_id.Map.mem id reference then None
                else Some (Disagreement.Unexpected_output id))
          loop None
      in
      match extra with
      | Some d -> Disagree d
      | None -> (
          match
            Tensor_id.Map.fold
              (fun id r acc ->
                match acc with
                | Some _ -> acc
                | None ->
                    if tensors_equal r (Tensor_id.Map.find id loop) then None
                    else Some id)
              reference None
          with
          | Some id -> Disagree (Disagreement.Value_mismatch id)
          | None -> Agree))

let compare_results reference loop =
  match (reference, loop) with
  | Ok reference, Ok loop -> compare_outputs reference loop
  | Error r, Error l ->
      let l : Kernel_eval.error = (l :> Kernel_eval.error) in
      let reference_kind = kind r and loop_kind = kind l in
      if not (String.equal reference_kind loop_kind) then
        Disagree
          (Disagreement.Error_kind
             { reference = reference_kind; loop = loop_kind })
      else if Stdlib.( = ) r l then Agree_on_failure reference_kind
      else Disagree (Disagreement.Error_payload reference_kind)
  | Ok _, Error l ->
      Disagree (Disagreement.Loop_only_failed (kind (l :> Kernel_eval.error)))
  | Error r, Ok _ -> Disagree (Disagreement.Reference_only_failed (kind r))

let compare ~(reference : (_, Kernel_eval.error) Err.t)
    ~(loop : (_, Loop_interp.error) Err.t) =
  compare_results (Err.payload reference) (Err.payload loop)

module Executor = struct
  type error =
    [ Loop_interp.error | `Js_compile of string | `Js_exception of string ]

  type t =
    Loop_program.t ->
    bind:(Tensor_id.t -> Tensor.packed option) ->
    (Tensor.packed Tensor_id.Map.t, error) Err.t
end

let installed : Executor.t option ref = ref None
let install e = installed := e

(* A host that refused or threw is a defect verdict of its own: it can never
   match a failure the reference reported. *)
let compare_executor ~reference (result : (_, Executor.error) Err.t) =
  match Err.payload result with
  | Error (`Js_compile m | `Js_exception m) ->
      Disagree (Disagreement.Host_failure m)
  | Error (#Loop_interp.error as e) ->
      compare_results (Err.payload reference) (Error e)
  | Ok outputs -> compare_results (Err.payload reference) (Ok outputs)

let run ?exec (plan : Fusion_plan.t) ~bind =
  let exec = match exec with Some _ -> exec | None -> !installed in
  match Err.payload (Loop_lower.lower plan) with
  | Error (`Unsupported u) -> Refused u
  | Ok program -> (
      let reference = Kernel_eval.run_plan plan ~bind in
      let verdict = compare ~reference ~loop:(Loop_interp.run program ~bind) in
      match (verdict, exec) with
      | Disagree _, _ | _, None -> verdict
      | (Agree | Agree_on_failure _ | Refused _), Some exec -> (
          match compare_executor ~reference (exec program ~bind) with
          | Disagree d -> Disagree (Disagreement.Executor d)
          | Agree | Agree_on_failure _ | Refused _ -> verdict))
