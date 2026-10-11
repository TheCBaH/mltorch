(* Candidates observed during a pass, and the comparison of two sets of final
   outputs under the declared fidelity. *)

open Inference_compare_schema
module W = Wire
module Codecs = Inference_compare_codecs
module L = Codecs.Logical_value
module Core = Inference_compare_core
module Pol = Inference_compare_policy
module M = Inference_compare_metrics
open Certify_base

type observed = (string, Tensor_spec.t * string) Hashtbl.t

let observed_key k side =
  Id.Canonical_value.to_string k ^ "/" ^ Side.to_string side

let observer (keys : Id.Canonical_value.t list) (table : observed)
    (s : Core.Coordinator.snapshot) =
  List.iter
    (fun (k, side, _, v) ->
      if List.exists (Id.Canonical_value.equal k) keys then
        Hashtbl.replace table (observed_key k side) (L.spec v, L.contents v))
    s.candidates

(* ---- comparing two sets of final outputs ------------------------------------- *)

type compared = { exact : bool }

let value ctx (spec, bytes) =
  Result.map_error
    (fun e -> Format.asprintf "%a" Codecs.Codec_error.pp (Err.Error.kind e))
    (L.make ctx spec bytes)

let bitwise (a : L.t) (b : L.t) =
  Tensor_spec.equal (L.spec a) (L.spec b)
  && String.equal (L.contents a) (L.contents b)

type how = Bitwise | Policy

let compare_outputs stage ~ctx ~algorithm ~(contract : Computation_contract.t)
    ~(finals : Final_output.t list) ~how
    ~(left : Id.Output_occurrence.t -> L.t option)
    ~(right : Id.Output_occurrence.t -> L.t option) ~what =
  let policy_digest = Pol.Seal.placeholder algorithm in
  let rec go exact = function
    | [] -> Ok { exact }
    | (o : Final_output.t) :: rest -> (
        let name = Id.Output_occurrence.to_string o.occurrence in
        match (left o.occurrence, right o.occurrence) with
        | None, _ | _, None ->
            fail stage Incomplete
              (Printf.sprintf "%s: output %s is missing" what name)
        | Some a, Some b -> (
            let same = bitwise a b in
            match how with
            | Bitwise ->
                if same then go exact rest
                else
                  fail stage Disagreement
                    (Printf.sprintf "%s: output %s is not bitwise equal" what
                       name)
            | Policy -> (
                let policy =
                  List.find_map
                    (fun (occ, p) ->
                      if Id.Output_occurrence.equal occ o.occurrence then Some p
                      else None)
                    contract.final_policies
                in
                match policy with
                | None ->
                    fail stage Invalid
                      (Printf.sprintf "output %s has no final policy" name)
                | Some policy -> (
                    let comparison_id =
                      match Id.Comparison.of_string ("final:" ^ name) with
                      | Ok c -> c
                      | Error _ -> invalid_arg "Certify.compare_outputs"
                    in
                    match
                      M.Comparison.compare ctx M.Tensor_metrics.empty_env policy
                        ~comparison_id ~policy_digest ~left:a ~right:b
                    with
                    | Error e ->
                        fail stage Failed
                          (Format.asprintf "%s: output %s: %a" what name
                             M.Comparison.pp_error (Err.Error.kind e))
                    | Ok c ->
                        if c.verdict = Pass then go (exact && same) rest
                        else
                          fail stage Disagreement
                            (Printf.sprintf
                               "%s: output %s fails its final policy" what name)
                    ))))
  in
  go true finals
