open Inference_compare_schema
module Metrics = Inference_compare_metrics

type result = {
  bindings : Policy_binding.t list;
  empty_matches : Id.Metric_rule.t list;
}

let ( let* ) = Err.bind
let fail e = Policy_error.fail e
let policy_key = Comparison_policy.preimage

let metric_key (m : Tensor_metric.t) =
  Wire.canonical_string (Tensor_metric.encode m)

type matched = {
  rule : Metric_rule.t;
  site : Comparison_site.t;
  witnesses : Topology_witness.t list;
}

let config_code (c : Metrics.Metric_config_error.t) : Failure.code =
  match c with
  | Duplicate_metric_id _ -> Policy_conflict
  | _ -> Unsupported_policy

let resolve ~ctx ~codecs ~env ~placeholder_digest sites
    (selection : Policy_selection.t) =
  let rules =
    List.stable_sort
      (fun (a : Metric_rule.t) (b : Metric_rule.t) ->
        Id.Metric_rule.compare a.id b.id)
      selection.rules
  in
  let limits = ctx.Selector_eval.limits in
  (* Explicit bindings name existing sites, once. *)
  let* () =
    let rec go seen = function
      | [] -> Ok ()
      | (c, _) :: rest ->
          if Comparison_sites.find sites c = None then
            fail (Policy_error.make ~comparison:c Unavailable_comparison_site)
          else if List.exists (Id.Comparison.equal c) seen then
            fail (Policy_error.make ~comparison:c Policy_conflict)
          else go (c :: seen) rest
    in
    go [] selection.explicit
  in
  let* () =
    let rec go = function
      | [] -> Ok ()
      | (r : Metric_rule.t) :: rest ->
          let* () = Selector_eval.check ctx sites ~rule:r.id r.selector in
          go rest
    in
    go rules
  in
  (* Rule matches per site, with witnesses. *)
  let matches = ref [] in
  let witness_total = ref 0 in
  let empty = ref [] in
  let* () =
    let rec per_rule = function
      | [] -> Ok ()
      | (r : Metric_rule.t) :: rest ->
          let hits =
            List.filter_map
              (fun site ->
                Option.map
                  (fun witnesses -> { rule = r; site; witnesses })
                  (Selector_eval.eval ctx site r.selector))
              sites
          in
          witness_total :=
            !witness_total
            + List.fold_left (fun a m -> a + List.length m.witnesses) 0 hits;
          if
            Int64.compare
              (Int64.of_int !witness_total)
              limits.max_retained_witnesses
            > 0
          then
            fail
              (Policy_error.make ~rule:r.id
                 ~detail:
                   (Limit
                      {
                        limit = Witnesses;
                        bound = limits.max_retained_witnesses;
                        actual = Int64.of_int !witness_total;
                      })
                 Resource_exhausted)
          else if hits = [] then (
            match r.empty_match with
            | Reject -> fail (Policy_error.make ~rule:r.id Empty_rule_match)
            | Allow ->
                empty := r.id :: !empty;
                per_rule rest)
          else (
            matches := !matches @ hits;
            per_rule rest)
    in
    per_rule rules
  in
  let for_site (s : Comparison_site.t) =
    List.filter
      (fun m -> Id.Comparison.equal m.site.comparison s.comparison)
      !matches
  in
  let rec per_site acc = function
    | [] -> Ok (List.rev acc)
    | (site : Comparison_site.t) :: rest ->
        let ms = for_site site in
        let selecting =
          List.filter_map
            (fun m ->
              match m.rule.action with
              | Select_policy p -> Some (m, p)
              | _ -> None)
            ms
        in
        let adders =
          List.concat_map
            (fun m ->
              match m.rule.action with Add_metrics l -> [ (m, l) ] | _ -> [])
            ms
        in
        let explicit =
          List.find_map
            (fun (c, p) ->
              if Id.Comparison.equal c site.comparison then Some p else None)
            selection.explicit
        in
        let conflict ?rule () =
          fail
            (Policy_error.make ?rule ~comparison:site.comparison ~unit:site.unit
               Policy_conflict)
        in
        let* base, base_rules =
          match explicit with
          | Some p -> (
              match
                List.find_opt
                  (fun (_, q) -> policy_key q <> policy_key p)
                  selecting
              with
              | Some (m, _) -> conflict ~rule:m.rule.id ()
              | None -> Ok ((p, Policy_binding.Explicit), []))
          | None -> (
              match selecting with
              | [] -> (
                  match selection.default with
                  | Some p -> Ok ((p, Policy_binding.Default), [])
                  | None ->
                      fail
                        (Policy_error.make ~comparison:site.comparison
                           ~unit:site.unit Missing_policy))
              | _ -> (
                  let top =
                    List.fold_left
                      (fun a (m, _) ->
                        max a (Rule_priority.to_int m.rule.priority))
                      min_int selecting
                  in
                  let tied =
                    List.filter
                      (fun (m, _) -> Rule_priority.to_int m.rule.priority = top)
                      selecting
                  in
                  let _, p = List.hd tied in
                  match
                    List.find_opt
                      (fun (_, q) -> policy_key q <> policy_key p)
                      tied
                  with
                  | Some (m, _) -> conflict ~rule:m.rule.id ()
                  | None ->
                      let ids = List.map (fun (m, _) -> m.rule.id) tied in
                      Ok ((p, Policy_binding.Rules ids), ids)))
        in
        let base_policy, base_kind = base in
        (* Additive metrics, from every matching rule. *)
        let* all_metrics =
          let known =
            ref (List.map (fun m -> (m, None)) base_policy.tensor_metrics)
          in
          let rec go = function
            | [] -> Ok ()
            | ((m : matched), metrics) :: more ->
                let rec each = function
                  | [] -> Ok ()
                  | (x : Tensor_metric.t) :: ms -> (
                      match
                        List.find_opt
                          (fun ((y : Tensor_metric.t), _) ->
                            Id.Tensor_metric.equal y.id x.id)
                          !known
                      with
                      | Some (y, _) ->
                          if metric_key y = metric_key x then each ms
                          else
                            fail
                              (Policy_error.make ~rule:m.rule.id
                                 ~comparison:site.comparison ~unit:site.unit
                                 ~metric:x.id Policy_conflict)
                      | None ->
                          known := !known @ [ (x, Some m.rule.id) ];
                          each ms)
                in
                let* () = each metrics in
                go more
          in
          let* () = go adders in
          Ok (List.map fst !known)
        in
        let policy =
          Comparison_policy.canonicalize
            { base_policy with tensor_metrics = all_metrics }
        in
        let* () =
          match Metrics.Policy_check.validate ~codecs env site.spec policy with
          | Ok () -> Ok ()
          | Error e ->
              let c = Err.Error.kind e in
              fail
                (Policy_error.make ~comparison:site.comparison ~unit:site.unit
                   ~detail:(Config c) (config_code c))
        in
        let rule_matches =
          List.map
            (fun m ->
              {
                Policy_binding.Rule_match.rule = m.rule.id;
                selected_as_base =
                  List.exists (Id.Metric_rule.equal m.rule.id) base_rules;
                witnesses = m.witnesses;
              })
            ms
        in
        let binding : Policy_binding.t =
          {
            site;
            policy;
            policy_digest = placeholder_digest;
            base = base_kind;
            matches = rule_matches;
          }
        in
        per_site (binding :: acc) rest
  in
  let* bindings = per_site [] sites in
  Ok { bindings; empty_matches = List.rev !empty }
