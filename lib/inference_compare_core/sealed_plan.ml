open Inference_compare_schema
module Plan = Inference_compare_plan
module Ex = Inference_compare_extract
module Pol = Inference_compare_policy
module H = Inference_compare_codecs.Payload_hasher
module Unit_plan = Plan.Correspondence_check.Unit_plan

type t = {
  structural : Plan.Structural.t;
  extractions : Ex.Extraction.t list;
  bindings : Policy_bindings.t;
  plan : Execution_plan.t;
}

type error =
  | Extraction_mismatch
  | Hasher of H.error
  | Missing_binding of Id.Comparison.t
  | Stale_binding of Id.Comparison.t
  | Stale_plan
  | Unbound_site of Id.Comparison.t

let pp_error ppf = function
  | Extraction_mismatch -> Fmt.string ppf "extractions do not match the plan"
  | Hasher e -> H.pp_error ppf e
  | Missing_binding c ->
      Fmt.pf ppf "no policy binding for %a" Id.Comparison.pp c
  | Stale_binding c ->
      Fmt.pf ppf "policy binding for %a is stale" Id.Comparison.pp c
  | Stale_plan -> Fmt.string ppf "decoded plan differs from the recomputed plan"
  | Unbound_site c ->
      Fmt.pf ppf "binding for unknown comparison %a" Id.Comparison.pp c

let ( let* ) = Err.bind
let fail e = Err.fail ~pp_error e

let hex ~hashers ~algorithm preimage =
  match H.digest_string hashers algorithm preimage with
  | Ok d -> Ok (Digest.Payload.hex d)
  | Error e -> fail (Hasher (Err.Error.kind e))

let digest make ~hashers ~algorithm preimage =
  let* h = hex ~hashers ~algorithm preimage in
  match make algorithm ~hex:h with
  | Ok d -> Ok d
  | Error _ -> fail (Hasher (Unknown_algorithm algorithm))

let site_key s = Wire.canonical_string (Comparison_site.encode s)

let seal ~hashers ~algorithm structural extractions
    (bindings : Policy_bindings.t) =
  let order = Plan.Unit_order.order (Plan.Structural.order structural) in
  (* Extractions: exactly one per unit and side, in execution order. *)
  let* () =
    let expected =
      List.concat_map (fun u -> [ (u, Side.Left); (u, Side.Right) ]) order
    in
    let actual =
      List.map (fun (e : Ex.Extraction.t) -> (e.unit, e.side)) extractions
    in
    if
      List.length expected = List.length actual
      && List.for_all2
           (fun (u, s) (u', s') -> Id.Unit.equal u u' && Side.equal s s')
           expected actual
      && List.for_all
           (fun (e : Ex.Extraction.t) ->
             let idx =
               match e.side with
               | Left -> Plan.Structural.left structural
               | Right -> Plan.Structural.right structural
             in
             Graph_key.equal e.descriptor.source_graph
               (Plan.Graph_index.key idx))
           extractions
    then Ok ()
    else fail Extraction_mismatch
  in
  (* Bindings: the plan's sites, exactly. *)
  let* sites =
    match Pol.Comparison_sites.build structural with
    | Ok s -> Ok s
    | Error _ -> fail Extraction_mismatch
  in
  let* () =
    let rec go = function
      | [] -> Ok ()
      | (s : Comparison_site.t) :: rest -> (
          match
            List.find_opt
              (fun (b : Policy_binding.t) ->
                Id.Comparison.equal b.site.comparison s.comparison)
              bindings.bindings
          with
          | None -> fail (Missing_binding s.comparison)
          | Some b ->
              if String.equal (site_key b.site) (site_key s) then go rest
              else fail (Stale_binding s.comparison))
    in
    go sites
  in
  let* () =
    match
      List.find_opt
        (fun (b : Policy_binding.t) ->
          not
            (List.exists
               (fun (s : Comparison_site.t) ->
                 Id.Comparison.equal s.comparison b.site.comparison)
               sites))
        bindings.bindings
    with
    | Some b -> fail (Unbound_site b.site.comparison)
    | None -> Ok ()
  in
  let mapping = Plan.Structural.mapping structural in
  let correspondence = Plan.Structural.correspondence structural in
  let unit_order = Plan.Structural.order structural in
  let* mapping_digest =
    digest Digest.Mapping.make ~hashers ~algorithm
      (Mapping_artifact.preimage mapping)
  in
  let* frontier_mapping_digest =
    digest Digest.Frontier_mapping.make ~hashers ~algorithm
      (Mapping_artifact.frontier_preimage mapping)
  in
  let planned =
    List.map
      (fun u ->
        let p =
          Option.get (Plan.Correspondence_check.unit_plan correspondence u)
        in
        let ex side =
          List.find
            (fun (e : Ex.Extraction.t) ->
              Id.Unit.equal e.unit u && Side.equal e.side side)
            extractions
        in
        {
          Planned_unit.unit = u;
          left_graph = (ex Left).descriptor;
          right_graph = (ex Right).descriptor;
          predecessors = Plan.Unit_order.predecessors unit_order u;
          canonical_reads = p.canonical_reads;
          canonical_writes = p.canonical_writes;
          input_bindings = p.input_bindings;
          input_checks = p.input_checks;
          output_bindings = p.output_bindings;
          output_pairs = p.output_pairs;
        })
      order
  in
  let consumers =
    List.map
      (fun (l : Plan.Unit_order.Lifetime.t) ->
        (l.canonical, Plan.Unit_order.consumers unit_order l.canonical))
      (Plan.Unit_order.lifetimes unit_order)
  in
  let unsealed : Execution_plan.t =
    {
      mapping_digest;
      frontier_mapping_digest;
      policy_selection_digest = bindings.selection_digest;
      policy_bindings_digest = bindings.bindings_digest;
      comparison_policies =
        List.map
          (fun (b : Policy_binding.t) -> (b.site.comparison, b.policy_digest))
          bindings.bindings;
      order = planned;
      final_values = Plan.Structural.final_values structural;
      consumers;
      coverage = Plan.Structural.coverage structural;
      plan_digest =
        (match Digest.Plan.make algorithm ~hex:(String.make 64 '0') with
        | Ok d -> d
        | Error _ -> invalid_arg "Sealed_plan.seal");
    }
  in
  let* plan_digest =
    digest Digest.Plan.make ~hashers ~algorithm
      (Execution_plan.preimage unsealed)
  in
  Ok { structural; extractions; bindings; plan = { unsealed with plan_digest } }

let verify_decoded t (decoded : Execution_plan.t) =
  if
    Digest.Plan.equal decoded.plan_digest t.plan.plan_digest
    && String.equal
         (Execution_plan.preimage decoded)
         (Execution_plan.preimage t.plan)
  then Ok ()
  else fail Stale_plan

let plan t = t.plan
let structural t = t.structural
let extractions t = t.extractions
let bindings t = t.bindings
