open Inference_compare_schema
module Plan = Inference_compare_plan
module Pol = Inference_compare_policy
module M = Inference_compare_metrics
module P = Plan_fixtures
module C = Chain_fixture
module S = Policy_selector_test

let structural = C.structural C.attention

let policy ?(atol = 1e-3) ?(metrics = []) () : Comparison_policy.t =
  {
    element = Metric_fixtures.floating ~atol ();
    statistics = { direction = Right_minus_left; variance = Population };
    relative_scale = Right_scale;
    tensor_metrics = metrics;
  }

let metric ?(role = Tensor_metric.Diagnostic_only) id kind : Tensor_metric.t =
  {
    id = P.text (Id.Tensor_metric.of_string id);
    kind;
    partition = Whole_tensor;
    role;
  }

let rmse ?(bound = 0.1) () =
  metric "rmse" Root_mean_square_error ~role:(Required (At_most bound))

let cosine =
  metric "cos"
    (Cosine_similarity { zero_norm = Undefined })
    ~role:(Required (At_least 0.9))

let mae = metric "mae" Mean_absolute_error

let rule ?(priority = 0) ?(empty = Metric_rule.Reject) id selector action :
    Metric_rule.t =
  {
    id = P.text (Id.Metric_rule.of_string id);
    description = P.text (Prose.Description.of_string id);
    selector;
    priority = Rule_priority.of_int priority;
    action;
    empty_match = empty;
  }

let matmul = S.on Left (S.prod (S.np ~op:"matmul" ()))
let softmax = S.on Left (S.prod (S.np ~op:"softmax" ()))
let at_add = S.on Left (S.cons ~port:0 (S.np ~op:"add" ()))

let sel ?default ?(explicit = []) rules : Policy_selection.t =
  { default; explicit; rules }

let ctx ?limits s = S.ctx ?limits s

let resolve ?(s = structural) ?limits selection =
  let sites = S.sites s in
  Pol.Rule_resolve.resolve ~ctx:(ctx ?limits s)
    ~codecs:Inference_compare_codecs.Element_codec.Registry.builtin
    ~env:M.Tensor_metrics.empty_env
    ~placeholder_digest:
      (Pol.Seal.placeholder Inference_compare_sha256.algorithm)
    sites selection

let short_digest d = String.sub (Digest.Comparison_policy.hex d) 0 8

let show ?s ?limits name selection =
  match resolve ?s ?limits selection with
  | Error e ->
      Format.printf "%s: %a@." name Pol.Policy_error.pp (Err.Error.kind e)
  | Ok r ->
      Format.printf "%s:@." name;
      List.iter
        (fun (b : Policy_binding.t) ->
          Format.printf "  %a base=%s atol=%s metrics=[%s] matches=[%s]@."
            Id.Comparison.pp b.site.comparison
            (match b.base with
            | Default -> "default"
            | Explicit -> "explicit"
            | Rules rs ->
                "rules:"
                ^ String.concat "+" (List.map Id.Metric_rule.to_string rs))
            (match b.policy.element with
            | Some (Floating f) -> Printf.sprintf "%g" f.atol
            | _ -> "-")
            (String.concat ","
               (List.map
                  (fun (m : Tensor_metric.t) -> Id.Tensor_metric.to_string m.id)
                  b.policy.tensor_metrics))
            (String.concat ","
               (List.map
                  (fun (m : Policy_binding.Rule_match.t) ->
                    Id.Metric_rule.to_string m.rule
                    ^ if m.selected_as_base then "*" else "")
                  b.matches)))
        r.bindings;
      if r.empty_matches <> [] then
        Format.printf "  empty matches: %a@."
          Fmt.(list ~sep:comma Id.Metric_rule.pp)
          r.empty_matches

let p_default = policy ~atol:1e-3 ()
let p_hi = policy ~atol:1e-6 ()
let p_lo = policy ~atol:1e-2 ()

let%expect_test
    "base policy: default, priority and visible lower-priority matches" =
  show "default only" (sel ~default:p_default []);
  show "priority picks the matmul policy; the lower one stays visible"
    (sel ~default:p_default
       [
         rule ~priority:1 "lo" Any_site (Select_policy p_lo);
         rule ~priority:10 "hi" matmul (Select_policy p_hi);
       ]);
  show "rule order is irrelevant"
    (sel ~default:p_default
       [
         rule ~priority:10 "hi" matmul (Select_policy p_hi);
         rule ~priority:1 "lo" Any_site (Select_policy p_lo);
       ]);
  show "equal priority, identical policies keep both rule ids"
    (sel
       [
         rule "a" matmul (Select_policy p_hi);
         rule "b" Any_site (Select_policy p_hi);
       ]);
  show "equal priority, different policies conflict"
    (sel
       [
         rule "a" matmul (Select_policy p_hi);
         rule "b" Any_site (Select_policy p_lo);
       ]);
  show "no base and no default" (sel [ rule "a" matmul (Select_policy p_hi) ]);
  [%expect
    {|
    default only:
      cmp_a base=default atol=0.001 metrics=[] matches=[]
      cmp_m base=default atol=0.001 metrics=[] matches=[]
      cmp_mm base=default atol=0.001 metrics=[] matches=[]
      cmp_o base=default atol=0.001 metrics=[] matches=[]
      cmp_s base=default atol=0.001 metrics=[] matches=[]
      cmp_t base=default atol=0.001 metrics=[] matches=[]
    priority picks the matmul policy; the lower one stays visible:
      cmp_a base=rules:lo atol=0.01 metrics=[] matches=[lo*]
      cmp_m base=rules:hi atol=1e-06 metrics=[] matches=[hi*,lo]
      cmp_mm base=rules:lo atol=0.01 metrics=[] matches=[lo*]
      cmp_o base=rules:hi atol=1e-06 metrics=[] matches=[hi*,lo]
      cmp_s base=rules:lo atol=0.01 metrics=[] matches=[lo*]
      cmp_t base=rules:lo atol=0.01 metrics=[] matches=[lo*]
    rule order is irrelevant:
      cmp_a base=rules:lo atol=0.01 metrics=[] matches=[lo*]
      cmp_m base=rules:hi atol=1e-06 metrics=[] matches=[hi*,lo]
      cmp_mm base=rules:lo atol=0.01 metrics=[] matches=[lo*]
      cmp_o base=rules:hi atol=1e-06 metrics=[] matches=[hi*,lo]
      cmp_s base=rules:lo atol=0.01 metrics=[] matches=[lo*]
      cmp_t base=rules:lo atol=0.01 metrics=[] matches=[lo*]
    equal priority, identical policies keep both rule ids:
      cmp_a base=rules:b atol=1e-06 metrics=[] matches=[b*]
      cmp_m base=rules:a+b atol=1e-06 metrics=[] matches=[a*,b*]
      cmp_mm base=rules:b atol=1e-06 metrics=[] matches=[b*]
      cmp_o base=rules:a+b atol=1e-06 metrics=[] matches=[a*,b*]
      cmp_s base=rules:b atol=1e-06 metrics=[] matches=[b*]
      cmp_t base=rules:b atol=1e-06 metrics=[] matches=[b*]
    equal priority, different policies conflict: policy_conflict rule=b comparison=cmp_m unit=u_mm
    no base and no default: missing_policy comparison=cmp_a unit=u_add |}]

let%expect_test "explicit bindings are authoritative" =
  let cmp c = P.comparison c in
  show "explicit overrides the default"
    (sel ~default:p_default ~explicit:[ (cmp "cmp_m", p_hi) ] []);
  show "a rule naming the same policy is consistent"
    (sel ~default:p_default
       ~explicit:[ (cmp "cmp_m", p_hi) ]
       [ rule "same" matmul (Select_policy p_hi) ]);
  show "a rule naming another policy conflicts"
    (sel ~default:p_default
       ~explicit:[ (cmp "cmp_m", p_hi) ]
       [ rule "other" matmul (Select_policy p_lo) ]);
  show "explicit binding for no site"
    (sel ~default:p_default ~explicit:[ (cmp "cmp_nope", p_hi) ] []);
  show "explicit binding listed twice"
    (sel ~default:p_default
       ~explicit:[ (cmp "cmp_m", p_hi); (cmp "cmp_m", p_hi) ]
       []);
  [%expect
    {|
    explicit overrides the default:
      cmp_a base=default atol=0.001 metrics=[] matches=[]
      cmp_m base=explicit atol=1e-06 metrics=[] matches=[]
      cmp_mm base=default atol=0.001 metrics=[] matches=[]
      cmp_o base=default atol=0.001 metrics=[] matches=[]
      cmp_s base=default atol=0.001 metrics=[] matches=[]
      cmp_t base=default atol=0.001 metrics=[] matches=[]
    a rule naming the same policy is consistent:
      cmp_a base=default atol=0.001 metrics=[] matches=[]
      cmp_m base=explicit atol=1e-06 metrics=[] matches=[same]
      cmp_mm base=default atol=0.001 metrics=[] matches=[]
      cmp_o base=rules:same atol=1e-06 metrics=[] matches=[same*]
      cmp_s base=default atol=0.001 metrics=[] matches=[]
      cmp_t base=default atol=0.001 metrics=[] matches=[]
    a rule naming another policy conflicts: policy_conflict rule=other comparison=cmp_m unit=u_mm
    explicit binding for no site: unavailable_comparison_site comparison=cmp_nope
    explicit binding listed twice: policy_conflict comparison=cmp_m |}]

let%expect_test "additive metrics accumulate from every matching rule" =
  show "two rules add different required metrics to a mixed-use site"
    (sel ~default:p_default
       [
         rule "r1" matmul (Add_metrics [ rmse () ]);
         rule "r2" at_add (Add_metrics [ cosine ]);
       ]);
  show "priority does not gate additions"
    (sel ~default:p_default
       [
         rule ~priority:99 "big" Any_site (Select_policy p_hi);
         rule ~priority:0 "tiny" matmul (Add_metrics [ rmse () ]);
       ]);
  show "identical metric from two rules merges"
    (sel ~default:p_default
       [
         rule "r1" matmul (Add_metrics [ rmse () ]);
         rule "r2" at_add (Add_metrics [ rmse () ]);
       ]);
  show "the same metric id with another threshold conflicts"
    (sel ~default:p_default
       [
         rule "r1" matmul (Add_metrics [ rmse () ]);
         rule "r2" at_add (Add_metrics [ rmse ~bound:0.5 () ]);
       ]);
  show "an added metric conflicting with the base policy"
    (sel
       ~default:(policy ~metrics:[ rmse () ] ())
       [ rule "r1" matmul (Add_metrics [ rmse ~bound:0.5 () ]) ]);
  show "a diagnostic addition keeps the base checks"
    (sel ~default:p_default [ rule "r1" matmul (Add_metrics [ mae ]) ]);
  [%expect
    {|
    two rules add different required metrics to a mixed-use site:
      cmp_a base=default atol=0.001 metrics=[] matches=[]
      cmp_m base=default atol=0.001 metrics=[cos,rmse] matches=[r1,r2]
      cmp_mm base=default atol=0.001 metrics=[] matches=[]
      cmp_o base=default atol=0.001 metrics=[rmse] matches=[r1]
      cmp_s base=default atol=0.001 metrics=[] matches=[]
      cmp_t base=default atol=0.001 metrics=[] matches=[]
    priority does not gate additions:
      cmp_a base=rules:big atol=1e-06 metrics=[] matches=[big*]
      cmp_m base=rules:big atol=1e-06 metrics=[rmse] matches=[big*,tiny]
      cmp_mm base=rules:big atol=1e-06 metrics=[] matches=[big*]
      cmp_o base=rules:big atol=1e-06 metrics=[rmse] matches=[big*,tiny]
      cmp_s base=rules:big atol=1e-06 metrics=[] matches=[big*]
      cmp_t base=rules:big atol=1e-06 metrics=[] matches=[big*]
    identical metric from two rules merges:
      cmp_a base=default atol=0.001 metrics=[] matches=[]
      cmp_m base=default atol=0.001 metrics=[rmse] matches=[r1,r2]
      cmp_mm base=default atol=0.001 metrics=[] matches=[]
      cmp_o base=default atol=0.001 metrics=[rmse] matches=[r1]
      cmp_s base=default atol=0.001 metrics=[] matches=[]
      cmp_t base=default atol=0.001 metrics=[] matches=[]
    the same metric id with another threshold conflicts: policy_conflict rule=r2 comparison=cmp_m unit=u_mm metric=rmse
    an added metric conflicting with the base policy: policy_conflict rule=r1 comparison=cmp_m unit=u_mm metric=rmse
    a diagnostic addition keeps the base checks:
      cmp_a base=default atol=0.001 metrics=[] matches=[]
      cmp_m base=default atol=0.001 metrics=[mae] matches=[r1]
      cmp_mm base=default atol=0.001 metrics=[] matches=[]
      cmp_o base=default atol=0.001 metrics=[mae] matches=[r1]
      cmp_s base=default atol=0.001 metrics=[] matches=[]
      cmp_t base=default atol=0.001 metrics=[] matches=[] |}]

let%expect_test "empty matches and invalid actions" =
  show "empty match rejected by default"
    (sel ~default:p_default
       [
         rule "none" (Comparison (P.comparison "cmp_zz")) (Add_metrics [ mae ]);
       ]);
  show "selector over nothing, reject"
    (sel ~default:p_default
       [
         rule "none"
           (S.on Left (S.prod (S.np ~op:"conv" ())))
           (Add_metrics [ mae ]);
       ]);
  show "selector over nothing, allowed"
    (sel ~default:p_default
       [
         rule ~empty:Allow "none"
           (S.on Left (S.prod (S.np ~op:"conv" ())))
           (Add_metrics [ mae ]);
       ]);
  show "added cosine with the wrong threshold direction"
    (sel ~default:p_default
       [
         rule "bad" matmul
           (Add_metrics
              [
                metric "cos2"
                  (Cosine_similarity { zero_norm = Undefined })
                  ~role:(Required (At_most 0.5));
              ]);
       ]);
  show "unknown reference inside a selector"
    (sel ~default:p_default
       [ rule "ghost" (Unit (P.uid "u_ghost")) (Add_metrics [ mae ]) ]);
  show "witness budget"
    ~limits:{ P.limits with max_retained_witnesses = 1L }
    (sel ~default:p_default
       [
         rule "any" (S.on Left (Boundary Cluster_output)) (Add_metrics [ mae ]);
       ]);
  [%expect
    {|
    empty match rejected by default: invalid_selector rule=none
    selector over nothing, reject: empty_rule_match rule=none
    selector over nothing, allowed:
      cmp_a base=default atol=0.001 metrics=[] matches=[]
      cmp_m base=default atol=0.001 metrics=[] matches=[]
      cmp_mm base=default atol=0.001 metrics=[] matches=[]
      cmp_o base=default atol=0.001 metrics=[] matches=[]
      cmp_s base=default atol=0.001 metrics=[] matches=[]
      cmp_t base=default atol=0.001 metrics=[] matches=[]
      empty matches: none
    added cosine with the wrong threshold direction: unsupported_policy comparison=cmp_m unit=u_mm (threshold direction does not fit the metric)
    unknown reference inside a selector: invalid_selector rule=ghost
    witness budget: resource_exhausted rule=any limit=1 actual=6 |}]

(* ---- sealing ------------------------------------------------------------- *)

let seal ?(s = structural) selection =
  match resolve ~s selection with
  | Error e ->
      Error (Format.asprintf "%a" Pol.Policy_error.pp (Err.Error.kind e))
  | Ok r -> (
      match
        Pol.Seal.seal ~hashers:Inference_compare_sha256.registry
          ~algorithm:Inference_compare_sha256.algorithm ~selection r.bindings
      with
      | Ok b -> Ok b
      | Error e ->
          Error (Format.asprintf "%a" Pol.Seal.pp_error (Err.Error.kind e)))

let digest = function
  | Ok (b : Policy_bindings.t) ->
      String.sub (Digest.Policy_bindings.hex b.bindings_digest) 0 12
  | Error m -> "ERROR " ^ m

let%expect_test "sealed digests are stable and track every input" =
  let base =
    sel ~default:p_default
      [
        rule ~priority:10 "hi" matmul (Select_policy p_hi);
        rule "extra" at_add (Add_metrics [ rmse () ]);
      ]
  in
  let d0 = digest (seal base) in
  Format.printf "repeat planning identical: %b@." (d0 = digest (seal base));
  Format.printf "rule order identical: %b@."
    (d0 = digest (seal { base with rules = List.rev base.rules }));
  Format.printf "changed threshold differs: %b@."
    (d0
    <> digest
         (seal
            (sel ~default:p_default
               [
                 rule ~priority:10 "hi" matmul (Select_policy p_hi);
                 rule "extra" at_add (Add_metrics [ rmse ~bound:0.2 () ]);
               ])));
  Format.printf "changed priority differs: %b@."
    (d0
    <> digest
         (seal
            (sel ~default:p_default
               [
                 rule ~priority:11 "hi" matmul (Select_policy p_hi);
                 rule "extra" at_add (Add_metrics [ rmse () ]);
               ])));
  Format.printf "changed default differs: %b@."
    (d0 <> digest (seal { base with default = Some p_lo }));
  (* a graph change: the first matmul becomes a conv, so `matmul` and the
     consumer arrangement select different sites *)
  let changed =
    C.structural
      {
        C.attention with
        nodes =
          List.map
            (fun (n : C.node) ->
              if n.name = "n_mm" then { n with op = "conv" } else n)
            C.attention.nodes;
      }
  in
  Format.printf "changed graph differs: %b@."
    (d0 <> digest (seal ~s:changed base));
  (* a mapping change: one more declared member pair is one more site *)
  let with_pair =
    C.structural { C.attention with pairs = [ ("u_att", "o") ] }
  in
  Format.printf "changed mapping differs: %b@."
    (d0 <> digest (seal ~s:with_pair base));
  (* A rule that matches nothing changes no site policy, but it is part of the
     declared selection and so of the sealed identity. *)
  let silent =
    {
      base with
      rules =
        rule ~empty:Allow "silent"
          (S.on Left (S.prod (S.np ~op:"conv" ())))
          (Add_metrics [ mae ])
        :: base.rules;
    }
  in
  Format.printf "unmatched rule differs: %b@." (d0 <> digest (seal silent));
  (match seal base with
  | Ok b ->
      Format.printf
        "policy digests recorded: %b, selection digest recorded: %b@."
        (List.for_all
           (fun (x : Policy_binding.t) ->
             not
               (Digest.Comparison_policy.equal x.policy_digest
                  (Pol.Seal.placeholder Inference_compare_sha256.algorithm)))
           b.bindings)
        (String.length (Digest.Policy_selection.hex b.selection_digest) = 64)
  | Error m -> Format.printf "%s@." m);
  [%expect
    {|
    repeat planning identical: true
    rule order identical: true
    changed threshold differs: true
    changed priority differs: true
    changed default differs: true
    changed graph differs: true
    changed mapping differs: true
    unmatched rule differs: true
    policy digests recorded: true, selection digest recorded: true |}]

let%expect_test "both donor passes share one binding set" =
  (* Bindings are resolved from the selection and the plan alone: nothing about
     donors, input values or observed errors enters resolution. *)
  let base =
    sel ~default:p_default [ rule "r" matmul (Add_metrics [ rmse () ]) ]
  in
  let a = digest (seal base) and b = digest (seal base) in
  Format.printf "same binding digest for the reference and tested passes: %b@."
    (a = b);
  [%expect {| same binding digest for the reference and tested passes: true |}]

let%expect_test "donor overrides" =
  let ov unit output donor : Donor_override.t =
    { unit = P.uid unit; output = P.ckey output; donor }
  in
  let exp overrides : Experiment.t =
    {
      experiment_id = P.text (Id.Experiment.of_string "follow_left");
      default_donor = Left;
      overrides;
      compare_policy_digest =
        Pol.Seal.placeholder Inference_compare_sha256.algorithm;
    }
  in
  let show name overrides =
    match Pol.Donor_check.validate structural (exp overrides) with
    | Ok () -> Format.printf "%s: ok@." name
    | Error e ->
        Format.printf "%s: %a@." name Pol.Policy_error.pp (Err.Error.kind e)
  in
  show "none" [];
  show "a written, consumed output" [ ov "u_mm" "m" Right ];
  show "a final output" [ ov "u_tail" "t" Right ];
  show "wrong unit for the output" [ ov "u_add" "m" Right ];
  show "an initial value no unit writes" [ ov "u_mm" "x" Right ];
  show "unknown output" [ ov "u_mm" "zz" Right ];
  show "the same output twice" [ ov "u_mm" "m" Right; ov "u_mm" "m" Left ];
  [%expect
    {|
    none: ok
    a written, consumed output: ok
    a final output: ok
    wrong unit for the output: unknown_override unit=u_add
    an initial value no unit writes: unknown_override unit=u_mm
    unknown output: unknown_override unit=u_mm
    the same output twice: unknown_override unit=u_mm |}]
