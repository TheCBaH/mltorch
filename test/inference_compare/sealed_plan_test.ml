open Inference_compare_schema
module Core = Inference_compare_core
module SP = Core.Sealed_plan
module K = Coord_fixture
module C = Chain_fixture
module P = Plan_fixtures

let hashers = K.hashers
let algorithm = K.algorithm

let residual =
  C.synthetic
    [
      C.node "n1" "add" [ "x"; "w" ] [ "h" ];
      C.node "n2" "mul" [ "h"; "w" ] [ "m" ];
      C.node "n3" "add" [ "m"; "x" ] [ "y" ];
    ]
    ~sources:[ "x"; "w" ] ~finals:[ "y" ]
    ~units:[ ("A", [ "n1" ]); ("B", [ "n2" ]); ("D", [ "n3" ]) ]

let seal (k : K.t) =
  SP.seal ~hashers ~algorithm k.structural k.extractions k.bindings

let digest_of = function
  | Ok p -> String.sub (Digest.Plan.hex (SP.plan p).plan_digest) 0 12
  | Error e -> "ERROR " ^ Format.asprintf "%a" SP.pp_error (Err.Error.kind e)

let kres = K.make residual

let%expect_test
    "sealing joins the validated plan, extractions and policy bindings" =
  let p = SP.plan kres.sealed in
  Format.printf "units in order: %s@."
    (String.concat " -> "
       (List.map (fun (u : Planned_unit.t) -> Id.Unit.to_string u.unit) p.order));
  Format.printf "predecessors: %s@."
    (String.concat "; "
       (List.map
          (fun (u : Planned_unit.t) ->
            Id.Unit.to_string u.unit ^ "<-"
            ^ String.concat "," (List.map Id.Unit.to_string u.predecessors))
          p.order));
  Format.printf "finals: %s; coverage: %s@."
    (String.concat "," (List.map Id.Canonical_value.to_string p.final_values))
    (match p.coverage with Complete -> "complete" | Partial _ -> "partial");
  Format.printf "consumers of w: %s@."
    (String.concat ","
       (List.map Id.Unit.to_string (List.assoc (P.ckey "w") p.consumers)));
  Format.printf "policies bound: %d@." (List.length p.comparison_policies);
  Format.printf "plan binds the sealed bindings: %b@."
    (Digest.Policy_bindings.equal p.policy_bindings_digest
       kres.bindings.bindings_digest);
  [%expect
    {|
    units in order: A -> B -> D
    predecessors: A<-; B<-A; D<-B
    finals: y; coverage: complete
    consumers of w: A,B
    policies bound: 3
    plan binds the sealed bindings: true |}]

let%expect_test "identity is stable and tracks every input" =
  let d0 = digest_of (seal kres) in
  Format.printf "repeat: %b@." (d0 = digest_of (seal kres));
  (* a policy change reseals the bindings and changes the plan *)
  let looser =
    K.make
      ~policy:
        {
          K.exact_policy with
          statistics = { direction = Left_minus_right; variance = Sample };
        }
      residual
  in
  Format.printf "different policy: %b@." (d0 <> digest_of (seal looser));
  let other =
    K.make
      (C.synthetic
         [
           C.node "n1" "add" [ "x"; "w" ] [ "h" ];
           C.node "n2" "mul" [ "h"; "w" ] [ "m" ];
           C.node "n3" "sub" [ "m"; "x" ] [ "y" ];
         ]
         ~sources:[ "x"; "w" ] ~finals:[ "y" ]
         ~units:[ ("A", [ "n1" ]); ("B", [ "n2" ]); ("D", [ "n3" ]) ])
  in
  Format.printf "different graph: %b@." (d0 <> digest_of (seal other));
  let regrouped =
    K.make
      (C.synthetic
         [
           C.node "n1" "add" [ "x"; "w" ] [ "h" ];
           C.node "n2" "mul" [ "h"; "w" ] [ "m" ];
           C.node "n3" "add" [ "m"; "x" ] [ "y" ];
         ]
         ~sources:[ "x"; "w" ] ~finals:[ "y" ]
         ~units:[ ("A", [ "n1"; "n2" ]); ("D", [ "n3" ]) ])
  in
  Format.printf "different partition: %b@." (d0 <> digest_of (seal regrouped));
  [%expect
    {|
    repeat: true
    different policy: true
    different graph: true
    different partition: true |}]

let%expect_test "bindings and extractions must describe this plan" =
  let sealed_with ?extractions ?bindings () =
    SP.seal ~hashers ~algorithm kres.structural
      (Option.value extractions ~default:kres.extractions)
      (Option.value bindings ~default:kres.bindings)
  in
  let show name r =
    Format.printf "%s: %s@." name
      (match r with Ok _ -> "accepted" | Error _ -> digest_of r)
  in
  show "baseline" (sealed_with ());
  show "an extraction missing"
    (sealed_with ~extractions:(List.tl kres.extractions) ());
  show "extractions out of order"
    (sealed_with ~extractions:(List.rev kres.extractions) ());
  show "an extraction repeated"
    (sealed_with ~extractions:(List.hd kres.extractions :: kres.extractions) ());
  (* bindings sealed against a different (regrouped) plan *)
  let regrouped =
    K.make
      (C.synthetic
         [
           C.node "n1" "add" [ "x"; "w" ] [ "h" ];
           C.node "n2" "mul" [ "h"; "w" ] [ "m" ];
           C.node "n3" "add" [ "m"; "x" ] [ "y" ];
         ]
         ~sources:[ "x"; "w" ] ~finals:[ "y" ]
         ~units:[ ("A", [ "n1"; "n2" ]); ("D", [ "n3" ]) ])
  in
  show "bindings from another plan"
    (sealed_with ~bindings:regrouped.bindings ());
  show "a binding missing"
    (sealed_with
       ~bindings:
         { kres.bindings with bindings = List.tl kres.bindings.bindings }
       ());
  let extra = List.hd regrouped.bindings.bindings in
  show "a binding for an unknown comparison"
    (sealed_with
       ~bindings:
         {
           kres.bindings with
           bindings =
             kres.bindings.bindings
             @ [
                 {
                   extra with
                   site =
                     { extra.site with comparison = P.comparison "cmp_ghost" };
                 };
               ];
         }
       ());
  (* a binding whose anchors differ is stale even with the right comparison id *)
  let first = List.hd kres.bindings.bindings in
  let moved = { first with site = { first.site with left_anchors = [] } } in
  show "a binding with other anchors"
    (sealed_with
       ~bindings:
         {
           kres.bindings with
           bindings = moved :: List.tl kres.bindings.bindings;
         }
       ());
  [%expect
    {|
    baseline: accepted
    an extraction missing: ERROR extractions do not match the plan
    extractions out of order: ERROR extractions do not match the plan
    an extraction repeated: ERROR extractions do not match the plan
    bindings from another plan: ERROR no policy binding for cmp_h
    a binding missing: ERROR no policy binding for cmp_h
    a binding for an unknown comparison: ERROR binding for unknown comparison cmp_ghost
    a binding with other anchors: ERROR policy binding for cmp_h is stale |}]

let%expect_test "a decoded plan is accepted only if it recomputes identically" =
  let p = SP.plan kres.sealed in
  let text = Wire_text.to_string (Execution_plan.encode p) in
  let decoded = Wire_text.decode_string Execution_plan.decode text in
  let show name d =
    match d with
    | Error e ->
        Format.printf "%s: decode error %a@." name
          (Err.Error.pp_kind Wire.pp_error)
          e
    | Ok d -> (
        match SP.verify_decoded kres.sealed d with
        | Ok () -> Format.printf "%s: accepted@." name
        | Error e ->
            Format.printf "%s: %a@." name SP.pp_error (Err.Error.kind e))
  in
  show "roundtrip" decoded;
  (match decoded with
  | Ok d ->
      show "dropped unit" (Ok { d with order = List.tl d.order });
      show "reordered units" (Ok { d with order = List.rev d.order });
      show "changed coverage"
        (Ok
           {
             d with
             coverage = Partial [ P.text (Prose.Description.of_string "x") ];
           });
      show "tampered policy digest"
        (Ok
           {
             d with
             policy_bindings_digest =
               Result.get_ok
                 (Digest.Policy_bindings.make algorithm
                    ~hex:(String.make 64 'a'));
           });
      show "tampered plan digest"
        (Ok
           {
             d with
             plan_digest =
               Result.get_ok
                 (Digest.Plan.make algorithm ~hex:(String.make 64 'b'));
           })
  | Error _ -> ());
  [%expect
    {|
    roundtrip: accepted
    dropped unit: decoded plan differs from the recomputed plan
    reordered units: decoded plan differs from the recomputed plan
    changed coverage: decoded plan differs from the recomputed plan
    tampered policy digest: decoded plan differs from the recomputed plan
    tampered plan digest: decoded plan differs from the recomputed plan |}]
