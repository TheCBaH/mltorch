open Inference_compare_schema
open Compose_cases

let%expect_test "composing with an identity map changes nothing" =
  let i = art map_xy in
  let right_id = art (identity_map gy) and left_id = art (identity_map gx) in
  ignore (accepted gy gy right_id);
  same_meaning "I . id_Y" i (compose_ok i right_id);
  same_meaning "id_X . I" i (compose_ok left_id i);
  [%expect
    {|
    I . id_Y: same meaning (27 facts)
    id_X . I: same meaning (27 facts) |}]

let%expect_test "regrouping a chain of maps preserves its meaning" =
  let i = art map_xy and s = art map_yz and p = art map_zw in
  let left = compose_ok (compose_ok i s) p in
  let right = compose_ok i (compose_ok s p) in
  ignore (accepted gx gw left);
  ignore (accepted gx gw right);
  same_meaning "(I.S).P vs I.(S.P)" left right;
  let i = art map_ab and s = art map_bc and p = art map_cd in
  let left = compose_ok (compose_ok i s) p in
  ignore (accepted ga gd left);
  same_meaning "fusion chain (I.S).P vs I.(S.P)" left
    (compose_ok i (compose_ok s p));
  [%expect
    {|
    (I.S).P vs I.(S.P): same meaning (27 facts)
    fusion chain (I.S).P vs I.(S.P): same meaning (12 facts) |}]

let%expect_test "maps that do not meet are refused, never repaired" =
  let i = art map_xy and s = art map_yz in
  (* not the same intermediate graph *)
  refused i (art map_zw);
  (* the second map leaves an intermediate node out *)
  refused i { s with units = [ List.hd s.units ] };
  refused { i with units = [ List.hd i.units ] } s;
  (* the final outputs do not line up *)
  refused i
    {
      s with
      frontier =
        {
          s.frontier with
          final_outputs =
            List.map
              (fun (f : Final_output.t) ->
                { f with left = P.tensor F.key_y "h_y" })
              s.frontier.final_outputs;
        };
    };
  (* no reverse route: the right side cannot be read back as the left value *)
  refused i
    {
      s with
      tensor_relations =
        List.map
          (fun (r : Tensor_relation.t) ->
            {
              r with
              routes =
                List.filter
                  (fun (v : Value_route.t) ->
                    List.exists (Tensor_ref.equal v.destination) r.right_members)
                  r.routes;
            })
          s.tensor_relations;
    };
  [%expect
    {|
    intermediate graph differs: synthetic/v1@sha256:2222222222222222222222222222222222222222222222222222222222222222 vs synthetic/v1@sha256:3333333333333333333333333333333333333333333333333333333333333333
    intermediate node synthetic/v1@sha256:2222222222222222222222222222222222222222222222222222222222222222:/n2_y is absent from the second map
    intermediate node synthetic/v1@sha256:2222222222222222222222222222222222222222222222222222222222222222:/n2_y is absent from the first map
    final output 0 does not line up
    no route for synthetic/v1@sha256:1111111111111111111111111111111111111111111111111111111111111111:/h toward the right side |}]

let%expect_test "two different routes between the same tensors need a choice" =
  let i = art map_xy and s = art map_yz in
  let extra =
    {
      Value_route.route_id = P.text (Id.Value_route.of_string "other");
      sources = [ P.tensor F.key_y "x_y" ];
      destination = P.tensor F.key_z "x_z";
      steps = [ Conversion_step.Reshape [ 3L; 1L; 1L ] ];
      claim = Identical;
      lossy = false;
    }
  in
  let s =
    {
      s with
      tensor_relations =
        List.map
          (fun (r : Tensor_relation.t) ->
            if List.exists (fun t -> tlocal t = "x_y") r.left_members then
              { r with routes = extra :: r.routes }
            else r)
          s.tensor_relations;
    }
  in
  refused i s;
  [%expect
    {| ambiguous route synthetic/v1@sha256:1111111111111111111111111111111111111111111111111111111111111111:/x -> synthetic/v1@sha256:3333333333333333333333333333333333333333333333333333333333333333:/x_z |}]

let%expect_test "a pair through the intermediate graph is compared in its terms"
    =
  let y = P.tensor F.key_x "y"
  and y_y = P.tensor F.key_y "y_y"
  and y_z = P.tensor F.key_z "y_z" in
  (* first: compare x's y (reshaped) with y_y as is; second: y_y as is with y_z reshaped *)
  let i =
    with_pair (art map_xy) ~unit:"u2" ~id:"p1" ~left:y ~right:y_y
      ~left_steps:[ Conversion_step.Reshape [ 3L; 1L ] ]
      ~right_steps:[] ~spec:(spec_of gy)
  and s =
    with_pair (art map_yz) ~unit:"u2" ~id:"p2" ~left:y_y ~right:y_z
      ~left_steps:[]
      ~right_steps:[ Conversion_step.Reshape [ 3L; 1L ] ]
      ~spec:(spec_of gy)
  in
  ignore (accepted gx gy i);
  ignore (accepted gy gz s);
  let m = compose_ok i s in
  ignore (accepted gx gz m);
  List.iter
    (fun (r : Tensor_relation.t) ->
      List.iter
        (fun (p : Member_pair.t) ->
          Printf.printf "pair %s ~ %s, left steps %d, right steps %d, spec %s\n"
            (tlocal p.left) (tlocal p.right)
            (List.length p.left_to_comparison)
            (List.length p.right_to_comparison)
            (String.concat ";"
               (List.map Int64.to_string p.comparison_spec.logical_shape)))
        r.member_pairs)
    m.tensor_relations;
  [%expect {| pair y ~ y_z, left steps 1, right steps 1, spec 3;1 |}]

let%expect_test
    "a pair that converts the shared graph's side cannot be composed" =
  let y = P.tensor F.key_x "y"
  and y_y = P.tensor F.key_y "y_y"
  and y_z = P.tensor F.key_z "y_z" in
  let i =
    with_pair (art map_xy) ~unit:"u2" ~id:"p1" ~left:y ~right:y_y ~left_steps:[]
      ~right_steps:[ Conversion_step.Reshape [ 3L ] ]
      ~spec:(spec_of gx)
  and s =
    with_pair (art map_yz) ~unit:"u2" ~id:"p2" ~left:y_y ~right:y_z
      ~left_steps:[]
      ~right_steps:[ Conversion_step.Reshape [ 3L; 1L ] ]
      ~spec:(spec_of gy)
  in
  refused i s;
  [%expect {| comparison of pair p1 cannot be composed |}]

let%expect_test "a composed route is only as strong as its weakest part" =
  let precision s = P.text (Id.Precision.of_string s) in
  let claims i s =
    let m =
      compose_ok (with_claim (art map_xy) i) (with_claim (art map_yz) s)
    in
    List.sort_uniq compare
      (List.concat_map
         (fun (r : Tensor_relation.t) ->
           List.map (fun (v : Value_route.t) -> claim_text v.claim) r.routes)
         m.tensor_relations)
  in
  List.iter
    (fun (i, s) ->
      Printf.printf "%s . %s = %s\n" (claim_text i) (claim_text s)
        (String.concat "," (claims i s)))
    [
      (Claim.Identical, Claim.Identical);
      (Identical, Equivalent);
      (Equivalent, Approximate [ precision "bf16" ]);
      (Approximate [ precision "f16" ], Approximate [ precision "bf16" ]);
      (Approximate [ precision "f16" ], Unverifiable);
    ];
  [%expect
    {|
    identical . identical = identical
    identical . equivalent = equivalent
    equivalent . approximate bf16 = approximate bf16
    approximate f16 . approximate bf16 = approximate bf16,f16
    approximate f16 . unverifiable = unverifiable |}]
