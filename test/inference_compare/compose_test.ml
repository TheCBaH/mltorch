open Inference_compare_schema
open Compose_cases

let%expect_test "each supplied map is itself valid" =
  List.iter
    (fun m ->
      let s = valid m (art m) in
      Printf.printf "%d units\n"
        (List.length (Plan.Unit_order.order (Plan.Structural.order s))))
    [ map_xy; map_yz; map_zw ];
  [%expect {|
    2 units
    2 units
    2 units |}]

let%expect_test "renaming and reshaping compose into a valid map" =
  let m = compose_ok (art map_xy) (art map_yz) in
  show m;
  ignore (accepted gx gz m);
  [%expect
    {|
    unit [n1] -> [n1_z] in [w_z,x_z] out [h_z]
    unit [n2] -> [n2_z] in [h_z] out [y_z]
    relation h <-> h_z routes 2
    relation w <-> w_z routes 2
    relation x <-> x_z routes 2
    relation y <-> y_z routes 2 |}]

let%expect_test "conversion steps keep their order and are not collapsed" =
  let m = compose_ok (art map_xy) (art map_yz) in
  List.iter
    (fun (r : Tensor_relation.t) ->
      if List.exists (fun t -> tlocal t = "x") r.left_members then
        List.iter print_endline (steps_text r))
    m.tensor_relations;
  [%expect
    {|
    x->x_z:{"shape":["3","1"],"tag":"reshape"};{"shape":["3"],"tag":"reshape"}
    x_z->x:{"shape":["3","1"],"tag":"reshape"};{"shape":["3"],"tag":"reshape"} |}]

let%expect_test "a fused map composed with a split map is one executable unit" =
  let m = compose_ok (art map_ab) (art map_bc) in
  show m;
  List.iter
    (fun donor -> summary (run_composed ~donor ga gc m))
    [ Side.Left; Side.Right ];
  [%expect
    {|
    unit [n1,n2] -> [g1,g2] in [x_z] out [y_z]
    relation x <-> x_z routes 2
    relation y <-> y_z routes 2
    complete agreement c-84c04582b0f2fd14 | cmp_y=pass
    complete agreement c-84c04582b0f2fd14 | cmp_y=pass |}]

let%expect_test "deleted and created nodes keep an explicit disposition" =
  ignore (accepted dx dy (art map_dxy));
  ignore (accepted dy dz (art map_dyz));
  let m = compose_ok (art map_dxy) (art map_dyz) in
  show m;
  List.iter print_endline
    (List.sort compare (List.map disposition_text m.dispositions));
  ignore (accepted dx dz m);
  [%expect
    {|
    unit [] -> [nc_z] in [x_z] out []
    unit [n1] -> [n1_z] in [x_z] out [h_z]
    unit [nd] -> [] in [] out []
    relation h <-> h_z routes 2
    relation x <-> x_z routes 2
    created nc_z
    eliminated nd |}]

let%expect_test "a folded constant is carried with its pinned payload" =
  ignore (accepted fx fy (art map_fxy));
  ignore (accepted fy fz (art map_fyz));
  let m = compose_ok (art map_fxy) (art map_fyz) in
  show m;
  List.iter
    (fun (i : Initial_binding.t) ->
      print_endline ("initial " ^ Id.Canonical_value.to_string i.key))
    m.frontier.initial_values;
  ignore (accepted fx fz m);
  [%expect
    {|
    unit [k1,n] -> [n_z] in [wf_z,x_z] out [y_z]
    relation x <-> x_z routes 2
    relation y <-> y_z routes 2
    initial x
    initial w
    initial k_wf_z |}]

(* A route that packs several source tensors into one is not composed: the
   direction it would give is refused rather than guessed from the single-source
   routes. *)
let%expect_test "a pack route is refused, not composed" =
  let first = art map_xy in
  let packed =
    {
      first with
      Mapping_artifact.tensor_relations =
        List.map
          (fun (r : Tensor_relation.t) ->
            match r.left_members with
            | [ w ] when tlocal w = "w" ->
                let x =
                  List.find
                    (fun m -> tlocal m = "x")
                    (List.concat_map
                       (fun (r : Tensor_relation.t) -> r.left_members)
                       first.tensor_relations)
                in
                {
                  r with
                  routes =
                    List.map
                      (fun (v : Value_route.t) ->
                        if
                          Tensor_ref.equal v.destination
                            (List.hd r.right_members)
                        then { v with sources = [ w; x ] }
                        else v)
                      r.routes;
                }
            | _ -> r)
          first.tensor_relations;
    }
  in
  (match F.compose packed (art map_yz) with
  | Ok _ -> print_endline "composed"
  | Error e -> print_endline (err_text e));
  [%expect
    {| no route for synthetic/v1@sha256:3333333333333333333333333333333333333333333333333333333333333333:/w_z toward the left side |}]
