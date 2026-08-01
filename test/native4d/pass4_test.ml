(* A Native4D rewrite through the SHARED framework — stage 7's acceptance
   criterion, and the thing that distinguishes "two dialects" from "one dialect
   and a copy of it".

   Nothing under test here is Native4D-specific except the op projector. The
   recipe, the planning, the id discipline, the claim propagation, the map and
   its verification are the modules Native's nine passes use. *)

open Native4d

let build ~outputs m =
  match Builder.build ~outputs m with
  | Ok g -> g
  | Error e ->
      invalid_arg (Format.asprintf "%a" Builder.pp_error e.Core.Error.kind)

let nhwc = Shape4.of_ints ~n:1 ~h:2 ~w:2 ~c:3

(* An identity Permute4 between two ops, so removing it is observable. *)
let identity_permute () =
  build
    ~outputs:(fun o -> [ o ])
    (let open Builder in
     let* x = input ~shape:nhwc () in
     let* p = permute4 Ops4.Permute4.identity x in
     relu p)

let run ?verify g =
  match Framework.Rewrite4.origin g with
  | Error e ->
      Format.printf "origin: %a@." Framework.Rewrite4.pp_error e.Core.Error.kind
  | Ok (Framework.Rewrite4.Origin state) -> (
      match
        Framework.Pass4.run_reporting ?verify state [ Trim_permute4.pass ]
      with
      | Error e ->
          Format.printf "%a@." Framework.Pass4.pp_error e.Core.Error.kind
      | Ok
          {
            Framework.Pass4.audits;
            step = Framework.Rewrite4.Step (final, map);
          } ->
          Format.printf "@[<v 2>after:@,%a@]@." Graph.pp
            (Framework.Rewrite4.graph final);
          Format.printf "@[<v 2>map:@,%a@]@." Graph_map.pp map;
          List.iter
            (fun { Framework.Pass4.Audit.pass; report } ->
              Format.printf "audit %s: %s@." pass
                (Map_verify.Report.summary report))
            audits)

let%expect_test "pass4: an identity permute is trimmed" =
  run (identity_permute ());
  [%expect
    {|
    after:
      graph4
      inputs: [t0 [H=2 W=2 C=3]]
      nodes:
        n1: [t2] = relu x=t0
      outputs: [t2 [H=2 W=2 C=3]]
    map:
      values:
        {t0, t1} -> {t0} identical
      nodes:
        {n0} -> {}
      provenance:
        none |}]

(* THE POINT. The pass is verified by the same [Map_verify] the Native passes
   are, through [Pass4]'s own instantiation — so a Native4D rewrite is not just
   expressible in the shared framework, it is held to the same bar. *)
let%expect_test "pass4: the rewrite verifies through the shared framework" =
  run ~verify:Map_verify.Policy.Require_proved (identity_permute ());
  [%expect
    {|
    after:
      graph4
      inputs: [t0 [H=2 W=2 C=3]]
      nodes:
        n1: [t2] = relu x=t0
      outputs: [t2 [H=2 W=2 C=3]]
    map:
      values:
        {t0, t1} -> {t0} identical
      nodes:
        {n0} -> {}
      provenance:
        none
    audit trim_permute4: 2 clusters: 2 proved (structural) |}]
