(* The attention layer has no Native4D counterpart: the domain check refuses
   the node with the typed row, and lowering it is the same refusal. One
   sequence, so every tensor is four-axis and the node is what is refused. *)

module F = Native_test.Attention_layer_fixtures

let%expect_test "Native4D refuses an attention layer" =
  let g = F.node { (F.all_flags F.small) with F.Config.batch = 1 } in
  let view =
    Graph_view.of_graph g |> Err.or_raise ~pp_error:Graph_view.pp_error
  in
  (match Err.payload (Native4d.Domain.check view) with
  | Ok () -> Fmt.pr "domain: in the dialect@."
  | Error e -> Fmt.pr "domain: %a@." Native4d.Error.pp e);
  Lower_fixtures.outcome "lowered" g;
  [%expect
    {|
    domain: node n0: no legalization for
              attention_layer
                x=t0
                norm=none
                q=(t1, t2)
                k=(t3, t4)
                v=(t5, t6)
                o=(t7, t8)
                layer_scale=t9
                rope=none
                mask=none
                params={heads=2; scale=default}
    lowered                    node n0: no legalization for
      attention_layer
        x=t0
        norm=none
        q=(t1, t2)
        k=(t3, t4)
        v=(t5, t6)
        o=(t7, t8)
        layer_scale=t9
        rope=none
        mask=none
        params={heads=2; scale=default} |}]
