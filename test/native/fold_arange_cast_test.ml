(* [Fold_arange_cast]: an int [Arange] immediately cast to float becomes one
   f32 [Arange]. See lib/native/transform/passes/fold_arange_cast.ml.

   The point of this file is the CLAIM, not the rewrite: the pass asserts the
   fused node's value is [Identical] to the two-node chain it replaces, and
   this fixture is small enough for the verifier to prove that exhaustively
   rather than decline it as too large. *)

let build name m =
  Graph_builder.build ~name ~outputs:(fun o -> [ o ]) m
  |> Err.or_raise ~pp_error:(fun ppf e ->
      Fmt.pf ppf "fixture %s: %a" name Graph_builder.pp_error e)

let arange_params : Factory.Arange.params =
  { start = 0.; stop = 4.; step = 1.; fmt = Payload.Fmt Payload.I64 }

(* The pass's target shape: [Arange]'s sole consumer is a float [To_copy]. *)
let int_arange_then_cast () =
  build "int_arange_then_cast"
    (let open Graph_builder in
     let* a = arange arange_params in
     let* f = to_copy Pointwise.To_copy.Float a in
     relu f)

(* A second, real consumer of the raw int [Arange] output -- the pass must
   decline rather than fold away a value something else still reads. *)
let live_arange_and_cast () =
  build "live_arange_and_cast"
    (let open Graph_builder in
     let* a = arange arange_params in
     let* f = to_copy Pointwise.To_copy.Float a in
     add f a)

let%expect_test "fold_arange_cast: the fold, and its map" =
  let g = int_arange_then_cast () in
  (match Rewrite.origin g with
  | Error e -> Format.printf "origin: %a@." Rewrite.pp_error (Err.Error.kind e)
  | Ok (Rewrite.Origin state) -> (
      match Pass.run_all state [ Fold_arange_cast.pass ] with
      | Error e -> Format.printf "%a@." Pass.pp_error (Err.Error.kind e)
      | Ok (Rewrite.Step (final, map)) ->
          Format.printf "@[<v 2>after:@,%a@]@." Graph_ir.pp
            (Rewrite.graph final);
          Format.printf "@[<v 2>map:@,%a@]@." Graph_map.pp map));
  [%expect
    {|
    after:
      graph
      inputs: []
      nodes:
        n3: [t1 f32 [C=4] ->[n2]] =
          arange params={start=0; stop=4; step=1; fmt=f32}
        n2: [t2 f32 [C=4]] = relu x=t1 <-n3
      outputs: [t2 f32 [C=4] <-n2]
    map:
      values:
        {t0} -> {} identical
      nodes:
        {n0, n1} -> {n3}
      provenance:
        none |}]

let fractional_arange_params : Factory.Arange.params =
  { start = 0.25; stop = 3.; step = 0.5; fmt = Payload.Fmt Payload.I64 }

let%expect_test "fold_arange_cast: a fractional step is left unfolded" =
  let g =
    build "fractional_int_arange_then_cast"
      (let open Graph_builder in
       let* a = arange fractional_arange_params in
       let* f = to_copy Pointwise.To_copy.Float a in
       relu f)
  in
  (match Rewrite.origin g with
  | Error e -> Format.printf "origin: %a@." Rewrite.pp_error (Err.Error.kind e)
  | Ok (Rewrite.Origin state) -> (
      match Pass.run_all state [ Fold_arange_cast.pass ] with
      | Error e -> Format.printf "%a@." Pass.pp_error (Err.Error.kind e)
      | Ok (Rewrite.Step (_, map)) ->
          Format.printf "changed: %b@."
            (not
               (Correspondence.is_empty (Graph_map.values map)
               && Node_map.is_empty (Graph_map.nodes map)))));
  [%expect {| changed: false |}]

(* Justifies the guard above: had the fold fired on this fixture, it would
   have been WRONG. The two-node chain's actual Direct value goes through a
   real float->int64->float round trip (`eval_direct.ml`'s dtype-specific
   [Arange] materialization truncates toward zero via [Int64.of_float]); a
   naive fused F32 [Arange] -- what the pass would emit without the guard --
   computes the raw fractional formula with no such round trip. Comparing the
   two directly (bypassing the pass, which correctly never fires here) proves
   they are NOT [Identical] for this fixture: exactly the case the guard
   exists to keep the pass from silently misclaiming. *)
let%expect_test
    "fold_arange_cast: an unguarded fusion would have disagreed with the \
     two-node chain on a fractional step" =
  let two_node =
    build "fractional_two_node"
      (let open Graph_builder in
       let* a = arange fractional_arange_params in
       to_copy Pointwise.To_copy.Float a)
  in
  let naive_fused =
    build "fractional_naive_fused"
      (Graph_builder.arange
         {
           fractional_arange_params with
           Factory.Arange.fmt = Payload.Fmt Payload.F32;
         })
  in
  let result g =
    Eval_direct.run g ~inputs:[]
    |> Err.or_raise ~pp_error:Eval_direct.pp_error
    |> Tensor_id.Map.find (List.hd g.Graph_ir.Graph.outputs)
  in
  Format.printf "two-node (truncated through int64): %a@." Tensor.pp
    (result two_node);
  Format.printf "naive F32 fusion (exact fractional):  %a@." Tensor.pp
    (result naive_fused);
  [%expect
    {|
    two-node (truncated through int64): tensor f32 [C=6] {0, 0, 1, 1, 2, 2}
    naive F32 fusion (exact fractional):  tensor f32 [C=6] {0.25, 0.75, 1.25, 1.75, 2.25, 2.75} |}]

let%expect_test "fold_arange_cast: a live int consumer is left alone" =
  let g = live_arange_and_cast () in
  (match Rewrite.origin g with
  | Error e -> Format.printf "origin: %a@." Rewrite.pp_error (Err.Error.kind e)
  | Ok (Rewrite.Origin state) -> (
      match Pass.run_all state [ Fold_arange_cast.pass ] with
      | Error e -> Format.printf "%a@." Pass.pp_error (Err.Error.kind e)
      | Ok (Rewrite.Step (_, map)) ->
          Format.printf "changed: %b@."
            (not
               (Correspondence.is_empty (Graph_map.values map)
               && Node_map.is_empty (Graph_map.nodes map)))));
  [%expect {| changed: false |}]

(* THE CLAIM. [To_copy]'s [Float] pixel is [S.load x out] unchanged and
   [Arange]'s own pixel never reads [fmt], so the verifier compares two
   structurally equal terms. *)
let%expect_test "fold_arange_cast: the Identical claim is proved, not declined"
    =
  let g = int_arange_then_cast () in
  match Rewrite.origin g with
  | Error e -> Format.printf "origin: %a@." Rewrite.pp_error (Err.Error.kind e)
  | Ok (Rewrite.Origin state) ->
      (match
         Pass.run_reporting ~verify:Map_verify.Policy.Require_proved state
           [ Fold_arange_cast.pass ]
       with
      | Error e -> Format.printf "%a@." Pass.pp_error (Err.Error.kind e)
      | Ok { Pass.audits; _ } ->
          List.iter
            (fun ({ id; report } : Pass.Audit.t) ->
              Format.printf "%a: %s@." Pass.Exec_id.pp id
                (Map_verify.Report.summary report))
            audits.reports);
      [%expect
        {| fold_arange_cast#0: 3 clusters: 2 proved (structural), 1 vacuous |}]
