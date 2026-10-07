(* M10.3: pressure feedback to SSA output blocking — each legal group's
   pressure through the production pipeline of both targets, the choice, and
   every blocked kernel bitwise against the per-node reference. *)

open Graph_ir
module F = Native_test.Graph_fixtures
module M = Machine_model.Mir_model
module B = Machine_model.Mir_blocking
module T = Model_test

let run ~route ~blocking g =
  let b =
    Loop_ir.Loop_bundle.build ~config:Loop_ir.Loop_bundle_c.default_config g
    |> Err.or_raise ~pp_error:Loop_ir.Loop_bundle.pp_error
  in
  let constants =
    List.map
      (fun id -> (id, T.values ~salt:3 (T.sig_of g id)))
      b.Loop_ir.Loop_bundle.constants
  in
  let inputs =
    List.mapi
      (fun i id -> (id, T.values ~salt:(i + 1) (T.sig_of g id)))
      b.Loop_ir.Loop_bundle.inputs
  in
  let reference =
    Err.or_raise ~pp_error:Eval_direct.pp_error
      (Eval_direct.run g ~constants ~inputs)
  in
  match M.prepare ~route ~blocking ~pipeline:Ssa_backends.Pipeline.Exact b with
  | Error rs ->
      (Fmt.str "refused: %a" Fmt.(list ~sep:(any "; ") M.Refusal.pp) rs, [])
  | Ok m -> (
      let decisions = M.blocking m in
      match
        M.Context.create m ~constants:(fun id -> List.assoc_opt id constants)
      with
      | Error s -> (Fmt.str "%a" M.Stop.pp s, decisions)
      | Ok cx -> (
          match
            M.Context.run cx ~inputs:(fun id -> List.assoc_opt id inputs)
          with
          | Error s -> (Fmt.str "%a" M.Stop.pp s, decisions)
          | Ok outs ->
              ( (if
                   List.for_all2
                     (fun id t ->
                       T.bits t = T.bits (Tensor_id.Map.find id reference))
                     g.Graph.outputs outs
                 then "bitwise"
                 else "DIFFERS"),
                decisions )))

let routes = M.Route.[ Aarch64 M.Stage.Scanned; X86_64 M.Stage.Scanned ]

let kernels () =
  [
    ( "linear 8 -> 16, 3 rows",
      F.build "linear"
        Graph_builder.(
          let* x = input ~shape:(F.s 1 1 1 1 3 8) () in
          let* w = constant ~shape:(F.s 16 1 1 1 1 8) () in
          linear { Linear.Linear.in_features = Dim.extent 8 } ~x ~weight:w ())
    );
    ( "bmm 2x(5x7 . 7x12)",
      F.build "bmm"
        Graph_builder.(
          let* a = input ~shape:(F.s 1 1 1 2 5 7) () in
          let* b = input ~shape:(F.s 1 1 1 2 7 12) () in
          bmm a b) );
    ("chain", F.chain ());
  ]

let%expect_test "feedback chooses per target; every group bitwise" =
  List.iter
    (fun (name, g) ->
      Fmt.pr "%s:@." name;
      List.iter
        (fun route ->
          let verdict, decisions = run ~route ~blocking:B.Policy.Feedback g in
          Fmt.pr "  %s feedback: %s@." (M.Route.name route) verdict;
          List.iter
            (fun (node, d) ->
              if List.length d.B.Decision.candidates > 1 then
                Fmt.pr "    %a: @[%a@]@." Node_id.pp node B.Decision.pp d)
            decisions;
          List.iter
            (fun group ->
              Fmt.pr "  %s group %d: %s@." (M.Route.name route) group
                (fst (run ~route ~blocking:(B.Policy.Group group) g)))
            B.groups)
        routes)
    (kernels ());
  [%expect
    {|
    linear 8 -> 16, 3 rows:
      aarch64 scanned feedback: bitwise
        n0: chosen 8
            group 1: peak fpr 4, gpr 9; hot stores none, loads none; 0 helper calls; frame 16 bytes
            group 8: peak fpr 18, gpr 16; hot stores none, loads none; 0 helper calls; frame 64 bytes
            group 4: peak fpr 10, gpr 12; hot stores none, loads none; 0 helper calls; frame 32 bytes
            group 2: peak fpr 6, gpr 10; hot stores none, loads none; 0 helper calls; frame 16 bytes
      aarch64 scanned group 8: bitwise
      aarch64 scanned group 4: bitwise
      aarch64 scanned group 2: bitwise
      x86_64 scanned feedback: bitwise
        n0: chosen 1
            group 1: peak fpr 4, gpr 10; hot stores gpr 3, loads gpr 2; 0 helper calls; frame 56 bytes
            group 8: peak fpr 18, gpr 17; hot stores fpr 9, gpr 20, loads fpr 9, gpr 19; 0 helper calls; frame 152 bytes
            group 4: peak fpr 10, gpr 13; hot stores fpr 1, gpr 12, loads fpr 1, gpr 11; 0 helper calls; frame 72 bytes
            group 2: peak fpr 6, gpr 11; hot stores gpr 7, loads gpr 6; 0 helper calls; frame 72 bytes
      x86_64 scanned group 8: bitwise
      x86_64 scanned group 4: bitwise
      x86_64 scanned group 2: bitwise
    bmm 2x(5x7 . 7x12):
      aarch64 scanned feedback: bitwise
        n0: chosen 8
            group 1: peak fpr 4, gpr 11; hot stores none, loads none; 0 helper calls; frame 16 bytes
            group 8: peak fpr 18, gpr 19; hot stores none, loads none; 0 helper calls; frame 80 bytes
            group 4: peak fpr 10, gpr 14; hot stores none, loads none; 0 helper calls; frame 48 bytes
            group 2: peak fpr 6, gpr 12; hot stores none, loads none; 0 helper calls; frame 32 bytes
      aarch64 scanned group 8: bitwise
      aarch64 scanned group 4: bitwise
      aarch64 scanned group 2: bitwise
      x86_64 scanned feedback: bitwise
        n0: chosen 1
            group 1: peak fpr 4, gpr 12; hot stores gpr 8, loads gpr 7; 0 helper calls; frame 72 bytes
            group 8: peak fpr 18, gpr 20; hot stores fpr 9, gpr 47, loads fpr 9, gpr 35; 0 helper calls; frame 152 bytes
            group 4: peak fpr 10, gpr 15; hot stores fpr 1, gpr 24, loads fpr 1, gpr 19; 0 helper calls; frame 72 bytes
            group 2: peak fpr 6, gpr 13; hot stores gpr 16, loads gpr 13; 0 helper calls; frame 72 bytes
      x86_64 scanned group 8: bitwise
      x86_64 scanned group 4: bitwise
      x86_64 scanned group 2: bitwise
    chain:
      aarch64 scanned feedback: bitwise
      aarch64 scanned group 8: bitwise
      aarch64 scanned group 4: bitwise
      aarch64 scanned group 2: bitwise
      x86_64 scanned feedback: bitwise
      x86_64 scanned group 8: bitwise
      x86_64 scanned group 4: bitwise
      x86_64 scanned group 2: bitwise |}]

let%expect_test "the choice: least hot spilling, then the largest group" =
  let module P = Machine_alloc.Mir_pressure in
  let c group spills =
    {
      B.Candidate.group;
      pressure =
        (match spills with
        | None -> Error "not lowered"
        | Some n ->
            Ok
              {
                P.peak = [];
                hot_stores =
                  (if n > 0 then [ (Machine_ir.Mir_target.Bank.Fpr, n) ] else []);
                hot_loads = [];
                stats = Machine_alloc.Mir_alloc_stats.empty;
                frame = None;
                helper_calls = 0;
              });
    }
  in
  List.iter
    (fun cs -> Fmt.pr "%d@." (B.decide cs).B.Decision.chosen)
    [
      [ c 1 (Some 0); c 8 (Some 5); c 4 (Some 0); c 2 (Some 0) ];
      [ c 1 (Some 3); c 8 (Some 9); c 4 (Some 4); c 2 (Some 3) ];
      [ c 1 (Some 2); c 8 None; c 4 (Some 2) ];
      [ c 1 None; c 2 None ];
    ];
  [%expect {|
    4
    2
    4
    1 |}]
