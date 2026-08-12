(* [Me_pt2]: components, not pairs (.ai/model_explorer_design.md).

   The property under test is DISJOINTNESS — no node id appears in two entries —
   which is what sync_navigation requires and what a pairwise encoding
   silently breaks. It is asserted over the output rather than argued from the
   algorithm, because the case that breaks it is a decomposition and a fold
   meeting at one node, which no single fixture "looks" like. *)

module P = Me_pt2
module NG = Pt2_native_graph

let limits = Me_limits.Limits.untrusted

let origin ?(path = NG.Graph_path.root) index =
  {
    NG.Node_origin.graph_path = path;
    index;
    target = "aten.op";
    name = None;
    metadata = Schema_runtime.String_map.empty;
  }

let origins l =
  List.fold_left
    (fun m (node, os) ->
      Graph_ir.Node_id.Map.add (Graph_ir.Node_id.of_int node) os m)
    Graph_ir.Node_id.Map.empty l

let show ?(source_nodes = []) label l =
  match P.of_origins ~limits ~source_nodes (origins l) with
  | Error e ->
      Format.printf "%s: %a@." label
        (fun fmt -> function
          | `Over_limit (f, n) -> Fmt.pf fmt "over limit %s = %d" f n)
        (Err.Error.kind e)
  | Ok t ->
      Printf.printf "%s:\n" label;
      List.iter
        (fun (e : Me_session.Mapping_entry.t) ->
          Printf.printf "  [%s] <-> [%s]\n"
            (String.concat " " e.Me_session.Mapping_entry.left)
            (String.concat " " e.Me_session.Mapping_entry.right))
        t.P.entries;
      if t.P.created <> [] then
        Printf.printf "  created %s\n" (String.concat " " t.P.created);
      if t.P.deleted <> [] then
        Printf.printf "  deleted %s\n" (String.concat " " t.P.deleted);
      (* The requirement, checked rather than assumed. *)
      let all =
        List.concat_map
          (fun (e : Me_session.Mapping_entry.t) ->
            e.Me_session.Mapping_entry.left @ e.Me_session.Mapping_entry.right)
          t.P.entries
      in
      Printf.printf "  disjoint: %b\n"
        (List.length all = List.length (List.sort_uniq compare all))

let%expect_test "one to one" =
  show "1:1" [ (0, [ origin 0 ]); (1, [ origin 1 ]) ];
  [%expect
    {|
    1:1:
      [root#0] <-> [n0]
      [root#1] <-> [n1]
      disjoint: true |}]

let%expect_test "one PT2 node lowered into several native nodes" =
  show "1:N" [ (0, [ origin 0 ]); (1, [ origin 0 ]); (2, [ origin 0 ]) ];
  [%expect {|
    1:N:
      [root#0] <-> [n0 n1 n2]
      disjoint: true |}]

let%expect_test "several PT2 nodes folded into one native node" =
  show "N:1" [ (0, [ origin 0; origin 1; origin 2 ]) ];
  [%expect
    {|
    N:1:
      [root#0 root#1 root#2] <-> [n0]
      disjoint: true |}]

let%expect_test "a decomposition and a fold meeting at one node" =
  (* The case components exist for, and the one a pairwise encoding gets wrong:
     PT2 #0 lowers to native n0 and n1, and native n1 also covers PT2 #1. The
     four ids are ONE component. Emitting (p0,n0), (p0,n1), (p1,n1) would put
     p0 in two entries and n1 in two, which sync_navigation forbids — and each
     pair, read alone, looks perfectly correct. *)
  show "chain" [ (0, [ origin 0 ]); (1, [ origin 0; origin 1 ]) ];
  [%expect
    {|
    chain:
      [root#0 root#1] <-> [n0 n1]
      disjoint: true |}]

let%expect_test "a longer chain closes into a single component" =
  show "long chain"
    [
      (0, [ origin 0 ]);
      (1, [ origin 0; origin 1 ]);
      (2, [ origin 1; origin 2 ]);
      (3, [ origin 2 ]);
      (4, [ origin 9 ]);
    ];
  [%expect
    {|
    long chain:
      [root#0 root#1 root#2] <-> [n0 n1 n2 n3]
      [root#9] <-> [n4]
      disjoint: true |}]

let%expect_test "created and deleted come off the same relation" =
  show ~source_nodes:[ "root#0"; "root#7" ] "created + deleted"
    [ (0, [ origin 0 ]); (1, []) ];
  [%expect
    {|
    created + deleted:
      [root#0] <-> [n0]
      created n1
      deleted root#7
      disjoint: true |}]

let%expect_test "nested-graph origins do not become navigation" =
  (* The lowerer is root-only, so a non-root [graph_path] names a node in a
     nested graph with no native counterpart. Pairing one would send the right
     pane somewhere unrelated, so such a native node is [created] instead. *)
  let nested = NG.Graph_path.child NG.Graph_path.root 3 in
  show "nested only" [ (0, [ origin ~path:nested 0 ]) ];
  show "root and nested" [ (0, [ origin 0; origin ~path:nested 1 ]) ];
  [%expect
    {|
    nested only:
      created n0
      disjoint: true
    root and nested:
      [root#0] <-> [n0]
      disjoint: true |}]

let%expect_test "the output does not depend on insertion order" =
  (* Determinism is what the session's "two loads produce identical JSON" claim
     rests on, and a union-find whose representative depended on insertion
     order would break it while every entry stayed correct. *)
  let a =
    [ (0, [ origin 0 ]); (1, [ origin 0; origin 1 ]); (2, [ origin 1 ]) ]
  in
  let b = List.rev a in
  let render l =
    match P.of_origins ~limits ~source_nodes:[] (origins l) with
    | Error _ -> "error"
    | Ok t ->
        String.concat ";"
          (List.map
             (fun (e : Me_session.Mapping_entry.t) ->
               String.concat "," e.Me_session.Mapping_entry.left
               ^ "|"
               ^ String.concat "," e.Me_session.Mapping_entry.right)
             t.P.entries)
  in
  Printf.printf "same: %b\n%s\n" (String.equal (render a) (render b)) (render a);
  [%expect {|
    same: true
    root#0,root#1|n0,n1,n2 |}]

let%expect_test "the member ceilings" =
  let tight =
    Err.or_raise ~pp_error:Me_limits.pp_error
      (Me_limits.Limits.create ~max_mapping_members_total:2 limits)
  in
  let l = [ (0, [ origin 0 ]); (1, [ origin 1 ]) ] in
  (match P.of_origins ~limits:tight ~source_nodes:[] (origins l) with
  | Ok _ -> print_endline "total: ok"
  | Error e -> (
      match Err.Error.kind e with
      | `Over_limit (f, n) -> Printf.printf "total: over limit %s = %d\n" f n));
  let tight_e =
    Err.or_raise ~pp_error:Me_limits.pp_error
      (Me_limits.Limits.create ~max_mapping_members_per_entry:2 limits)
  in
  let big = [ (0, [ origin 0; origin 1; origin 2 ]) ] in
  (match P.of_origins ~limits:tight_e ~source_nodes:[] (origins big) with
  | Ok _ -> print_endline "per entry: ok"
  | Error e -> (
      match Err.Error.kind e with
      | `Over_limit (f, n) ->
          Printf.printf "per entry: over limit %s = %d\n" f n));
  [%expect
    {|
    total: over limit mappingMembers = 4
    per entry: over limit mappingMembersPerEntry = 4 |}]
