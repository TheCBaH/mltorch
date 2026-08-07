(* Dead-code elimination: which nodes go, what the map says about their edges,
   and the two properties that distinguish this implementation from the obvious
   wrong ones. See .ai/native_transform_design.md §12g. *)

open Graph_ir

let s1c n = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:n

let build name m =
  Graph_builder.build ~name ~outputs:(fun o -> [ o ]) m
  |> Core.or_raise (fun ppf e ->
      Fmt.pf ppf "fixture %s: %a" name Graph_builder.pp_error e)

let run ?(pass = Dce.pass) g =
  match Rewrite.origin g with
  | Error e ->
      Format.printf "origin: %a@." Rewrite.pp_error e.Core.Error.kind;
      None
  | Ok (Rewrite.Origin state) -> (
      match Pass.run_all state [ pass ] with
      | Error e ->
          Format.printf "%a@." Pass.pp_error e.Core.Error.kind;
          None
      | Ok (Rewrite.Step (final, map)) ->
          Format.printf "@[<v 2>after:@,%a@]@." Graph_ir.pp
            (Rewrite.graph final);
          Format.printf "@[<v 2>map:@,%a@]@." Graph_map.pp map;
          Some (Rewrite.graph final))

(* ---- what it removes ------------------------------------------------------ *)

(* One live chain and one dead branch off the same input. *)
let dead_branch () =
  build "dead_branch"
    (let open Graph_builder in
     let* x = input ~shape:(s1c 4) () in
     let* _dead = sqrt x in
     relu x)

let%expect_test "dce: a node nothing reads is removed, with a delete cluster" =
  ignore (run (dead_branch ()));
  [%expect
    {|
    after:
      graph
      inputs: [t0 f32 [C=4] ->[n1]]
      nodes:
        n1: [t2 f32 [C=4]] = relu x=t0
      outputs: [t2 f32 [C=4] <-n1]
    map:
      values:
        {t1} -> {} identical
      nodes:
        {n0} -> {}
      provenance:
        none |}]

(* A [Discard] sink is dead by construction: it produces nothing, so it is never
   reachable from an output. This is the pruning .ai/native_multi_output_design.md
   defers to a future pass. *)
let discarded_indices () =
  build "discarded_indices"
    (let open Graph_builder in
     let* x = input ~shape:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:4 ~w:4 ~c:3) () in
     let* values, indices =
       max_pool2d_with_indices
         {
           kernel = { h = Dim.extent 2; w = Dim.extent 2 };
           stride = { h = Op_config.Pos.of_int 2; w = Op_config.Pos.of_int 2 };
           pad =
             { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
         }
         x
     in
     let* () = discard indices in
     return values)

let%expect_test "dce: a Discard sink is removed, its op left live" =
  ignore (run (discarded_indices ()));
  [%expect
    {|
    after:
      graph
      inputs: [t0 f32 [H=4 W=4 C=3] ->[n0]]
      nodes:
        n0: [t1 f32 [H=2 W=2 C=3], t2 f32 [H=2 W=2 C=3]] =
          max_pool2d_with_indices
            x=t0
            params={kernel={h=2; w=2}; stride={h=2; w=2}; pad={h=0; w=0}}
      outputs: [t1 f32 [H=2 W=2 C=3] <-n0]
    map:
      values:
        identity
      nodes:
        {n1} -> {}
      provenance:
        none |}]

(* ---- the dead chain: why the predicate is global -------------------------- *)

(* [n] chained dead nodes hanging off the input. A LOCAL predicate ("every output
   unused") matches only the terminal one per sweep, so the chain peels one link
   at a time and a fixed point needs [n] iterations — at n=16 that exactly
   exhausts [Pass.fixpoint]'s default fuel. Reachability sees the whole chain at
   once, so one sweep suffices however long it is. *)
let dead_chain n =
  build "dead_chain"
    (let open Graph_builder in
     let* x = input ~shape:(s1c 4) () in
     let rec chain k acc =
       if k = 0 then return acc
       else
         let* next = sqrt acc in
         chain (k - 1) next
     in
     let* _dead = chain n x in
     relu x)

(* The local predicate, written out, so the regression below compares against
   the implementation this pass deliberately is not. *)
let local_dce =
  Pass.fixpoint
    (Pass.per_node ~name:"local_dce"
       {
         Pass.on_node =
           (fun { Pass.view; _ } (n : node) ->
             let unused id =
               (not (Graph_view.is_graph_output view id))
               && Graph_view.uses view id = []
             in
             if n.Node.outputs <> [] && List.for_all unused n.Node.outputs then
               Some (Recipe.replace ~remove:[ n.Node.id ] ~insert:[] ())
             else None);
       })

let node_count g = List.length g.Graph.nodes

(* Silent twin of [run]: reports the node count, or the pass error VERBATIM. The
   error text is the point — "it failed" would keep passing if the local
   predicate started failing for some unrelated reason, which is the whole
   distinction this test exists to draw. *)
let outcome label ~pass g =
  (* [Rewrite.error] is a subrow of [Pass.error], so both failures print through
     one printer without an intermediate error type. *)
  let result =
    match Rewrite.origin g with
    | Error e -> Error (e.Core.Error.kind :> Pass.error)
    | Ok (Rewrite.Origin state) -> (
        match Pass.run_all state [ pass ] with
        | Error e -> Error e.Core.Error.kind
        | Ok (Rewrite.Step (final, _)) -> Ok (Rewrite.graph final))
  in
  match result with
  | Ok g -> Format.printf "%s: %d nodes@." label (node_count g)
  | Error kind -> Format.printf "%s: %a@." label Pass.pp_error kind

(* THE REGRESSION. A 20-link dead chain: reachability removes it in one sweep;
   the local predicate under a fixed point exhausts its fuel first, and says so.
   Both halves are asserted, because a test pinning only the passing side would
   still pass if the implementation silently reverted to the local predicate. *)
let%expect_test "dce: a long dead chain goes in one sweep" =
  let g = dead_chain 20 in
  Format.printf "before: %d nodes@." (node_count g);
  outcome "reachability" ~pass:Dce.pass g;
  outcome "local predicate" ~pass:local_dce g;
  [%expect
    {|
    before: 21 nodes
    reachability: 1 nodes
    local predicate: pass local_dce did not converge |}]

(* ---- verification is not optional ----------------------------------------- *)

(* THE OTHER REGRESSION. [Pass.verified] is reached only through the private
   [Pass.of_sweep], and [Pass.run_with] does not wrap what a pass returns — so a
   hand-built [Pass.t] calling [Rewrite.apply] would skip verification silently
   even under [run_all ~verify]. This asserts the audit exists, which is the
   cheap standing guard on that whole class: any future pass that bypasses
   [of_sweep] fails it. *)
let%expect_test "dce: the pass verifies its own step" =
  let g = dead_branch () in
  match Rewrite.origin g with
  | Error e -> Format.printf "origin: %a@." Rewrite.pp_error e.Core.Error.kind
  | Ok (Rewrite.Origin state) ->
      (match
         Pass.run_reporting ~verify:Map_verify.Policy.Require_proved state
           [ Dce.pass ]
       with
      | Error e -> Format.printf "%a@." Pass.pp_error e.Core.Error.kind
      | Ok { Pass.audits; _ } ->
          List.iter
            (fun ({ id; report } : Pass.Audit.t) ->
              Format.printf "audit %a: %s@." Pass.Exec_id.pp id
                (Map_verify.Report.summary report))
            audits.reports;
          Format.printf "audits: %d@." (Pass.Audit_log.retained audits));
      [%expect
        {|
    audit dce#0: 3 clusters: 2 proved (structural), 1 vacuous
    audits: 1 |}]
