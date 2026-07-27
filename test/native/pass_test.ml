(* Exercises the pass driver: a sweep merging every match it accepts into one
   step, a fixed point composing its iterations into one mapping, a pipeline
   composing passes, and constant payloads accumulating across the whole run.
   See .ai/native_transform_design.md §5.

   The passes here are test-local on purpose — the real ones land in later
   stages, and the driver should be exercised by something small enough to read
   in one screen. *)

open Graph_ir

let t_ n = Tensor_id.of_int n

let is_identity_perm perm =
  List.for_all (fun (out, inp) -> Axis.equal out inp) perm

(* Drop an identity permute, tying its output to its input. *)
let trim_identity =
  Pass.per_node ~name:"trim_identity"
    {
      Pass.on_node =
        (fun _env (n : node) ->
          match (n.Node.op, n.Node.outputs) with
          | Permute { perm; x }, [ out ] when is_identity_perm perm ->
              Some (Recipe.trim ~remove:[ n.Node.id ] ~tie:[ (out, x) ])
          | _ -> None);
    }

(* The same, but only for a permute reading a graph input — so each sweep can
   remove exactly one link of a chain, which is what makes a fixed point
   observably different from a single sweep. *)
let trim_identity_at_input =
  Pass.of_pattern ~name:"trim_identity_at_input"
    ~pattern:(fun anchor ->
      Pattern.(
        let* p, n =
          def anchor (function
            | Permute (p : Permute.Permute.t) when is_identity_perm p.perm ->
                Some p
            | _ -> None)
        in
        let* v_def = optional (peek p.x (fun _ -> Some ())) in
        let+ () = guard (v_def = None) in
        (n.Node.id, anchor, p.x)))
    ~build:(fun (node, out, x) _region ->
      Recipe.trim ~remove:[ node ] ~tie:[ (out, x) ])

let run_pipeline ?(show = true) g passes =
  match Rewrite.origin g with
  | Error e -> Format.printf "origin: %a@." Rewrite.pp_error e.Core.Error.kind
  | Ok (Rewrite.Origin state) -> (
      match Pass.run_all state passes with
      | Error e -> Format.printf "%a@." Pass.pp_error e.Core.Error.kind
      | Ok (Rewrite.Step (final, map)) ->
          if show then
            Format.printf "@[<v 2>after:@,%a@]@." Graph_ir.pp
              (Rewrite.graph final);
          Format.printf "@[<v 2>map:@,%a@]@." Graph_map.pp map)

(* ---- sweeps -------------------------------------------------------------- *)

let%expect_test "one sweep merges every match into a single step" =
  (* All three permutes are identity, so per_node offers three builders; they
     are planned in sequence, merged, and applied once — one step, one map. *)
  let g = Graph_fixtures.permute_identity_chain () in
  Format.printf "@[<v 2>before:@,%a@]@." Graph_ir.pp g;
  run_pipeline g [ trim_identity ];
  [%expect
    {|
    before:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n0]]
      nodes:
        n0: [t1 f32 [H=2 W=3 C=4] ->[n1]] = permute x=t0 perm=[]
        n1: [t2 f32 [H=2 W=3 C=4] ->[n2]] = permute x=t1 <-n0 perm=[]
        n2: [t3 f32 [H=2 W=3 C=4] ->[n3]] = permute x=t2 <-n1 perm=[]
        n3: [t4 f32 [H=2 W=3 C=4]] = relu x=t3 <-n2
      outputs: [t4 f32 [H=2 W=3 C=4] <-n3]
    after:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n3]]
      nodes:
        n3: [t4 f32 [H=2 W=3 C=4]] = relu x=t0
      outputs: [t4 f32 [H=2 W=3 C=4] <-n3]
    map:
      values:
        {t0, t1, t2, t3} -> {t0} identical
      nodes:
        {n0} -> {}
        {n1} -> {}
        {n2} -> {}
      provenance:
        none |}]

let%expect_test "a pass that matches nothing is the identity step" =
  run_pipeline (Graph_fixtures.diamond ()) [ trim_identity ];
  [%expect
    {|
    after:
      graph
      inputs: [t0 f32 [C=4] ->[n0, n1]]
      nodes:
        n0: [t1 f32 [C=4] ->[n2]] = relu x=t0
        n1: [t2 f32 [C=4] ->[n2]] = mul a=t0 b=t0
        n2: [t3 f32 [C=4]] = add a=t1 <-n0 b=t2 <-n1
      outputs: [t3 f32 [C=4] <-n2]
    map:
      values:
        identity
      nodes:
        identity
      provenance:
        none |}]

(* ---- fixed point --------------------------------------------------------- *)

let%expect_test "fixpoint keeps sweeping and composes into one mapping" =
  (* Each sweep can only trim the permute currently reading the graph input, so
     the chain unwinds one node at a time; the caller still sees a single
     mapping from the original graph to the final one. *)
  let g = Graph_fixtures.permute_identity_chain () in
  run_pipeline g [ Pass.fixpoint trim_identity_at_input ];
  [%expect
    {|
    after:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n3]]
      nodes:
        n3: [t4 f32 [H=2 W=3 C=4]] = relu x=t0
      outputs: [t4 f32 [H=2 W=3 C=4] <-n3]
    map:
      values:
        {t0, t1, t2, t3} -> {t0} identical
      nodes:
        {n0} -> {}
        {n1} -> {}
        {n2} -> {}
      provenance:
        none |}]

let%expect_test "a single sweep of the same pass removes only one node" =
  run_pipeline
    (Graph_fixtures.permute_identity_chain ())
    [ trim_identity_at_input ];
  [%expect
    {|
    after:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n1]]
      nodes:
        n1: [t2 f32 [H=2 W=3 C=4] ->[n2]] = permute x=t0 perm=[]
        n2: [t3 f32 [H=2 W=3 C=4] ->[n3]] = permute x=t2 <-n1 perm=[]
        n3: [t4 f32 [H=2 W=3 C=4]] = relu x=t3 <-n2
      outputs: [t4 f32 [H=2 W=3 C=4] <-n3]
    map:
      values:
        {t0, t1} -> {t0} identical
      nodes:
        {n0} -> {}
      provenance:
        none |}]

let%expect_test "fixpoint reports a pass that will not converge" =
  (* A pass that keeps reporting a change without reaching a fixed point would
     otherwise spin; exhausting the bound is an error rather than a silent stop,
     since the answer would depend on the bound. *)
  let churn =
    Pass.per_node ~name:"churn"
      {
        Pass.on_node =
          (fun _env (n : node) ->
            match (n.Node.op, n.Node.outputs) with
            | Relu { x }, [ out ] ->
                (* Replace the relu with an identical one: always a change. *)
                Some
                  (Recipe.replace ~remove:[ n.Node.id ]
                     ~insert:
                       [
                         {
                           Recipe.op = Relu { x };
                           outputs = [ out ];
                           from = [ n.Node.id ];
                         };
                       ]
                     ())
            | _ -> None);
      }
  in
  run_pipeline ~show:false
    (Graph_fixtures.permute_noop ())
    [ Pass.fixpoint ~max_iters:3 churn ];
  [%expect {| pass churn did not converge |}]

(* ---- pipelines ----------------------------------------------------------- *)

let%expect_test "run_all composes the passes into one SRC to DST mapping" =
  (* Two different passes, one mapping: the second pass's map is composed onto
     the first's, so nothing downstream has to know a pipeline ran. *)
  let g = Graph_fixtures.permute_identity_chain () in
  run_pipeline g [ trim_identity_at_input; trim_identity ];
  [%expect
    {|
    after:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n3]]
      nodes:
        n3: [t4 f32 [H=2 W=3 C=4]] = relu x=t0
      outputs: [t4 f32 [H=2 W=3 C=4] <-n3]
    map:
      values:
        {t0, t1, t2, t3} -> {t0} identical
      nodes:
        {n0} -> {}
        {n1} -> {}
        {n2} -> {}
      provenance:
        none |}]

(* ---- cumulative constants ------------------------------------------------ *)

let%expect_test "constant payloads accumulate across a pipeline" =
  (* Two passes each fold one node into a constant. The second could not run at
     all if the state returned a payload delta rather than carrying them, since
     it needs the graph the first produced and its own payload alongside. *)
  let g = Graph_fixtures.const_arith () in
  let ones shape = Tensor.materialize shape (fun _ -> 1.) in
  let fold_node node out =
    Pass.per_node ~name:"fold"
      {
        Pass.on_node =
          (fun _env (n : node) ->
            if Node_id.equal n.Node.id node then
              Some
                (Recipe.fold_to_constant ~node ~output:out
                   ~value:(ones (Graph_fixtures.s1c 3))
                   ~sources:(Graph_ir.operands n.Node.op))
            else None);
      }
  in
  (match Rewrite.origin g with
  | Error e -> Format.printf "origin: %a@." Rewrite.pp_error e.Core.Error.kind
  | Ok (Rewrite.Origin state) -> (
      Format.printf "@[<v 2>before:@,%a@]@." Graph_ir.pp (Rewrite.graph state);
      match
        Pass.run_all state
          [
            fold_node (Node_id.of_int 0) (t_ 4);
            fold_node (Node_id.of_int 1) (t_ 5);
          ]
      with
      | Error e -> Format.printf "%a@." Pass.pp_error e.Core.Error.kind
      | Ok (Rewrite.Step (final, map)) ->
          Format.printf "@[<v 2>after:@,%a@]@." Graph_ir.pp
            (Rewrite.graph final);
          Format.printf "payloads: %a@."
            (Fmt.brackets (Fmt.list ~sep:Fmt.comma Tensor_id.pp))
            (List.map fst (Tensor_id.Map.bindings (Rewrite.constants final)));
          Format.printf "@[<v 2>map:@,%a@]@." Graph_map.pp map));
  [%expect
    {|
    before:
      graph
      inputs:
        [t0 f32 [C=3] ->[n2], t1 f32 [C=3] ->[n0] constant,
         t2 f32 [C=3] ->[n0] constant, t3 f32 [C=3] ->[n1] constant]
      nodes:
        n0: [t4 f32 [C=3] ->[n1]] = mul a=t1 b=t2
        n1: [t5 f32 [C=3] ->[n2]] = mul a=t4 <-n0 b=t3
        n2: [t6 f32 [C=3]] = add a=t0 b=t5 <-n1
      outputs: [t6 f32 [C=3] <-n2]
    after:
      graph
      inputs: [t0 f32 [C=3] ->[n2], t5 f32 [C=3] ->[n2] constant]
      nodes:
        n2: [t6 f32 [C=3]] = add a=t0 b=t5
      outputs: [t6 f32 [C=3] <-n2]
    payloads: [t5]
    map:
      values:
        {t1} -> {} identical
        {t2} -> {} identical
        {t3} -> {} identical
        {t4} -> {} identical
      nodes:
        {n0} -> {}
        {n1} -> {}
      provenance:
        {t1, t2, t3} -> t5 |}]
