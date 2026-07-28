(* The symbolic transformation verifier: run a pass, then check the map it
   produced actually holds — without payloads for the graph inputs, so a
   [proved] verdict is a statement about EVERY input rather than one sample.
   See .ai/native_transform_verify.md.

   Goldens carry verdicts and ids only, never magnitudes: a printed float would
   make them depend on the platform's floating point, a verdict does not (the
   same reason fold_batch_norm_test.ml prints one). *)

open Graph_ir

type error = [ Map_verify.error | Pass.error | `Origin of Rewrite.error ]

let pp_error ppf : [< error ] -> unit = function
  | `Origin e -> Rewrite.pp_error ppf e
  | #Pass.error as e -> Pass.pp_error ppf e
  | #Map_verify.error as e -> Map_verify.pp_error ppf e

let pp_result pp_ok = Core.Pretty.core_result ~ok:pp_ok ~error:pp_error

let lift_origin (r : ('a, Rewrite.error) Core.result) : ('a, error) Core.result
    =
  Core.map_error (fun e -> `Origin e) r

let lift_pass (r : ('a, Pass.error) Core.result) : ('a, error) Core.result =
  Core.map_error (fun e -> (e :> error)) r

let lift_verify (r : ('a, Map_verify.error) Core.result) :
    ('a, error) Core.result =
  Core.map_error (fun e -> (e :> error)) r

(* Run [passes] over [g] and verify the map that comes back. *)
let verified g passes =
  let open Core.Syntax in
  let* (Rewrite.Origin state) = lift_origin (Rewrite.origin g) in
  let* step = lift_pass (Pass.run_all state passes) in
  lift_verify (Map_verify.step state step)

let check name g passes =
  Format.printf "%s: %a@." name
    (pp_result (fun ppf r ->
         Format.fprintf ppf "%s" (Map_verify.Report.summary r)))
    (verified g passes)

let detail name g passes =
  Format.printf "@[<v 2>%s:@,%a@]@." name
    (pp_result Map_verify.Report.pp_verdicts)
    (verified g passes)

(* ---- the structural passes ------------------------------------------------

   Every one of these rearranges indices without touching arithmetic, so the
   ground terms come out identical and the proof holds for all payloads. *)

let%expect_test "verify: trim / chain permute" =
  check "permute_noop [trim]"
    (Graph_fixtures.permute_noop ())
    [ Trim_permute.pass ];
  check "permute_sequence [trim]"
    (Graph_fixtures.permute_sequence ())
    [ Trim_permute.pass ];
  check "permute_identity_chain [chain;trim]"
    (Graph_fixtures.permute_identity_chain ())
    [ Chain_permute.pass; Trim_permute.pass ];
  check "permute_pair [chain]"
    (Graph_fixtures.permute_pair ())
    [ Chain_permute.pass ];
  check "permute_partial_cancel [chain;trim]"
    (Graph_fixtures.permute_partial_cancel ())
    [ Chain_permute.pass; Trim_permute.pass ];
  check "permute_shared [chain]"
    (Graph_fixtures.permute_shared ())
    [ Chain_permute.pass ];
  [%expect
    {|
    permute_noop [trim]: 2 proved, 0 refuted, 0 unproved, 0 vacuous of 2
    permute_sequence [trim]: 2 proved, 0 refuted, 0 unproved, 1 vacuous of 3
    permute_identity_chain [chain;trim]: 3 proved, 0 refuted, 0 unproved, 1 vacuous of 4
    permute_pair [chain]: 3 proved, 0 refuted, 0 unproved, 1 vacuous of 4
    permute_partial_cancel [chain;trim]: 3 proved, 0 refuted, 0 unproved, 1 vacuous of 4
    permute_shared [chain]: 5 proved, 0 refuted, 0 unproved, 0 vacuous of 5 |}]

let%expect_test "verify: bypass / sink permute" =
  check "sink_permute_unary [sink]"
    (Graph_fixtures.sink_permute_unary ())
    [ Sink_permute.pass ];
  check "sink_permute_binary [sink]"
    (Graph_fixtures.sink_permute_binary ())
    [ Sink_permute.pass ];
  check "sink_permute_broadcast [sink]"
    (Graph_fixtures.sink_permute_broadcast ())
    [ Sink_permute.pass ];
  check "sink_permute_mean_basic [sink_mean]"
    (Graph_fixtures.sink_permute_mean_basic ())
    [ Sink_permute_mean.pass ];
  [%expect
    {|
    sink_permute_unary [sink]: 3 proved, 0 refuted, 0 unproved, 2 vacuous of 5
    sink_permute_binary [sink]: 5 proved, 0 refuted, 0 unproved, 3 vacuous of 8
    sink_permute_broadcast [sink]: 5 proved, 0 refuted, 0 unproved, 3 vacuous of 8
    sink_permute_mean_basic [sink_mean]: 2 proved, 0 refuted, 0 unproved, 2 vacuous of 4 |}]

(* [reuse_permute] is the case that forces the frontier to cross CORRESPONDING
   edges: its map mentions only a deletion and a creation, so proving the
   untouched output cluster has to expand through an edge present in both
   graphs. An expansion that stopped at corresponding edges would leave these
   unproved. Sub and Div are the load-bearing ones — transposing a rebuilt op's
   operands would silently negate or invert rather than fail to type-check. *)
let%expect_test "verify: reuse_permute, including the non-commutative ops" =
  check "reuse_permute_basic"
    (Graph_fixtures.reuse_permute_basic ())
    [ Reuse_permute.pass ];
  check "reuse_permute_sub_order"
    (Graph_fixtures.reuse_permute_sub_order ())
    [ Reuse_permute.pass ];
  check "reuse_permute_div_order"
    (Graph_fixtures.reuse_permute_div_order ())
    [ Reuse_permute.pass ];
  check "reuse_permute_self_inverse"
    (Graph_fixtures.reuse_permute_self_inverse ())
    [ Reuse_permute.pass ];
  [%expect
    {|
    reuse_permute_basic: 4 proved, 0 refuted, 0 unproved, 2 vacuous of 6
    reuse_permute_sub_order: 4 proved, 0 refuted, 0 unproved, 2 vacuous of 6
    reuse_permute_div_order: 4 proved, 0 refuted, 0 unproved, 2 vacuous of 6
    reuse_permute_self_inverse: 3 proved, 0 refuted, 0 unproved, 0 vacuous of 3 |}]

(* ---- what the verifier must NOT prove -------------------------------------

   A test that cannot fail proves nothing, so each of these is a map that is
   wrong in a specific way, checked to come back unproved. Stage 1 has no
   probe, so a failed comparison is [Unproved Exhausted] rather than
   [Refuted] — the prover ran out of moves, it did not exhibit a
   counterexample. *)

let s = Graph_fixtures.nhwc ~h:3 ~w:3 ~c:2

let build name m =
  match Graph_builder.build ~name ~outputs:(fun o -> [ o ]) m with
  | Ok g -> g
  | Error e ->
      invalid_arg
        (Format.asprintf "%a" Graph_builder.pp_error e.Core.Error.kind)

let verify_map map ~src ~dst =
  Format.printf "%a@."
    (pp_result Map_verify.Report.pp_verdicts)
    (lift_verify (Map_verify.run map ~src ~dst))

(* Two graphs with identical shapes and ids but a different operator. An
   identity map claims every edge is bit-identical; the output edge is not. *)
let%expect_test "verify: an identity map over a changed operator is unproved" =
  let binop op =
    build "binop"
      Graph_builder.(
        let* a = input ~shape:s ~name:"a" () in
        let* b = input ~shape:s ~name:"b" () in
        op a b)
  in
  let src = binop (fun a b -> Graph_builder.add a b) in
  let dst = binop (fun a b -> Graph_builder.sub a b) in
  verify_map Graph_map.identity ~src ~dst;
  [%expect
    {|
    {t0} -> {t0} identical: proved (structural) [exhaustive]
    {t1} -> {t1} identical: proved (structural) [exhaustive]
    {t2} -> {t2} identical: unproved: exhausted at (0): src.t2 vs dst.t2 [exhaustive] |}]

(* The cluster {t0, t1} <-> {t0} is the shape trimming an identity permute
   produces: the input and the trimmed output both correspond to the surviving
   edge. Here the permute is a REAL one, so t1 is not t0 and the claim is
   false — but only for the non-canonical member. The canonical member pairs
   with t0 on both sides and agrees, so a checker that compared one
   representative per side would call this proved.

   t1 is dead in the source (the relu reads t0 directly), which keeps the lie
   confined to this one cluster: the output cluster is honest and must still
   prove. Shape-preserving (H = W = 3) so the failure is a value difference,
   not a shape mismatch. *)
let%expect_test "verify: a false claim about a non-canonical cluster member" =
  let src =
    build "src"
      Graph_builder.(
        let* a = input ~shape:s ~name:"a" () in
        let* _dead = permute Graph_fixtures.swap_hw a in
        relu a)
  in
  let dst =
    build "dst"
      Graph_builder.(
        let* a = input ~shape:s ~name:"a" () in
        relu a)
  in
  let ids l = Tensor_id.Set.of_list (List.map Tensor_id.of_int l) in
  let cluster src dst =
    {
      Correspondence.Cluster.src = ids src;
      dst = ids dst;
      label = Correspondence.Identical;
    }
  in
  let map =
    {
      Graph_map.values =
        Correspondence.of_clusters
          [ cluster [ 0; 1 ] [ 0 ]; cluster [ 2 ] [ 1 ] ];
      (* the source's dead permute node goes away; the two relus pair up *)
      nodes =
        Node_map.of_clusters
          [
            Node_map.delete (Node_id.of_int 0);
            Node_map.pair (Node_id.of_int 1) (Node_id.of_int 0);
          ];
      provenance = Provenance.empty;
    }
  in
  verify_map map ~src ~dst;
  [%expect
    {|
    {t0, t1} -> {t0} identical: unproved: exhausted at (1,0): src.t0 vs src.t1 [exhaustive]
    {t2} -> {t1} identical: proved (structural) [exhaustive] |}]

(* ---- the rounding boundary ------------------------------------------------

   Every node output is materialized as float32 (Schedule.evaluate ->
   Tensor.materialize), so a stage boundary rounds. Inlining that away would
   turn f32(f32(a+b)*c) into f32((a+b)*c) and let a future fusion be "proved"
   identical while changing bits. [Round] keeps the boundary in the term; these
   pin the three rules that may remove one. *)

let cell n = { Ground_expr.Cell.id = Tensor_id.of_int n; coord = Vec6.origin }

let show_norm ~stored_f32 e =
  let n = Ground_expr.normalise ~stored_f32 e in
  Format.printf "%a  blocked=[%a]@." Ground_expr.pp n.Ground_expr.expr
    (Fmt.list ~sep:Fmt.comma Ground_expr.Cell.pp)
    (Ground_expr.Cell.Set.elements n.Ground_expr.blocked)

let%expect_test "normalise: a cell is already stored, so its Round collapses" =
  let all_f32 _ = true in
  show_norm ~stored_f32:all_f32 (Ground_expr.Round (Ground_expr.Cell (cell 0)));
  (* idempotent *)
  show_norm ~stored_f32:all_f32
    (Ground_expr.Round (Ground_expr.Round (Ground_expr.Cell (cell 0))));
  (* a constant is folded to its f32 image, so a fold can be compared bitwise
     against a payload the pass computed through the same materialization *)
  show_norm ~stored_f32:all_f32 (Ground_expr.Round (Ground_expr.Const 0.1));
  [%expect
    {|
    t0(0)  blocked=[]
    t0(0)  blocked=[]
    0x1.99999ap-4  blocked=[] |}]

let%expect_test "normalise: a computed Round is NOT removed" =
  (* The whole point: only a stored value or a constant may lose its boundary.
     An arithmetic node keeps it, so two graphs that differ only in where they
     materialize do not compare equal. *)
  show_norm
    ~stored_f32:(fun _ -> true)
    (Ground_expr.Round
       (Ground_expr.Binary
          (Expr.Add, Ground_expr.Cell (cell 0), Ground_expr.Cell (cell 1))));
  [%expect {| f32((t0(0) + t1(0)))  blocked=[] |}]

let%expect_test "normalise: a non-f32 cell blocks the collapse" =
  (* [Payload.get_float] decodes I32/I64 via Int32/Int64.to_float, which leaves
     f32's exact range above 2^24, and I8/I16 through a dequantizing multiply.
     For those the materialization is observable, so the Round has to stay. *)
  show_norm
    ~stored_f32:(fun _ -> false)
    (Ground_expr.Round (Ground_expr.Cell (cell 0)));
  [%expect {| f32(t0(0))  blocked=[t0(0)] |}]

(* End to end: trimming an identity permute off a non-F32 input is not merely
   unproven, it is FALSE for a large enough value — the permute's f32
   materialization is what the source computes and the destination skips. The
   verifier reports the blocked collapse rather than proving it. *)
let%expect_test "verify: trimming a permute off an i32 input is unproved" =
  let g =
    build "i32_permute"
      Graph_builder.(
        let* a = input ~shape:s ~name:"a" ~fmt:(Payload.Fmt Payload.I32) () in
        let* t1 = permute Graph_fixtures.identity_perm a in
        relu t1)
  in
  detail "i32 input [trim]" g [ Trim_permute.pass ];
  [%expect
    {|
    i32 input [trim]:
      {t0, t1} -> {t0} identical: unproved: format blocks collapse: t0(0) in src.t1 [exhaustive]
      {t2} -> {t2} identical: unproved: format blocks collapse: t0(0) in src.t2 [exhaustive] |}]
