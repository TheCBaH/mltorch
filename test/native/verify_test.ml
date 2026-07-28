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
    permute_noop [trim]: 2 proved, 0 refuted, 0 tested, 0 unproved, 0 vacuous of 2
    permute_sequence [trim]: 2 proved, 0 refuted, 0 tested, 0 unproved, 1 vacuous of 3
    permute_identity_chain [chain;trim]: 3 proved, 0 refuted, 0 tested, 0 unproved, 1 vacuous of 4
    permute_pair [chain]: 3 proved, 0 refuted, 0 tested, 0 unproved, 1 vacuous of 4
    permute_partial_cancel [chain;trim]: 3 proved, 0 refuted, 0 tested, 0 unproved, 1 vacuous of 4
    permute_shared [chain]: 5 proved, 0 refuted, 0 tested, 0 unproved, 0 vacuous of 5 |}]

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
    sink_permute_unary [sink]: 3 proved, 0 refuted, 0 tested, 0 unproved, 2 vacuous of 5
    sink_permute_binary [sink]: 5 proved, 0 refuted, 0 tested, 0 unproved, 3 vacuous of 8
    sink_permute_broadcast [sink]: 5 proved, 0 refuted, 0 tested, 0 unproved, 3 vacuous of 8
    sink_permute_mean_basic [sink_mean]: 2 proved, 0 refuted, 0 tested, 0 unproved, 2 vacuous of 4 |}]

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
    reuse_permute_basic: 4 proved, 0 refuted, 0 tested, 0 unproved, 2 vacuous of 6
    reuse_permute_sub_order: 4 proved, 0 refuted, 0 tested, 0 unproved, 2 vacuous of 6
    reuse_permute_div_order: 4 proved, 0 refuted, 0 tested, 0 unproved, 2 vacuous of 6
    reuse_permute_self_inverse: 3 proved, 0 refuted, 0 tested, 0 unproved, 0 vacuous of 3 |}]

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
    {t2} -> {t2} identical: refuted: value at (0): src.t2 vs dst.t2 under {t0(0)=0x1p+0, t1(0)=0x1.98p+6} [exhaustive] |}]

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
    {t0, t1} -> {t0} identical: refuted: value at (1,0): src.t0 vs src.t1 under {t0(1,0)=0x1p+3, t0(1,0,0)=0x1.bp+5} [exhaustive]
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

(* ---- constant payloads ----------------------------------------------------

   Binding the model's constants narrows what a proof quantifies over — every
   INPUT, for these constants, rather than every payload — so it is only
   attempted when the unqualified comparison fails. The permute passes above
   stay at plain [structural] because they never need it; [fold_const] cannot
   be proved without it, since the destination edge IS a payload the pass
   computed. *)

let verified_with ?budget ?probe ~constants g passes =
  let open Core.Syntax in
  let* (Rewrite.Origin state) = lift_origin (Rewrite.origin ~constants g) in
  let* step = lift_pass (Pass.run_all state passes) in
  lift_verify (Map_verify.step ?budget ?probe state step)

let check_with ?budget ?probe ~constants name g passes =
  Format.printf "@[<v 2>%s:@,%a@]@." name
    (pp_result Map_verify.Report.pp_verdicts)
    (verified_with ?budget ?probe ~constants g passes)

let w_shape = Graph_fixtures.s 3 1 1 2 2 2

let hw_ramp shape =
  Tensor.materialize shape (fun c ->
      float_of_int
        ((Dim.to_int (Vec6.get c Axis.H) * 10) + Dim.to_int (Vec6.get c Axis.W)))

let%expect_test "verify: fold_const needs the constants, and gets them" =
  check_with
    ~constants:[ (Tensor_id.of_int 1, hw_ramp w_shape) ]
    "const_permute [fold_const]"
    (Graph_fixtures.const_permute ())
    [ Fold_const.pass ];
  [%expect
    {|
    const_permute [fold_const]:
      {t1} -> {} identical: vacuous
      {t0} -> {t0} identical: proved (structural) [exhaustive]
      {t2} -> {t2} identical: proved (structural, for these constants) [exhaustive]
      {t3} -> {t3} identical: proved (structural) [exhaustive] |}]

(* Folding is only correct if the pass reproduced the source's arithmetic
   exactly, materialization included. Perturbing the DESTINATION payload is how
   to simulate a fold that computed the wrong number — perturbing the source
   constant instead would just be folded faithfully and prove.

   The refuted terms here are closed (both sides are constants), so the witness
   is the empty valuation: two closed terms that differ need no assignment to
   separate them. *)
let%expect_test "verify: a fold that computed the wrong payload is refuted" =
  let bump (Tensor.Tensor t as packed) =
    Tensor.materialize t.Tensor.shape (fun c -> Tensor.read packed c +. 1.)
  in
  let result =
    let open Core.Syntax in
    let* (Rewrite.Origin state) =
      lift_origin
        (Rewrite.origin
           ~constants:[ (Tensor_id.of_int 1, hw_ramp w_shape) ]
           (Graph_fixtures.const_permute ()))
    in
    let* (Rewrite.Step (final, map)) =
      lift_pass (Pass.run_all state [ Fold_const.pass ])
    in
    lift_verify
      (Map_verify.run map ~src:(Rewrite.graph state)
         ~src_constants:(Rewrite.constants state) ~dst:(Rewrite.graph final)
         ~dst_constants:(Tensor_id.Map.map bump (Rewrite.constants final)))
  in
  Format.printf "@[<v 2>const_permute, folded payload off by one:@,%a@]@."
    (pp_result Map_verify.Report.pp_verdicts)
    result;
  [%expect
    {|
    const_permute, folded payload off by one:
      {t1} -> {} identical: vacuous
      {t0} -> {t0} identical: proved (structural) [exhaustive]
      {t2} -> {t2} identical: refuted: value at (0): src.t2 vs dst.t2 under {} [exhaustive]
      {t3} -> {t3} identical: proved (structural) [exhaustive] |}]

(* A probe may only run once expansion has reached the graph inputs. Cells left
   at a truncated frontier are internal stage results constrained by their
   producers, so assigning them independently could manufacture a
   "counterexample" no input can realise. Starve the rounds and the verdict must
   be [max_rounds] — never a refutation. *)
let%expect_test "verify: a truncated frontier never refutes" =
  let starved = { Map_verify.Budget.default with max_rounds = 0 } in
  check_with ~budget:starved ~constants:[]
    "reuse_permute_sub_order, no expansion allowed"
    (Graph_fixtures.reuse_permute_sub_order ())
    [ Reuse_permute.pass ];
  [%expect
    {|
    reuse_permute_sub_order, no expansion allowed:
      {t3} -> {} identical: vacuous
      {} -> {t5} identical: vacuous
      {t0} -> {t0} identical: proved (structural) [exhaustive]
      {t1} -> {t1} identical: proved (structural) [exhaustive]
      {t2} -> {t2} identical: proved (structural) [exhaustive]
      {t4} -> {t4} identical: unproved: over max_rounds [exhaustive] |}]

(* ---- cumulative verification ----------------------------------------------

   [Pass.run_all] already threads [Graph_map.compose], so everything above
   verifies a COMPOSED map whenever it is handed more than one pass. What is
   added here is the other half: verifying each step on its own, so a failure
   names the pass that caused it, and comparing the two.

   The per-step chain is already a proof of the end-to-end claim PROVIDED
   composition is sound; the composed check is what tests that proviso. It
   catches a composition error that makes a composed claim false at the
   endpoints — it does not validate compose's algebraic contract in general,
   since an over-conservative label is legal and therefore unverifiable, and a
   cluster set that is wrong but endpoint-consistent still passes. That stays
   graph_map_test.ml's job. *)

let cumulative = Map_verify.Budget.cumulative

(* Apply passes one at a time, verifying each step against the state it started
   from, so a failure names the pass that caused it. *)
let per_step ?constants g passes =
  let open Core.Syntax in
  let* (Rewrite.Origin origin) = lift_origin (Rewrite.origin ?constants g) in
  let rec go : type v.
      v Rewrite.t ->
      Pass.t list ->
      ((string * Map_verify.Report.t) list, error) Core.result =
   fun state -> function
     | [] -> Core.return []
     | p :: rest ->
         let* (Rewrite.Step (next, _) as step) =
           lift_pass (Pass.run_all state [ p ])
         in
         let* report =
           lift_verify (Map_verify.step ~budget:cumulative state step)
         in
         let+ rest = go next rest in
         (p.Pass.name, report) :: rest
  in
  go origin passes

let composed ?constants g passes =
  let open Core.Syntax in
  let* (Rewrite.Origin origin) = lift_origin (Rewrite.origin ?constants g) in
  let* step = lift_pass (Pass.run_all origin passes) in
  lift_verify (Map_verify.step ~budget:cumulative origin step)

(* The law worth pinning: if every step verifies, composition must not turn that
   into a refutation. The converse is NOT a law — a composed [Unproved] where
   every step is [Proved] is an acceptable outcome, since the composed frontier
   spans the whole pipeline and can run out of budget where a single step does
   not. Verification strength is also not monotone under composition: two steps
   whose roundings cancel can compose to a bit-identical pair. *)
let both name ?constants g passes =
  let report =
    let open Core.Syntax in
    let* steps = per_step ?constants g passes in
    let+ composed = composed ?constants g passes in
    let lines =
      List.map
        (fun (n, r) -> Printf.sprintf "%s: %s" n (Map_verify.Report.summary r))
        steps
      @ [ Printf.sprintf "composed: %s" (Map_verify.Report.summary composed) ]
    in
    let every_step_proved =
      List.for_all (fun (_, r) -> Map_verify.Report.proved r) steps
    in
    lines
    @ [
        Printf.sprintf "law (every step proved => composed not refuted): %b"
          ((not every_step_proved) || not (Map_verify.Report.refuted composed));
      ]
  in
  Format.printf "@[<v 2>%s:@,%a@]@." name
    (pp_result (Fmt.list ~sep:Fmt.cut Fmt.string))
    report

let%expect_test "verify: each step and their composition agree" =
  both "permute_identity_chain"
    (Graph_fixtures.permute_identity_chain ())
    [ Chain_permute.pass; Trim_permute.pass ];
  [%expect
    {|
    permute_identity_chain:
      chain_permute: 4 proved, 0 refuted, 0 tested, 0 unproved, 1 vacuous of 5
      trim_permute: 3 proved, 0 refuted, 0 tested, 0 unproved, 0 vacuous of 3
      composed: 3 proved, 0 refuted, 0 tested, 0 unproved, 1 vacuous of 4
      law (every step proved => composed not refuted): true |}]

(* Cross-iteration composition: a fixpoint fold collapses a multi-node constant
   sub-DAG one node at a time, so the composed map is a chain of per-iteration
   maps rather than a single step's. *)
let%expect_test "verify: a fixpoint over a constant sub-DAG" =
  let shape = Graph_fixtures.s1c 3 in
  let ramp base =
    Tensor.materialize shape (fun c ->
        base +. float_of_int (Dim.to_int (Vec6.get c Axis.C)))
  in
  let constants =
    [
      (Tensor_id.of_int 1, ramp 1.);
      (Tensor_id.of_int 2, ramp 10.);
      (Tensor_id.of_int 3, ramp 100.);
    ]
  in
  both "const_arith [fixpoint fold_const]" ~constants
    (Graph_fixtures.const_arith ())
    [ Pass.fixpoint Fold_const.pass ];
  [%expect
    {|
    const_arith [fixpoint fold_const]:
      fold_const: 3 proved, 0 refuted, 0 tested, 0 unproved, 4 vacuous of 7
      composed: 3 proved, 0 refuted, 0 tested, 0 unproved, 4 vacuous of 7
      law (every step proved => composed not refuted): true |}]

(* Terminal id packing renumbers post-origin ids, including graph inputs, and
   composing its map is the {t11} -> {} then {t12} -> {t11} hazard §9 of
   native_transform_design.md warns about: a resurrected dead id would fuse two
   clusters and claim the dead edge corresponds to the packed one. Verifying
   origin -> passes -> pack end to end is the check that a resurrection would
   actually be caught, rather than just producing a plausible-looking map. *)
let%expect_test "verify: origin -> passes -> pack, composed" =
  let result =
    let open Core.Syntax in
    let* (Rewrite.Origin s0) =
      lift_origin (Rewrite.origin (Graph_fixtures.permute_identity_chain ()))
    in
    let* (Rewrite.Step (s1, m01)) =
      lift_pass (Pass.run_all s0 [ Chain_permute.pass; Trim_permute.pass ])
    in
    let* (Rewrite.Step (s2, m12)) = lift_origin (Rewrite.pack s1) in
    let+ report =
      lift_verify
        (Map_verify.run ~budget:Map_verify.Budget.cumulative
           (Graph_map.compose m01 m12)
           ~src:(Rewrite.graph s0) ~dst:(Rewrite.graph s2))
    in
    Map_verify.Report.summary report
  in
  Format.printf "permute_identity_chain, passes then pack: %a@."
    (pp_result Fmt.string) result;
  [%expect
    {| permute_identity_chain, passes then pack: 3 proved, 0 refuted, 0 tested, 0 unproved, 1 vacuous of 4 |}]

(* ---- the pipeline hook ----------------------------------------------------

   [Pass.run_all ~verify] checks each step as it is applied, so the first
   offending pass stops the pipeline and the error names it. These live here
   rather than in pass_test.ml because what is under test is the verifier's
   effect on the driver, and the fixtures are already to hand. *)

(* Deliberately wrong: trims EVERY permute, tying its output to its input, when
   only an identity permute may be trimmed that way. The claim it leaves behind
   is [Identical] between two edges that differ. *)
let trim_any_permute =
  Pass.per_node ~name:"trim_any_permute"
    {
      Pass.on_node =
        (fun _env (n : node) ->
          match (n.Node.op, n.Node.outputs) with
          | Permute { x; _ }, [ out ] ->
              Some (Recipe.trim ~remove:[ n.Node.id ] ~tie:[ (out, x) ])
          | _ -> None);
    }

let piped ?verify g passes =
  let open Core.Syntax in
  let* (Rewrite.Origin state) = lift_origin (Rewrite.origin g) in
  let+ (Rewrite.Step (final, _)) =
    lift_pass (Pass.run_all ?verify state passes)
  in
  Format.asprintf "%d nodes" (List.length (Rewrite.graph final).Graph.nodes)

let%expect_test "hook: a broken pass is caught, and named" =
  let g () =
    build "swap"
      Graph_builder.(
        let* a = input ~shape:s ~name:"a" () in
        let* t1 = permute Graph_fixtures.swap_hw a in
        relu t1)
  in
  (* unverified, the bad rewrite sails through *)
  Format.printf "no policy: %a@." (pp_result Fmt.string)
    (piped (g ()) [ trim_any_permute ]);
  Format.printf "%a@." (pp_result Fmt.string)
    (piped ~verify:Map_verify.Policy.Require_proved (g ()) [ trim_any_permute ]);
  [%expect
    {|
    no policy: 1 nodes
    pass trim_any_permute rejected: 0 proved, 2 refuted, 0 tested, 0 unproved, 0 vacuous of 2
      {t0, t1} -> {t0} identical: refuted: value at (1,0): src.t0 vs src.t1 under {t0(1,0)=0x1p+3, t0(1,0,0)=0x1.bp+5} [exhaustive]
      {t2} -> {t2} identical: refuted: value at (1,0): src.t2 vs dst.t2 under {t0(1,0)=0x1p+3, t0(1,0,0)=0x1.bp+5} [exhaustive] |}]

(* The two policies exist because [Unproved] and [Refuted] are different
   answers. The i32 trim is genuinely unproven — and in fact false for a large
   enough value — but the verifier has exhibited no counterexample, so the
   release bar tolerates it while the development bar does not. *)
let%expect_test
    "hook: Reject_refuted tolerates unproved, Require_proved does not" =
  let g () =
    build "i32_permute"
      Graph_builder.(
        let* a = input ~shape:s ~name:"a" ~fmt:(Payload.Fmt Payload.I32) () in
        let* t1 = permute Graph_fixtures.identity_perm a in
        relu t1)
  in
  Format.printf "reject_refuted: %a@." (pp_result Fmt.string)
    (piped ~verify:Map_verify.Policy.Reject_refuted (g ()) [ Trim_permute.pass ]);
  Format.printf "require_proved: %a@." (pp_result Fmt.string)
    (piped ~verify:Map_verify.Policy.Require_proved (g ()) [ Trim_permute.pass ]);
  [%expect
    {|
    reject_refuted: 1 nodes
    require_proved: pass trim_permute rejected: 0 proved, 0 refuted, 0 tested, 2 unproved, 0 vacuous of 2
                      {t0, t1} -> {t0} identical: unproved: format blocks collapse: t0(0) in src.t1 [exhaustive]
                      {t2} -> {t2} identical: unproved: format blocks collapse: t0(0) in src.t2 [exhaustive] |}]

(* ---- the coefficient tier -------------------------------------------------

   Batch-norm folding re-associates: [(Σ xₖ·Wₖ)·s] becomes [Σ xₖ·(Wₖ·s)]. No
   structural comparison reaches that, and no exact one does either — the pass
   re-derives its constants numerically, and eps arrives as a constant EDGE with
   a payload against a source-side [Const]. So the honest verdict is agreement
   of the polynomial coefficients within a tolerance, which is evidence and
   never a proof.

   The eight-way check lives in fold_batch_norm_test.ml, next to the numeric one
   it sits alongside. What is here is the negative: a fold that is actually
   wrong must NOT come back agreeing, and — because the claim is [Equivalent],
   not [Identical] — must not come back refuted either. *)

let%expect_test "coefficients: a wrong fold disagrees, and is not refuted" =
  let s1c = Graph_fixtures.s1c in
  let weight_shape =
    Graph_fixtures.weight_shape ~out_channels:3 ~in_channels:2
  in
  let vec a =
    Tensor.materialize (s1c 3) (fun c -> a.(Dim.to_int (Vec6.get c Axis.C)))
  in
  let g =
    build "conv_bn"
      Graph_builder.(
        let* x = input ~shape:(Graph_fixtures.nhwc ~h:4 ~w:4 ~c:2) () in
        let* w = constant ~shape:weight_shape () in
        let* mean = constant ~shape:(s1c 3) () in
        let* var = constant ~shape:(s1c 3) () in
        let* y =
          conv2d (Graph_fixtures.conv_params ~in_channels:2) ~x ~weight:w ()
        in
        batch_norm Graph_fixtures.bn_params ~x:y ~running_mean:mean
          ~running_var:var ())
  in
  let ids =
    List.filter
      (fun id -> Graph_ir.input_kind g id = Input.Constant)
      g.Graph.inputs
  in
  let constants =
    List.combine ids
      [ hw_ramp weight_shape; vec [| 0.5; 1.; 1.5 |]; vec [| 4.; 1.; 0.25 |] ]
  in
  (* Scaling every folded payload by two changes the coefficients, not just
     their last bits, so tolerance cannot absorb it. *)
  let doubled (Tensor.Tensor t as packed) =
    Tensor.materialize t.Tensor.shape (fun c -> Tensor.read packed c *. 2.)
  in
  let report ~dst_constants name =
    let result =
      let open Core.Syntax in
      let* (Rewrite.Origin state) = lift_origin (Rewrite.origin ~constants g) in
      let* (Rewrite.Step (final, map)) =
        lift_pass (Pass.run_all state [ Fold_batch_norm.pass ])
      in
      lift_verify
        (Map_verify.run ~budget:Map_verify.Budget.cumulative map
           ~src:(Rewrite.graph state) ~src_constants:(Rewrite.constants state)
           ~dst:(Rewrite.graph final)
           ~dst_constants:(dst_constants (Rewrite.constants final)))
    in
    Format.printf "%s: %a@." name
      (pp_result (fun ppf r ->
           Fmt.list ~sep:(Fmt.any "; ")
             (fun ppf (_, (o : Map_verify.Outcome.t)) ->
               Map_verify.Verdict.pp ppf o.verdict)
             ppf
             (List.filter
                (fun (_, (o : Map_verify.Outcome.t)) ->
                  match o.verdict with
                  | Map_verify.Verdict.Vacuous -> false
                  | _ -> true)
                r.Map_verify.Report.clusters)))
      result
  in
  report ~dst_constants:Fun.id "honest fold";
  report ~dst_constants:(Tensor_id.Map.map doubled) "folded payloads doubled";
  [%expect
    {|
    honest fold: tested: agrees (1e-05); proved (structural); proved (structural); proved (structural); proved (structural)
    folded payloads doubled: tested: disagrees at {t0(0)=0x1p+0, t0(1)=0x1p+1, t0(1,0)=0x1p+3, t0(1,1)=0x1.2p+3, t0(1,0,0)=0x1.bp+5, t0(1,0,1)=0x1.b8p+5, t0(1,1,0)=0x1.e8p+5, t0(1,1,1)=0x1.fp+5}; proved (structural); proved (structural); proved (structural); proved (structural) |}]
