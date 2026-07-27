(* Exercises the match combinators, and pins each semantic the design commits to
   — anchoring, greedy overlap, rollback, idempotent re-claiming, and chain
   progress. See .ai/native_transform_design.md §11. *)

open Graph_ir

let t_ n = Tensor_id.of_int n
let view_of g = Graph_view.of_graph g |> Result.get_ok
let ids pp_id fmt l = Fmt.brackets (Fmt.list ~sep:Fmt.comma pp_id) fmt l

(* Projectors: a pattern destructures the payload records directly, so these are
   all a matcher needs to say "this kind of node". *)
let relu = function Relu (r : Pointwise.Relu.t) -> Some r | _ -> None
let permute = function Permute (p : Permute.Permute.t) -> Some p | _ -> None
let mul = function Mul (m : Pointwise.Bin.t) -> Some m | _ -> None
let add = function Add (a : Pointwise.Bin.t) -> Some a | _ -> None
let conv = function Conv2d (c : Conv.Conv2d.t) -> Some c | _ -> None

let show_match g pattern anchor =
  let v = view_of g in
  match Pattern.run (pattern anchor) v with
  | Error f -> Format.printf "no match: %a@." Pattern.pp_failure f
  | Ok ((), r) ->
      Format.printf "@[<v>%a@,@[<v 2>matched:@,%a@]@]@." Region.pp r Graph_ir.pp
        (Region.extract v r)

let show_failure g pattern anchor =
  let v = view_of g in
  match Pattern.run (pattern anchor) v with
  | Error f -> Format.printf "%a@." Pattern.pp_failure f
  | Ok ((), (r : Region.t)) ->
      Format.printf "matched %a@." (ids Node_id.pp)
        (Node_id.Set.elements r.nodes)

(* ---- walking up ---------------------------------------------------------- *)

let%expect_test "a linear chain is a let* sequence up the operand edges" =
  (* conv -> batch_norm -> relu, anchored at the relu's output. *)
  let g = Graph_fixtures.chain () in
  show_match g
    (fun out ->
      Pattern.(
        let* r, _ = def out relu in
        let* () = interior r.x in
        let* b, _ = def r.x (function Batch_norm b -> Some b | _ -> None) in
        let* () = interior b.x in
        let+ _ = def b.x conv in
        ()))
    (t_ 9);
  [%expect
    {|
    nodes: [n0, n1, n2]
    inputs: [t0, t1, t2, t3, t4, t5, t6]
    outputs: [t9]
    interior: [t7, t8]
    convex: true
    matched:
      graph
      inputs:
        [t0 f32 [H=4 W=4 C=2] ->[n0],
         t1 f32 [N=3 T=1 D=1 H=2 W=2 C=2] ->[n0] constant,
         t2 f32 [C=3] ->[n0] constant, t3 f32 [C=3] ->[n1] constant,
         t4 f32 [C=3] ->[n1] constant, t5 f32 [C=3] ->[n1] constant,
         t6 f32 [C=3] ->[n1] constant]
      nodes:
        n0: [t7 f32 [H=3 W=3 C=3] ->[n1]] =
          conv2d
            x=t0
            weight=t1
            bias=t2
            params={h={kernel=2; stride=1; pad_before=0; pad_after=0; dilation=1};
                   w={kernel=2; stride=1; pad_before=0; pad_after=0; dilation=1};
                   in_channels=2;
                   groups=1}
        n1: [t8 f32 [H=3 W=3 C=3] ->[n2]] =
          batch_norm
            x=t7 <-n0
            weight=t3
            bias=t4
            running_mean=t5
            running_var=t6
            params={channel=C; eps=1e-05}
        n2: [t9 f32 [H=3 W=3 C=3]] = relu x=t8 <-n1
      outputs: [t9 f32 [H=3 W=3 C=3] <-n2] |}]

let%expect_test "a union is two def calls on two operand fields" =
  let g = Graph_fixtures.diamond () in
  show_match g
    (fun out ->
      Pattern.(
        let* a, _ = def out add in
        let* _ = def a.a relu in
        let+ _ = def a.b mul in
        ()))
    (t_ 3);
  [%expect
    {|
    nodes: [n0, n1, n2]
    inputs: [t0]
    outputs: [t3]
    interior: [t1, t2]
    convex: true
    matched:
      graph
      inputs: [t0 f32 [C=4] ->[n0, n1]]
      nodes:
        n0: [t1 f32 [C=4] ->[n2]] = relu x=t0
        n1: [t2 f32 [C=4] ->[n2]] = mul a=t0 b=t0
        n2: [t3 f32 [C=4]] = add a=t1 <-n0 b=t2 <-n1
      outputs: [t3 f32 [C=4] <-n2] |}]

let%expect_test "re-claiming a node inside one match is idempotent" =
  (* Both arms of the diamond reach the same producer of t0; if claiming twice
     failed, no diamond pattern could ever match. *)
  let g = Graph_fixtures.diamond () in
  show_match g
    (fun out ->
      Pattern.(
        let* a, _ = def out add in
        let* _, left = def a.a relu in
        let* _, right = def a.b mul in
        let* () = claim left in
        let* () = claim right in
        let+ () = claim left in
        ()))
    (t_ 3);
  [%expect
    {|
    nodes: [n0, n1, n2]
    inputs: [t0]
    outputs: [t3]
    interior: [t1, t2]
    convex: true
    matched:
      graph
      inputs: [t0 f32 [C=4] ->[n0, n1]]
      nodes:
        n0: [t1 f32 [C=4] ->[n2]] = relu x=t0
        n1: [t2 f32 [C=4] ->[n2]] = mul a=t0 b=t0
        n2: [t3 f32 [C=4]] = add a=t1 <-n0 b=t2 <-n1
      outputs: [t3 f32 [C=4] <-n2] |}]

(* ---- fan-out ------------------------------------------------------------- *)

let%expect_test "interior rejects a shared edge and a graph output" =
  let g = Graph_fixtures.diamond () in
  (* t0 feeds both the relu and the mul. *)
  show_failure g
    (fun _ ->
      Pattern.(
        let+ () = interior (t_ 0) in
        ()))
    (t_ 0);
  (* t3 is the graph output, so it escapes even with no consumer. *)
  show_failure g
    (fun _ ->
      Pattern.(
        let+ () = interior (t_ 3) in
        ()))
    (t_ 3);
  (* t1 is consumed once by the add, so [interior] itself passes — but the
     pattern claimed no node, and a match with no nodes has no region. *)
  show_failure g
    (fun _ ->
      Pattern.(
        let+ () = interior (t_ 1) in
        ()))
    (t_ 1);
  [%expect
    {|
    t0 has more than one consumer
    t3 is used outside the match or is a graph output
    region claims no nodes |}]

let%expect_test "an edge read twice by one node is a single consumer" =
  (* mul x x: one consumer, two uses. Counting uses rather than consumers would
     refuse to fuse a perfectly fusable edge. *)
  let g = Graph_fixtures.diamond () in
  let v = view_of g in
  Format.printf "consumers of t0: %a@." (ids Node_id.pp)
    (List.map (fun (n : node) -> n.Node.id) (Graph_view.uses v (t_ 0)));
  [%expect {| consumers of t0: [n0, n1] |}]

let%expect_test "use walks forward, and refuses when the choice is ambiguous" =
  let g = Graph_fixtures.diamond () in
  (* Exactly one consumer of t0 is a relu. *)
  show_failure g
    (fun _ ->
      Pattern.(
        let+ _ = use (t_ 0) relu in
        ()))
    (t_ 0);
  (* But two consumers if we accept anything, so the pattern must not guess. *)
  show_failure g
    (fun _ ->
      Pattern.(
        let+ _ = use (t_ 0) (fun _ -> Some ()) in
        ()))
    (t_ 0);
  [%expect {|
    matched [n0]
    several consumers of t0 match |}]

(* ---- alternation and rollback -------------------------------------------- *)

let%expect_test "a failed branch discards its claims" =
  (* The first alternative claims the relu, then fails on the conv. The second
     claims only the relu's producer chain, so the region must not contain
     anything the abandoned branch touched. *)
  let g = Graph_fixtures.permute_noop () in
  show_match g
    (fun out ->
      Pattern.(
        choice
          [
            (let* r, _ = def out relu in
             let* _ = def r.x conv in
             return ());
            (let+ _ = def out relu in
             ());
          ]))
    (t_ 2);
  [%expect
    {|
    nodes: [n1]
    inputs: [t1]
    outputs: [t2]
    interior: []
    convex: true
    matched:
      graph
      inputs: [t1 f32 [H=2 W=3 C=4] ->[n1]]
      nodes:
        n1: [t2 f32 [H=2 W=3 C=4]] = relu x=t1
      outputs: [t2 f32 [H=2 W=3 C=4] <-n1] |}]

let%expect_test "optional yields None without failing the match" =
  let g = Graph_fixtures.permute_noop () in
  show_match g
    (fun out ->
      Pattern.(
        let* _ = def out relu in
        let+ missing = optional (def out conv) in
        ignore missing))
    (t_ 2);
  [%expect
    {|
    nodes: [n1]
    inputs: [t1]
    outputs: [t2]
    interior: []
    convex: true
    matched:
      graph
      inputs: [t1 f32 [H=2 W=3 C=4] ->[n1]]
      nodes:
        n1: [t2 f32 [H=2 W=3 C=4]] = relu x=t1
      outputs: [t2 f32 [H=2 W=3 C=4] <-n1] |}]

(* ---- repetition ---------------------------------------------------------- *)

let%expect_test "chain walks a run of permutes and stops at the input" =
  let g = Graph_fixtures.permute_sequence () in
  let v = view_of g in
  (match
     Pattern.run
       Pattern.(
         let* r, _ = def (t_ 3) relu in
         chain
           (fun edge ->
             let+ p, _ = def edge permute in
             (p.perm, p.x))
           r.x)
       v
   with
  | Error f -> Format.printf "no match: %a@." Pattern.pp_failure f
  | Ok (perms, r) ->
      Format.printf "%d permutes, region %a@." (List.length perms)
        (ids Node_id.pp)
        (Node_id.Set.elements r.nodes));
  [%expect {| 2 permutes, region [n0, n1, n2] |}]

let%expect_test "chain rejects a step that does not move upstream" =
  (* A step returning the edge it was given would loop forever; the progress
     rule turns that into a mismatch. *)
  let g = Graph_fixtures.permute_sequence () in
  show_failure g
    (fun _ ->
      Pattern.(
        let+ _ =
          chain
            (fun edge ->
              let+ _ = def edge permute in
              ((), edge))
            (t_ 2)
        in
        ()))
    (t_ 2);
  [%expect {| chain step did not move upstream of t2 |}]

(* ---- scanning ------------------------------------------------------------ *)

let%expect_test "scan anchors on every node output, including a second result" =
  (* max_pool2d_with_indices produces two edges; a pattern anchored on the
     indices has to be reachable, so both are anchors. *)
  let g = Graph_fixtures.multi_output () in
  let v = view_of g in
  let found =
    Pattern.scan
      (fun anchor ->
        Pattern.(
          let+ _ =
            def anchor (function Discard { x } -> Some x | _ -> None)
          in
          anchor))
      v
  in
  Format.printf "discard anchors: %a@." (ids Tensor_id.pp) (List.map fst found);
  (* Matching the pool from ANY of its outputs finds it once: both edges are
     anchors, but the second match overlaps the first and is dropped. *)
  let pooled =
    Pattern.scan
      (fun anchor ->
        Pattern.(
          let+ _ =
            def anchor (function
              | Max_pool2d_with_indices p -> Some p
              | _ -> None)
          in
          anchor))
      v
  in
  Format.printf "pool anchors: %a@." (ids Tensor_id.pp) (List.map fst pooled);
  (* Requiring the anchor to BE the indices edge shows the second output really
     is offered as an anchor, rather than the values edge always winning. *)
  let indices =
    Pattern.scan
      (fun anchor ->
        Pattern.(
          let* _, n =
            def anchor (function
              | Max_pool2d_with_indices p -> Some p
              | _ -> None)
          in
          let+ () = guard (List.nth_opt n.Node.outputs 1 = Some anchor) in
          anchor))
      v
  in
  Format.printf "indices anchors: %a@." (ids Tensor_id.pp)
    (List.map fst indices);
  [%expect
    {|
    discard anchors: []
    pool anchors: [t1]
    indices anchors: [t2] |}]

let%expect_test "scan is greedy: overlapping matches are dropped, not retried" =
  (* Anchored at either permute, the two-node pattern matches only at the
     second; a one-node pattern matches both, but the results must stay
     disjoint so they can all go into one recipe. *)
  let g = Graph_fixtures.permute_sequence () in
  let v = view_of g in
  let pairs =
    Pattern.scan
      (fun anchor ->
        Pattern.(
          let* p, outer = def anchor permute in
          let+ _, inner = def p.x permute in
          (outer.Node.id, inner.Node.id)))
      v
  in
  List.iter
    (fun ((a, b), (r : Region.t)) ->
      Format.printf "pair %a/%a region %a@." Node_id.pp a Node_id.pp b
        (ids Node_id.pp)
        (Node_id.Set.elements r.nodes))
    pairs;
  let singles =
    Pattern.scan
      (fun anchor ->
        Pattern.(
          let+ _, n = def anchor permute in
          n.Node.id))
      v
  in
  Format.printf "singles: %a@." (ids Node_id.pp) (List.map fst singles);
  [%expect {|
    pair n1/n0 region [n0, n1]
    singles: [n0, n1] |}]

(* ---- exclusive vs. shared claims ------------------------------------------ *)

(* Two anchors that both reach the diamond's [relu] node (n0, producing t1):
   directly, from t1 itself, and via [add]'s [a] operand from t3. Neither
   claims it through [def] (which is always exclusive) — [peek] finds the
   node, and the caller picks how to claim it, so the SAME pair of matches
   can be run under all four combinations of [claim]/[claim_shared]. *)
let scan_pair ~kind1 ~kind2 v =
  Pattern.scan
    (fun anchor ->
      Pattern.(
        if Tensor_id.equal anchor (t_ 1) then
          let* _, n = peek anchor relu in
          let+ () = kind1 n in
          `Via_t1
        else if Tensor_id.equal anchor (t_ 3) then
          let* a, _ = def anchor add in
          let* _, n = peek a.a relu in
          let+ () = kind2 n in
          `Via_t3
        else fail Rejected))
    v

let pp_found found =
  List.iter
    (fun (value, (r : Region.t)) ->
      let label = match value with `Via_t1 -> "t1" | `Via_t3 -> "t3" in
      Format.printf "%s: %a@." label (ids Node_id.pp)
        (Node_id.Set.elements r.nodes))
    found

let%expect_test "two shared claims on the same node do not conflict" =
  (* Neither match removes or rewrites n0, so both are accepted — and their
     RETURNED regions overlap on n0, which a flat node-disjoint rule (the
     pre-[claim_shared] behavior) could never allow. *)
  let g = Graph_fixtures.diamond () in
  pp_found
    (scan_pair ~kind1:Pattern.claim_shared ~kind2:Pattern.claim_shared
       (view_of g));
  [%expect {|
    t1: [n0]
    t3: [n0, n2] |}]

let%expect_test "exclusive first, shared second: the second is dropped" =
  let g = Graph_fixtures.diamond () in
  pp_found
    (scan_pair ~kind1:Pattern.claim ~kind2:Pattern.claim_shared (view_of g));
  [%expect {| t1: [n0] |}]

let%expect_test "shared first, exclusive second: the second is dropped too" =
  (* The mirror image of the case above — the conflict rule has to be
     symmetric in which side is exclusive, not just in which side scan
     reaches first. *)
  let g = Graph_fixtures.diamond () in
  pp_found
    (scan_pair ~kind1:Pattern.claim_shared ~kind2:Pattern.claim (view_of g));
  [%expect {| t1: [n0] |}]

let%expect_test "two exclusive claims on the same node conflict" =
  (* The complete matrix's remaining corner: both exclusive is the ordinary,
     pre-[claim_shared] node-disjoint rule. *)
  let g = Graph_fixtures.diamond () in
  pp_found (scan_pair ~kind1:Pattern.claim ~kind2:Pattern.claim (view_of g));
  [%expect {| t1: [n0] |}]
