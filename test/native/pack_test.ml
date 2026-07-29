(* Terminal id packing. See .ai/native_transform_design.md §9.

   The pipeline used throughout is the batch-norm fold followed by constant
   folding, because it produces the situation packing has to get right without
   any contrivance: it mints ids t10-t19, folding then kills most of them, and
   compacting the survivors lands them on numeric values that dead ids held. *)

open Graph_ir

let t_ n = Tensor_id.of_int n
let s1c = Graph_fixtures.s1c

let vec values =
  Tensor.materialize (s1c 3) (fun c -> values.(Dim.to_int (Vec6.get c Axis.C)))

let ramp shape =
  Tensor.materialize shape (fun c ->
      float_of_int
        ((Dim.to_int (Vec6.get c Axis.N) * 7)
        + (Dim.to_int (Vec6.get c Axis.H) * 3)
        + Dim.to_int (Vec6.get c Axis.W)
        + Dim.to_int (Vec6.get c Axis.C))
      /. 4.)

(* [Graph_fixtures.chain] is conv -> batch_norm -> relu with every parameter
   constant: t1 weight, t2 conv bias, t3 gamma, t4 beta, t5 mean, t6 var. *)
let chain_constants =
  [
    (t_ 1, ramp (Graph_fixtures.weight_shape ~out_channels:3 ~in_channels:2));
    (t_ 2, vec [| 1.; 2.; 3. |]);
    (t_ 3, vec [| 2.; 0.5; 1.5 |]);
    (t_ 4, vec [| -1.; 0.5; 2. |]);
    (t_ 5, vec [| 0.5; 1.; 1.5 |]);
    (t_ 6, vec [| 4.; 1.; 0.25 |]);
  ]

let folded = [ Fold_batch_norm.pass; Pass.fixpoint Fold_const.pass ]

let pp_payload_ids fmt constants =
  Fmt.brackets
    (Fmt.list ~sep:Fmt.comma Tensor_id.pp)
    fmt
    (List.map fst (Tensor_id.Map.bindings constants))

(* The state reached by a pipeline, handed to a callback rather than returned:
   its version is existential, so it cannot escape the branch that unpacks it,
   and the callback has to be explicitly polymorphic to receive it at all. *)
type sink = { consume : 'w. 'w Rewrite.t -> unit }

let piped g ~constants passes { consume } =
  match Rewrite.origin ~constants g with
  | Error e -> Format.printf "origin: %a@." Rewrite.pp_error e.Core.Error.kind
  | Ok (Rewrite.Origin state) -> (
      match Pass.run_all state passes with
      | Error e -> Format.printf "%a@." Pass.pp_error e.Core.Error.kind
      | Ok (Rewrite.Step (final, _)) -> consume final)

(* ---- what packing does ---------------------------------------------------- *)

let%expect_test "pack: origin ids stay put, post-origin ids compact" =
  (* The origin used t0-t9 and n0-n2, so the watermarks are t10/n3. Folding
     leaves the survivors at t15, t18, t19 and n11; packing brings them down to
     t10, t11, t12 and n3 — dense, and still disjoint from every origin id. *)
  piped (Graph_fixtures.chain ()) ~constants:chain_constants folded
    {
      consume =
        (fun folded_state ->
          Format.printf "@[<v 2>folded:@,%a@]@." Graph_ir.pp
            (Rewrite.graph folded_state);
          match Rewrite.pack folded_state with
          | Error e -> Format.printf "%a@." Rewrite.pp_error e.Core.Error.kind
          | Ok (Rewrite.Step (packed, map)) ->
              Format.printf "@[<v 2>packed:@,%a@]@." Graph_ir.pp
                (Rewrite.graph packed);
              Format.printf "@[<v 2>pack map:@,%a@]@." Graph_map.pp map);
    };
  [%expect
    {|
    folded:
      graph
      inputs:
        [t0 f32 [H=4 W=4 C=2] ->[n11],
         t15 f32 [N=3 T=1 D=1 H=2 W=2 C=2] ->[n11] constant,
         t18 f32 [C=3] ->[n11] constant]
      nodes:
        n11: [t19 f32 [H=3 W=3 C=3] ->[n2]] =
          conv2d
            x=t0
            weight=t15
            bias=t18
            params={h={kernel=2; stride=1; pad_before=0; pad_after=0; dilation=1};
                   w={kernel=2; stride=1; pad_before=0; pad_after=0; dilation=1};
                   in_channels=2;
                   groups=1}
        n2: [t9 f32 [H=3 W=3 C=3]] = relu x=t19 <-n11
      outputs: [t9 f32 [H=3 W=3 C=3] <-n2]
    packed:
      graph
      inputs:
        [t0 f32 [H=4 W=4 C=2] ->[n3],
         t10 f32 [N=3 T=1 D=1 H=2 W=2 C=2] ->[n3] constant,
         t11 f32 [C=3] ->[n3] constant]
      nodes:
        n3: [t12 f32 [H=3 W=3 C=3] ->[n2]] =
          conv2d
            x=t0
            weight=t10
            bias=t11
            params={h={kernel=2; stride=1; pad_before=0; pad_after=0; dilation=1};
                   w={kernel=2; stride=1; pad_before=0; pad_after=0; dilation=1};
                   in_channels=2;
                   groups=1}
        n2: [t9 f32 [H=3 W=3 C=3]] = relu x=t12 <-n3
      outputs: [t9 f32 [H=3 W=3 C=3] <-n2]
    pack map:
      values:
        {t15} -> {t10} identical
        {t18} -> {t11} identical
        {t19} -> {t12} identical
      nodes:
        {n11} -> {n3}
      provenance:
        none |}]

let%expect_test "pack: constant payloads move with their edges" =
  piped (Graph_fixtures.chain ()) ~constants:chain_constants folded
    {
      consume =
        (fun folded_state ->
          Format.printf "before: %a@." pp_payload_ids
            (Rewrite.constants folded_state);
          match Rewrite.pack folded_state with
          | Error e -> Format.printf "%a@." Rewrite.pp_error e.Core.Error.kind
          | Ok (Rewrite.Step (packed, _)) ->
              Format.printf "after:  %a@." pp_payload_ids
                (Rewrite.constants packed));
    };
  [%expect {|
    before: [t15, t18]
    after:  [t10, t11] |}]

let%expect_test "pack: a graph with nothing post-origin is left alone" =
  (* Trimming introduces no ids, so there is nothing to compact and the map is
     empty — packing is not an unconditional renumbering. *)
  piped
    (Graph_fixtures.permute_identity_chain ())
    ~constants:[]
    [ Pass.fixpoint Trim_permute.pass ]
    {
      consume =
        (fun trimmed ->
          match Rewrite.pack trimmed with
          | Error e -> Format.printf "%a@." Rewrite.pp_error e.Core.Error.kind
          | Ok (Rewrite.Step (packed, map)) ->
              Format.printf "@[<v 2>packed:@,%a@]@." Graph_ir.pp
                (Rewrite.graph packed);
              Format.printf "@[<v 2>pack map:@,%a@]@." Graph_map.pp map);
    };
  [%expect
    {|
    packed:
      graph
      inputs: [t0 f32 [H=2 W=3 C=4] ->[n3]]
      nodes:
        n3: [t4 f32 [H=2 W=3 C=4]] = relu x=t0
      outputs: [t4 f32 [H=2 W=3 C=4] <-n3]
    pack map:
      values:
        identity
      nodes:
        identity
      provenance:
        none |}]

let%expect_test "pack: packing twice changes nothing" =
  (* Idempotence is the check that the canonical order is genuinely canonical.
     A second pack re-enumerates the same structure, so it can only differ from
     the first if the enumeration depends on something other than the graph. *)
  piped (Graph_fixtures.chain ()) ~constants:chain_constants folded
    {
      consume =
        (fun folded_state ->
          match Rewrite.pack folded_state with
          | Error e -> Format.printf "%a@." Rewrite.pp_error e.Core.Error.kind
          | Ok (Rewrite.Step (once, _)) -> (
              match Rewrite.pack once with
              | Error e ->
                  Format.printf "%a@." Rewrite.pp_error e.Core.Error.kind
              | Ok (Rewrite.Step (twice, map)) ->
                  Format.printf "second pack moves nothing: %b@."
                    (Correspondence.is_empty (Graph_map.values map));
                  Format.printf "same graph: %b@."
                    (String.equal
                       (Format.asprintf "%a" Graph_ir.pp (Rewrite.graph once))
                       (Format.asprintf "%a" Graph_ir.pp (Rewrite.graph twice)))
              ));
    };
  [%expect {|
    second pack moves nothing: true
    same graph: true |}]

(* ---- composing through a pack --------------------------------------------- *)

let%expect_test "pack: the origin-to-packed map still resolves every origin id"
    =
  (* What the PT2 lens will walk. Every id the origin graph used has to have an
     answer in the composed map, whether that is a packed destination, an
     unchanged one, or nothing at all because the edge is gone. *)
  match Rewrite.origin ~constants:chain_constants (Graph_fixtures.chain ()) with
  | Error e -> Format.printf "origin: %a@." Rewrite.pp_error e.Core.Error.kind
  | Ok (Rewrite.Origin s0) ->
      (match Pass.run_all s0 folded with
      | Error e -> Format.printf "%a@." Pass.pp_error e.Core.Error.kind
      | Ok (Rewrite.Step (s1, m01)) -> (
          match Rewrite.pack s1 with
          | Error e -> Format.printf "%a@." Rewrite.pp_error e.Core.Error.kind
          | Ok (Rewrite.Step (_, m12)) ->
              let composed = Graph_map.compose m01 m12 in
              let snap = Rewrite.snapshot s0 in
              List.iter
                (fun id ->
                  let landed =
                    Correspondence.forward
                      (Graph_map.values composed)
                      (Option.get (Snapshot.edge snap id))
                  in
                  Format.printf "t%d -> %a@." (Tensor_id.to_int id)
                    (Fmt.braces (Fmt.list ~sep:Fmt.comma Tensor_id.pp))
                    (Tensor_id.Set.elements (Correspondence.raws landed)))
                (List.init 10 t_)));
      [%expect
        {|
    t0 -> {t0}
    t1 -> {}
    t2 -> {}
    t3 -> {}
    t4 -> {}
    t5 -> {}
    t6 -> {}
    t7 -> {}
    t8 -> {t12}
    t9 -> {t9} |}]

let%expect_test "pack: a dead id is not fused with the value packed onto it" =
  (* The case §3's identity-extension guard exists for, arising on its own
     rather than by construction: folding declares t10-t14 deleted, and packing
     then puts t15 ON t10. Extending the dead t10 across the pack map would fuse
     the two clusters and claim the dead edge is the packed one.

     Checked under both association orders, since the guard is a side condition
     on extension and this is exactly where it could break associativity. *)
  match Rewrite.origin ~constants:chain_constants (Graph_fixtures.chain ()) with
  | Error e -> Format.printf "origin: %a@." Rewrite.pp_error e.Core.Error.kind
  | Ok (Rewrite.Origin s0) ->
      (match Pass.run_all s0 [ Fold_batch_norm.pass ] with
      | Error e -> Format.printf "%a@." Pass.pp_error e.Core.Error.kind
      | Ok (Rewrite.Step (s1, m01)) -> (
          match Pass.run_all s1 [ Pass.fixpoint Fold_const.pass ] with
          | Error e -> Format.printf "%a@." Pass.pp_error e.Core.Error.kind
          | Ok (Rewrite.Step (s2, m12)) -> (
              Format.printf "@[<v 2>fold map (t10-t14 die here):@,%a@]@."
                Graph_map.pp m12;
              match Rewrite.pack s2 with
              | Error e ->
                  Format.printf "%a@." Rewrite.pp_error e.Core.Error.kind
              | Ok (Rewrite.Step (_, m23)) ->
                  Format.printf "@[<v 2>pack map (t15 lands on t10):@,%a@]@."
                    Graph_map.pp m23;
                  let left =
                    Graph_map.compose (Graph_map.compose m01 m12) m23
                  in
                  let right =
                    Graph_map.compose m01 (Graph_map.compose m12 m23)
                  in
                  Format.printf "@[<v 2>composed:@,%a@]@." Graph_map.pp left;
                  Format.printf "associative: %b@."
                    (String.equal
                       (Format.asprintf "%a" Graph_map.pp left)
                       (Format.asprintf "%a" Graph_map.pp right)))));
      [%expect
        {|
    fold map (t10-t14 die here):
      values:
        {t1} -> {} identical
        {t2} -> {} identical
        {t3} -> {} identical
        {t4} -> {} identical
        {t5} -> {} identical
        {t6} -> {} identical
        {t10} -> {} identical
        {t11} -> {} identical
        {t12} -> {} identical
        {t13} -> {} identical
        {t14} -> {} identical
        {t16} -> {} identical
        {t17} -> {} identical
      nodes:
        {n3} -> {}
        {n4} -> {}
        {n5} -> {}
        {n6} -> {}
        {n7} -> {}
        {n8} -> {}
        {n9} -> {}
        {n10} -> {}
      provenance:
        {t1, t3, t6, t10} -> t15
        {t2, t3, t4, t5, t6, t10} -> t18
    pack map (t15 lands on t10):
      values:
        {t15} -> {t10} identical
        {t18} -> {t11} identical
        {t19} -> {t12} identical
      nodes:
        {n11} -> {n3}
      provenance:
        none
    composed:
      values:
        {t1} -> {} identical
        {t2} -> {} identical
        {t3} -> {} identical
        {t4} -> {} identical
        {t5} -> {} identical
        {t6} -> {} identical
        {t7} -> {} identical
        {t8} -> {t12} equivalent
        {t9} -> {t9} equivalent
        {} -> {t10} identical
        {} -> {t11} identical
      nodes:
        {n0, n1} -> {n3}
      provenance:
        {t1, t3, t6} -> t10
        {t2, t3, t4, t5, t6} -> t11
    associative: true |}]
