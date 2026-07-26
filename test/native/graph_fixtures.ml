(* Shared graphs for the transformation tests. There was no fixture library
   before this — [s]/[s1c]/[conv_axis] are copy-pasted across five test files —
   so new tests build on these and the existing files are left alone.

   Every fixture is a plain [Graph_ir.graph] built with [Graph_builder], named
   after the shape it exercises rather than the pass that will consume it. *)

(* ---- shape helpers ------------------------------------------------------- *)

let s n t d h w c = Vec6.shape ~n ~t ~d ~h ~w ~c
let s1c n = s 1 1 1 1 1 n
let nhwc ~h ~w ~c = s 1 1 1 h w c

let conv_axis ~kernel ~stride ~pad : Conv.Conv2d.axis_window =
  {
    kernel = Dim.extent kernel;
    stride = Op_config.Pos.of_int stride;
    pad_before = Op_config.Nonneg.of_int pad;
    pad_after = Op_config.Nonneg.of_int pad;
    dilation = Op_config.Pos.of_int 1;
  }

let conv_params ~in_channels : Conv.Conv2d.params =
  {
    h = conv_axis ~kernel:2 ~stride:1 ~pad:0;
    w = conv_axis ~kernel:2 ~stride:1 ~pad:0;
    in_channels = Dim.extent in_channels;
    groups = Op_config.Pos.of_int 1;
  }

let bn_params : Norm.BatchNorm.params = { channel = Axis.C; eps = 1e-5 }

(* Conv weights live as [Cout,1,1,Kh,Kw,Cin]: Cout on N, Cin on C. *)
let weight_shape ~out_channels ~in_channels = s out_channels 1 1 2 2 in_channels

(* ---- permutations -------------------------------------------------------- *)

let identity_perm = Axis.[ (N, N); (T, T); (D, D); (H, H); (W, W); (C, C) ]
let swap_hw = Axis.[ (N, N); (T, T); (D, D); (H, W); (W, H); (C, C) ]
let swap_wc = Axis.[ (N, N); (T, T); (D, D); (H, H); (W, C); (C, W) ]

(* A 3-cycle and its inverse, so a two-permute chain composes to the identity. *)
let rotate_hwc = Axis.[ (N, N); (T, T); (D, D); (H, W); (W, C); (C, H) ]
let unrotate_hwc = Axis.[ (N, N); (T, T); (D, D); (H, C); (W, H); (C, W) ]

(* ---- fixtures ------------------------------------------------------------ *)

let build name m =
  match Graph_builder.build ~name ~outputs:(fun o -> [ o ]) m with
  | Ok g -> g
  | Error e ->
      invalid_arg
        (Format.asprintf "fixture %s: %a" name Graph_builder.pp_error
           e.Core.Error.kind)

(* [Graph_builder] hardcodes every op's OWN output to f32 (see
   .ai/native_transform_design.md §14), so two [permute]s reading the same
   edge always agree on format — there is no way to build a
   precision-INCOMPATIBLE alternate-layout consumer through the builder
   alone. Patching one tensor's format after the fact is what lets a fixture
   exercise the incompatible branch at all. *)
let with_fmt id fmt (g : Graph_ir.graph) =
  {
    g with
    Graph_ir.Graph.tensors =
      Tensor_id.Map.update id
        (Option.map (fun (sg : Tensor_sig.t) -> { sg with Tensor_sig.fmt }))
        g.Graph_ir.Graph.tensors;
  }

(* conv -> batch_norm -> relu, with every parameter a constant. The shape
   batch-norm folding matches. *)
let chain () =
  build "chain"
    Graph_builder.(
      let* x = input ~shape:(nhwc ~h:4 ~w:4 ~c:2) () in
      let* w =
        constant ~shape:(weight_shape ~out_channels:3 ~in_channels:2) ()
      in
      let* bias = constant ~shape:(s1c 3) () in
      let* gamma = constant ~shape:(s1c 3) () in
      let* beta = constant ~shape:(s1c 3) () in
      let* mean = constant ~shape:(s1c 3) () in
      let* var = constant ~shape:(s1c 3) () in
      let* y = conv2d (conv_params ~in_channels:2) ~x ~weight:w ~bias () in
      let* n =
        batch_norm bn_params ~x:y ~weight:gamma ~bias:beta ~running_mean:mean
          ~running_var:var ()
      in
      relu n)

(* One edge feeding two consumers that rejoin: the fan-out and union a pattern
   has to cope with, and the shape where an interior edge is NOT single-use. *)
let diamond () =
  build "diamond"
    Graph_builder.(
      let* x = input ~shape:(s1c 4) () in
      let* left = relu x in
      let* right = mul x x in
      add left right)

(* A skip connection: the add's two operands come from different depths. *)
let residual () =
  build "residual"
    Graph_builder.(
      let* x = input ~shape:(s1c 4) () in
      let* a = relu x in
      let* b = relu a in
      add x b)

(* A permute that does nothing — the trivial-trim case. *)
let permute_noop () =
  build "permute_noop"
    Graph_builder.(
      let* x = input ~shape:(nhwc ~h:2 ~w:3 ~c:4) () in
      let* p = permute identity_perm x in
      relu p)

(* Two permutes composing to the identity: trimming has to look at the whole
   chain, since neither node is a no-op alone. *)
let permute_sequence () =
  build "permute_sequence"
    Graph_builder.(
      let* x = input ~shape:(nhwc ~h:2 ~w:3 ~c:4) () in
      let* a = permute rotate_hwc x in
      let* b = permute unrotate_hwc a in
      relu b)

(* Three identity permutes in a row. A pass that only fires on a permute reading
   a graph input can remove exactly one per sweep, so this is what makes a fixed
   point observably different from a single sweep. *)
let permute_identity_chain () =
  build "permute_identity_chain"
    Graph_builder.(
      let* x = input ~shape:(nhwc ~h:2 ~w:3 ~c:4) () in
      let* a = permute identity_perm x in
      let* b = permute identity_perm a in
      let* c = permute identity_perm b in
      relu c)

(* A cancelling pair followed by a permute that does not cancel: only the run
   ending at the middle edge is trimmable, so this pins that a match is looked
   for at every edge rather than only at the end of a chain. *)
let permute_partial_cancel () =
  build "permute_partial_cancel"
    Graph_builder.(
      let* x = input ~shape:(nhwc ~h:2 ~w:3 ~c:4) () in
      let* a = permute rotate_hwc x in
      let* b = permute unrotate_hwc a in
      let* c = permute swap_hw b in
      relu c)

(* The same cancelling pair, but the intermediate edge has a second consumer.
   Trimming would leave that consumer without a producer, since the intermediate
   value is a real rearrangement rather than a copy of the input. *)
let permute_shared () =
  build "permute_shared"
    Graph_builder.(
      let* x = input ~shape:(nhwc ~h:2 ~w:3 ~c:4) () in
      let* a = permute rotate_hwc x in
      let* b = permute unrotate_hwc a in
      let* other = relu a in
      let* () = discard other in
      relu b)

(* Two permutes that compose to something else, which chaining should fuse into
   one node rather than remove. *)
let permute_pair () =
  build "permute_pair"
    Graph_builder.(
      let* x = input ~shape:(nhwc ~h:2 ~w:3 ~c:4) () in
      let* a = permute swap_hw x in
      let* b = permute swap_wc a in
      relu b)

(* A second [build] for fixtures declaring more than one graph output — every
   other fixture funnels its result through the shared [build]'s
   [~outputs:(fun o -> [ o ])], which only fits a single final edge. *)
let build2 name m =
  match Graph_builder.build ~name ~outputs:(fun (a, b) -> [ a; b ]) m with
  | Ok g -> g
  | Error e ->
      invalid_arg
        (Format.asprintf "fixture %s: %a" name Graph_builder.pp_error
           e.Core.Error.kind)

(* For a fixture whose final value IS already the list of output edges. *)
let buildn name m =
  match Graph_builder.build ~name ~outputs:Fun.id m with
  | Ok g -> g
  | Error e ->
      invalid_arg
        (Format.asprintf "fixture %s: %a" name Graph_builder.pp_error
           e.Core.Error.kind)

(* An elementwise op reading a permuted operand, with an inverse permute right
   after it: [Sink_permute]'s minimal case. One sweep (through [relu]) already
   makes the two permutes adjacent. *)
let sink_permute_unary () =
  build "sink_permute_unary"
    Graph_builder.(
      let* x = input ~shape:(nhwc ~h:2 ~w:3 ~c:4) () in
      let* a = permute rotate_hwc x in
      let* r = relu a in
      permute unrotate_hwc r)

(* Like [sink_permute_unary], but the two permutes are NOT inverses: after
   sinking, [Chain_permute] fuses the adjacent pair into one composite permute
   rather than [Trim_permute] cancelling it away. Pins the end-to-end
   interaction — sinking exists to feed chain/trim adjacent permutes, and this
   is the case where fusing, not trimming, is what consumes that adjacency. *)
let sink_permute_fuse () =
  build "sink_permute_fuse"
    Graph_builder.(
      let* x = input ~shape:(nhwc ~h:2 ~w:3 ~c:4) () in
      let* a = permute rotate_hwc x in
      let* r = relu a in
      permute swap_hw r)

(* The real ResNet residual-add shape: two permuted branches feeding [add],
   then [relu], then the inverse permute. Needs TWO [Sink_permute] sweeps
   (through [add], which exposes a match at [relu]) before the two permutes
   are adjacent, so callers must drive this with [Pass.fixpoint
   Sink_permute.pass] rather than a single application. *)
let sink_permute_binary () =
  build "sink_permute_binary"
    Graph_builder.(
      let* x = input ~shape:(nhwc ~h:2 ~w:3 ~c:4) () in
      let* y = input ~shape:(nhwc ~h:2 ~w:3 ~c:4) () in
      let* a = permute rotate_hwc x in
      let* b = permute rotate_hwc y in
      let* s = add a b in
      let* r = relu s in
      permute unrotate_hwc r)

(* Like [sink_permute_binary], but [y]'s pre-permute shape has a broadcast
   axis against [x] ([H=1 W=1] vs [H=2 W=3]). After the shared [rotate_hwc]
   the two operands' extent-1 axes land in different physical slots, which is
   the case the commutation argument has to hold for broadcasting, not just
   equal shapes — the numeric-equivalence test runs against this fixture. *)
let sink_permute_broadcast () =
  build "sink_permute_broadcast"
    Graph_builder.(
      let* x = input ~shape:(nhwc ~h:2 ~w:3 ~c:4) () in
      let* y = input ~shape:(nhwc ~h:1 ~w:1 ~c:4) () in
      let* a = permute rotate_hwc x in
      let* b = permute rotate_hwc y in
      let* s = add a b in
      let* r = relu s in
      permute unrotate_hwc r)

(* [add]'s two branches carry DIFFERENT perms, so [Sink_permute] must decline
   outright rather than rewrite around one of them. A cube input keeps both
   permutations at the same shape, so the fixture is buildable regardless of
   perm — the only thing left to make the pass decline is the mismatch
   itself. *)
let sink_permute_mismatch () =
  build "sink_permute_mismatch"
    Graph_builder.(
      let* x = input ~shape:(s 1 1 1 2 2 2) () in
      let* y = input ~shape:(s 1 1 1 2 2 2) () in
      let* a = permute rotate_hwc x in
      let* b = permute swap_hw y in
      add a b)

(* The permute feeding the anchor [relu] has a SECOND consumer (another
   [relu]), so it is not interior: sinking would delete a node the other
   consumer still needs. Mirrors [permute_shared]'s non-interior shape, but
   for [Sink_permute] rather than the trim/chain pair. *)
let sink_permute_shared () =
  build "sink_permute_shared"
    Graph_builder.(
      let* x = input ~shape:(nhwc ~h:2 ~w:3 ~c:4) () in
      let* a = permute rotate_hwc x in
      let* other = relu a in
      let* () = discard other in
      relu a)

(* The permute feeding the anchor [relu] has exactly one CONSUMER, but is
   itself also a graph output — the other half of [Pattern.interior], isolated
   from the shared-consumer case above. Sinking would leave that graph output
   without a producer. *)
let sink_permute_output () =
  build2 "sink_permute_output"
    Graph_builder.(
      let* x = input ~shape:(nhwc ~h:2 ~w:3 ~c:4) () in
      let* a = permute rotate_hwc x in
      let* r = relu a in
      return (a, r))

(* One of each op [Sink_permute] accepts, each fed its own singly-consumed
   permuted operand(s) so every branch is independently sinkable. Cheap
   insurance that the allowlist match arm actually fires for all six
   constructors, despite it looking exhaustive on inspection. *)
let sink_permute_allowlist () =
  build "sink_permute_allowlist"
    Graph_builder.(
      let* x = input ~shape:(nhwc ~h:2 ~w:3 ~c:4) () in
      let* y = input ~shape:(nhwc ~h:2 ~w:3 ~c:4) () in
      let pair () =
        let* a = permute rotate_hwc x in
        let+ b = permute rotate_hwc y in
        (a, b)
      in
      let* single_relu = permute rotate_hwc x in
      let* r_relu = relu single_relu in
      let* single_sqrt = permute rotate_hwc x in
      let* r_sqrt = sqrt single_sqrt in
      let* a1, b1 = pair () in
      let* r_add = add a1 b1 in
      let* a2, b2 = pair () in
      let* r_sub = sub a2 b2 in
      let* a3, b3 = pair () in
      let* r_mul = mul a3 b3 in
      let* a4, b4 = pair () in
      let* r_div = div a4 b4 in
      let* s1 = add r_add r_sub in
      let* s2 = add r_mul r_div in
      let* s3 = add r_relu r_sqrt in
      let* s4 = add s1 s2 in
      add s4 s3)

(* [Reuse_permute]'s motivating shape: [add(P(x_pre), b)] where [b] already has
   an existing inverse-layout consumer [qb = permute unrotate_hwc b] — reused
   directly rather than re-permuting [b]. [qb] is fed to a second consumer so it
   is not itself dead code. *)
let reuse_permute_basic () =
  build "reuse_permute_basic"
    Graph_builder.(
      let* x_pre = input ~shape:(nhwc ~h:2 ~w:3 ~c:4) () in
      let* b = input ~shape:(nhwc ~h:3 ~w:4 ~c:2) () in
      let* pa = permute rotate_hwc x_pre in
      let* qb = permute unrotate_hwc b in
      let* () = discard qb in
      add pa b)

(* [Found in review, P1]: two DISJOINT matches in one sweep, competing over
   one producer. [add1 = add(P(a), b)] reuses [qb = Q(b)]'s output without
   removing it; [add2 = add(qb, c)] would UNWRAP that very [qb] (its operand
   IS [qb]'s output, and [qb] is interior at scan time — its only declared
   consumer is [add2]). Without reserving the reused node, both could be
   accepted in the SAME sweep, and the merged recipe would remove [qb] while
   [add1]'s rebuilt op still read its output. [claim]ing [qb] on the reuse
   side is what makes `Pattern.scan`'s ordinary greedy-disjoint rule catch
   the overlap instead: only one of the two is accepted per sweep, and the
   other is safely retried — not corrupted — on the next. *)
let reuse_permute_competing_matches () =
  build2 "reuse_permute_competing_matches"
    Graph_builder.(
      let* a = input ~shape:(nhwc ~h:2 ~w:3 ~c:4) () in
      let* b = input ~shape:(nhwc ~h:3 ~w:4 ~c:2) () in
      let* c = input ~shape:(nhwc ~h:2 ~w:3 ~c:4) () in
      let* pa = permute rotate_hwc a in
      let* qb = permute unrotate_hwc b in
      let* pc = permute rotate_hwc c in
      let* () = discard pc in
      let* add1 = add pa b in
      let* add2 = add qb c in
      return (add1, add2))

(* [Found in review, P1]: 16 INDEPENDENT [add(P(x_i), b)] anchors all reusing
   the SAME [Q(b)] — every one of them merely READS [qb]'s output, none
   removes it, so all 16 are safe to accept in ONE sweep. Before
   [Pattern.claim_shared] existed, every reuse used the exclusive [claim],
   so [Pattern.scan] serialized these to one acceptance per sweep: 16
   sweeps needed, exactly exhausting [Pass.fixpoint]'s default 16-unit fuel
   with `Not_converged`, on a graph that was never actually stuck. *)
let reuse_permute_wide_fanout () =
  let width = 16 in
  buildn "reuse_permute_wide_fanout"
    Graph_builder.(
      let* b = input ~shape:(nhwc ~h:3 ~w:4 ~c:2) () in
      let* qb = permute unrotate_hwc b in
      let* () = discard qb in
      let rec go i acc =
        if i = width then return (List.rev acc)
        else
          let* xi = input ~shape:(nhwc ~h:2 ~w:3 ~c:4) () in
          let* pai = permute rotate_hwc xi in
          let* addi = add pai b in
          go (i + 1) (addi :: acc)
      in
      go 0 [])

(* Like [reuse_permute_basic], but [b] has no alternate-layout consumer at all:
   there is nothing to reuse, so the pass has to decline outright rather than
   speculate a new relayout for [b]. *)
let reuse_permute_missing_alternate () =
  build "reuse_permute_missing_alternate"
    Graph_builder.(
      let* x_pre = input ~shape:(nhwc ~h:2 ~w:3 ~c:4) () in
      let* b = input ~shape:(nhwc ~h:3 ~w:4 ~c:2) () in
      let* pa = permute rotate_hwc x_pre in
      add pa b)

(* [b] has an existing alternate-layout consumer, but its perm is NOT the
   inverse of the candidate [P] — [swap_hw], not [unrotate_hwc]. The candidate
   still has to be declined rather than treated as reusable. *)
let reuse_permute_wrong_alternate () =
  build "reuse_permute_wrong_alternate"
    Graph_builder.(
      let* x_pre = input ~shape:(nhwc ~h:2 ~w:3 ~c:4) () in
      let* b = input ~shape:(nhwc ~h:3 ~w:4 ~c:2) () in
      let* pa = permute rotate_hwc x_pre in
      let* qb = permute swap_hw b in
      let* () = discard qb in
      add pa b)

(* Two existing inverse consumers of [b]: the FIRST ([q_bad]) is patched to an
   incompatible format, so the pass must fall through to the SECOND ([q_ok])
   rather than declining the whole operand outright. Regression for the
   "incompatible candidate hides a safe later one" failure mode. *)
let reuse_permute_backtrack_candidate () =
  let g =
    build "reuse_permute_backtrack_candidate"
      Graph_builder.(
        let* x_pre = input ~shape:(nhwc ~h:2 ~w:3 ~c:4) () in
        let* b = input ~shape:(nhwc ~h:3 ~w:4 ~c:2) () in
        let* q_bad = permute unrotate_hwc b in
        let* () = discard q_bad in
        let* q_ok = permute unrotate_hwc b in
        let* () = discard q_ok in
        let* pa = permute rotate_hwc x_pre in
        add pa b)
  in
  with_fmt (Tensor_id.of_int 2) (Payload.Fmt Payload.F16) g

(* [Found in review, P1]: a self-inverse permutation ([swap_hw] on a square
   [H=W] shape) reused AND unwrapped in the SAME match. [pb]'s output is the
   pass's only candidate for BOTH roles — unwrap (it is [pb]'s own producer)
   and reuse (it is also its own inverse consumer) — so without disjointness
   tracking the pass would remove [pb] while the rebuilt [add] still read its
   output, corrupting the graph (`Rewrite.apply` failing with "operand t1 has
   no definition"). Declining outright is the only correct outcome: there is
   no OTHER node either operand could resolve through instead. *)
let reuse_permute_self_inverse () =
  build "reuse_permute_self_inverse"
    Graph_builder.(
      let* b = input ~shape:(nhwc ~h:2 ~w:2 ~c:4) () in
      let* pb = permute swap_hw b in
      add pb b)

(* [Sub] is not commutative, so getting the rebuilt op's operand order wrong
   would silently negate the result. Here the FIRST operand is the reused one
   and the SECOND is the unwrapped one — the mirror image of
   [reuse_permute_basic], where unwrap comes first — so a pass that always
   assumed "operand 0 unwraps" would get this one wrong. *)
let reuse_permute_sub_order () =
  build "reuse_permute_sub_order"
    Graph_builder.(
      let* x_pre = input ~shape:(nhwc ~h:2 ~w:3 ~c:4) () in
      let* a = input ~shape:(nhwc ~h:3 ~w:4 ~c:2) () in
      let* qa = permute unrotate_hwc a in
      let* () = discard qa in
      let* pb = permute rotate_hwc x_pre in
      sub a pb)

(* Same order-preservation concern for [Div]. *)
let reuse_permute_div_order () =
  build "reuse_permute_div_order"
    Graph_builder.(
      let* x_pre = input ~shape:(nhwc ~h:2 ~w:3 ~c:4) () in
      let* a = input ~shape:(nhwc ~h:3 ~w:4 ~c:2) () in
      let* qa = permute unrotate_hwc a in
      let* () = discard qa in
      let* pb = permute rotate_hwc x_pre in
      div a pb)

(* [Bypass_permute]'s minimal case: one [P] feeding exactly one inverse [Q],
   with no other use of either edge — both nodes disappear. *)
let bypass_permute_pair () =
  build "bypass_permute_pair"
    Graph_builder.(
      let* x = input ~shape:(nhwc ~h:2 ~w:3 ~c:4) () in
      let* y = permute rotate_hwc x in
      let* z = permute unrotate_hwc y in
      relu z)

(* One [P] feeding TWO separate inverse [Q] consumers, and nothing else — the
   fan-out [Trim_permute]'s single-chain walk cannot express. Both [Q]s and
   [P] itself all disappear, with both outputs tied straight to [x]. *)
let bypass_permute_fanout () =
  build2 "bypass_permute_fanout"
    Graph_builder.(
      let* x = input ~shape:(nhwc ~h:2 ~w:3 ~c:4) () in
      let* y = permute rotate_hwc x in
      let* z1 = permute unrotate_hwc y in
      let* z2 = permute unrotate_hwc y in
      let* r1 = relu z1 in
      let* r2 = relu z2 in
      return (r1, r2))

(* [P]'s output has a second, non-inverse consumer: the inverse [Q] still goes,
   but [P] has to stay for the other consumer's sake. *)
let bypass_permute_shared () =
  build2 "bypass_permute_shared"
    Graph_builder.(
      let* x = input ~shape:(nhwc ~h:2 ~w:3 ~c:4) () in
      let* y = permute rotate_hwc x in
      let* z = permute unrotate_hwc y in
      let* other = relu y in
      let* rz = relu z in
      return (rz, other))

(* Two inverse consumers of [y], one patched to an incompatible format: the
   compatible one ([q_ok]) is bypassed, but the incompatible one ([q_bad])
   is left as an ordinary (non-bypassed) consumer, which keeps [P] alive —
   filtering, not an all-or-nothing guard over every inverse consumer found. *)
let bypass_permute_mixed_compatibility () =
  let g =
    build2 "bypass_permute_mixed_compatibility"
      Graph_builder.(
        let* x = input ~shape:(nhwc ~h:2 ~w:3 ~c:4) () in
        let* y = permute rotate_hwc x in
        let* q_ok = permute unrotate_hwc y in
        let* q_bad = permute unrotate_hwc y in
        let* r_ok = relu q_ok in
        let* r_bad = relu q_bad in
        return (r_ok, r_bad))
  in
  with_fmt (Tensor_id.of_int 3) (Payload.Fmt Payload.F16) g

(* [P]'s output is itself a graph output, even though its only OTHER consumer
   is an inverse [Q]: [P] has to stay for the output contract. *)
let bypass_permute_output () =
  build2 "bypass_permute_output"
    Graph_builder.(
      let* x = input ~shape:(nhwc ~h:2 ~w:3 ~c:4) () in
      let* y = permute rotate_hwc x in
      let* z = permute unrotate_hwc y in
      let* rz = relu z in
      return (y, rz))

(* Two chained opportunities that need the OUTER fixed point over the whole
   relayout family, not just one pass through each. [y]'s producer [P] is
   shared between an inverse [Q] — fed straight to [Discard], so [Sink_permute]
   can never reach it: that pass only ever anchors on an elementwise op's OWN
   output, and a discarded permute is not one — and a plain [Relu] ([P] itself
   is not [interior], so [Sink_permute] cannot move it through that [Relu]
   either, yet). Bypassing [Q] is [Bypass_permute]'s job alone here, and it
   runs AFTER both [Sink_permute] stages in the sequence, so making [y]
   interior comes too late to be exploited in the SAME application: the
   [Relu] is still reading the permuted edge afterwards. Only a SECOND
   application — i.e. wrapping the whole sequence in one more
   [Pass.fixpoint] — sinks it. *)
let bypass_unlocks_sink () =
  build "bypass_unlocks_sink"
    Graph_builder.(
      let* z = input ~shape:(nhwc ~h:2 ~w:3 ~c:4) () in
      let* y = permute rotate_hwc z in
      let* q = permute unrotate_hwc y in
      let* () = discard q in
      relu y)

(* A reshape that only relabels axes — the non-unit extents keep their order —
   so it is expressible as a permute. *)
let reshape_relabel () =
  build "reshape_relabel"
    Graph_builder.(
      let* x = input ~shape:(s 1 1 1 1 1 6) () in
      let* r = reshape { Reshape.Reshape.shape = s 1 1 1 6 1 1 } x in
      relu r)

(* A reshape that genuinely mixes extents: H*W collapses onto C, which no axis
   permutation can express. *)
let reshape_flatten () =
  build "reshape_flatten"
    Graph_builder.(
      let* x = input ~shape:(s 1 1 1 2 3 1) () in
      let* r = reshape { Reshape.Reshape.shape = s 1 1 1 1 1 6 } x in
      relu r)

(* THE motivating case: a constant weight behind a permute, re-permuted on every
   inference until constant folding hoists it to load time. The perm is a real
   rearrangement — with the identity here the weight would be trimmed rather than
   folded, and the fixture would exercise the wrong pass. *)
let const_permute () =
  build "const_permute"
    Graph_builder.(
      let* x = input ~shape:(nhwc ~h:3 ~w:3 ~c:2) () in
      let* w = constant ~shape:(s 3 1 1 2 2 2) () in
      let* wp = permute swap_hw w in
      conv2d (conv_params ~in_channels:2) ~x ~weight:wp ())

(* A constant behind a multi-output op: foldable in principle, out of scope for
   [Fold_const], which binds one output per recipe. *)
let const_pool () =
  build "const_pool"
    Graph_builder.(
      let* c = constant ~shape:(nhwc ~h:4 ~w:4 ~c:2) () in
      let params : Pool.MaxPool2dWithIndices.params =
        {
          kernel = { h = Dim.extent 2; w = Dim.extent 2 };
          stride = { h = Op_config.Pos.of_int 2; w = Op_config.Pos.of_int 2 };
          pad = { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
        }
      in
      let* values, indices = max_pool2d_with_indices params c in
      let* () = discard indices in
      relu values)

(* A multi-node all-constant sub-DAG, for folding under a fixed point. *)
let const_arith () =
  build "const_arith"
    Graph_builder.(
      let* x = input ~shape:(s1c 3) () in
      let* a = constant ~shape:(s1c 3) () in
      let* b = constant ~shape:(s1c 3) () in
      let* c = constant ~shape:(s1c 3) () in
      let* ab = mul a b in
      let* abc = mul ab c in
      add x abc)

(* A multi-output op with its second result routed to a sink — the arity a
   matcher's anchoring rule and constant folding both have to handle. *)
let multi_output () =
  build "multi_output"
    Graph_builder.(
      let* x = input ~shape:(nhwc ~h:4 ~w:4 ~c:2) () in
      let params : Pool.MaxPool2dWithIndices.params =
        {
          kernel = { h = Dim.extent 2; w = Dim.extent 2 };
          stride = { h = Op_config.Pos.of_int 2; w = Op_config.Pos.of_int 2 };
          pad = { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
        }
      in
      let* values, indices = max_pool2d_with_indices params x in
      let* () = discard indices in
      relu values)

(* Two sibling groups with a rewrite opportunity spanning both: the placement
   case, where the replacement belongs in their common ancestor. *)
let grouped () =
  build "grouped"
    Graph_builder.(
      let* x = input ~shape:(nhwc ~h:2 ~w:3 ~c:4) () in
      let* a = group ~label:"first" (permute rotate_hwc x) in
      let* b = group ~label:"second" (permute unrotate_hwc a) in
      relu b)

let all =
  [
    ("bypass_permute_fanout", bypass_permute_fanout);
    ("bypass_permute_mixed_compatibility", bypass_permute_mixed_compatibility);
    ("bypass_permute_output", bypass_permute_output);
    ("bypass_permute_pair", bypass_permute_pair);
    ("bypass_permute_shared", bypass_permute_shared);
    ("bypass_unlocks_sink", bypass_unlocks_sink);
    ("chain", chain);
    ("const_arith", const_arith);
    ("const_permute", const_permute);
    ("const_pool", const_pool);
    ("diamond", diamond);
    ("grouped", grouped);
    ("multi_output", multi_output);
    ("permute_identity_chain", permute_identity_chain);
    ("permute_noop", permute_noop);
    ("permute_pair", permute_pair);
    ("permute_partial_cancel", permute_partial_cancel);
    ("permute_sequence", permute_sequence);
    ("permute_shared", permute_shared);
    ("reshape_flatten", reshape_flatten);
    ("reshape_relabel", reshape_relabel);
    ("residual", residual);
    ("reuse_permute_backtrack_candidate", reuse_permute_backtrack_candidate);
    ("reuse_permute_basic", reuse_permute_basic);
    ("reuse_permute_competing_matches", reuse_permute_competing_matches);
    ("reuse_permute_div_order", reuse_permute_div_order);
    ("reuse_permute_missing_alternate", reuse_permute_missing_alternate);
    ("reuse_permute_self_inverse", reuse_permute_self_inverse);
    ("reuse_permute_sub_order", reuse_permute_sub_order);
    ("reuse_permute_wide_fanout", reuse_permute_wide_fanout);
    ("reuse_permute_wrong_alternate", reuse_permute_wrong_alternate);
    ("sink_permute_allowlist", sink_permute_allowlist);
    ("sink_permute_binary", sink_permute_binary);
    ("sink_permute_broadcast", sink_permute_broadcast);
    ("sink_permute_fuse", sink_permute_fuse);
    ("sink_permute_mismatch", sink_permute_mismatch);
    ("sink_permute_output", sink_permute_output);
    ("sink_permute_shared", sink_permute_shared);
    ("sink_permute_unary", sink_permute_unary);
  ]
