(* The flagship end-to-end acceptance test for the sharing-aware ground-IR
   work's Milestone C (sharing-aware verification, see .ai/native_transform_verify.md
   and .ai/native_scan_design.md): proving a small scan-backed Region stage
   through [Map_verify]'s real cross-graph machinery
   -- grounding, frontier expansion/crossing, projection and structural
   comparison -- not just against a direct-evaluation oracle the way
   region_scan_construction_test.ml / ground_eval_scan_at_test.ml already do.

   [Lstm] is the only scan-backed op in the Graph_ir catalog, so it is the
   "small scan-backed Region stage" under test here, at the smallest shape
   that still exercises a genuine multi-step recurrence (hidden_size=1,
   input_size=1, seq=2, single layer/direction, with bias).

   The "independent unrolled Pixel construction" the plan's own wording
   suggests is NOT used: matching bit-for-bit against a hand-unrolled
   construction built from ordinary elementwise ops would need that
   construction's [Round] placement (one per op) to agree with the fused
   Region program's (one per stage) exactly, which no amount of correct
   scan-grounding logic can arrange -- doing so would mean either changing
   how LSTM lowers or adding a bespoke Graph_ir op whose only job is to dodge
   that mismatch, and either is exactly the kind of new production surface
   CLAUDE.md's "surgical changes" asks this step not to add.

   Instead, two SEPARATELY built LSTM graphs (distinct Tensor_ids throughout,
   never sharing an arena) stand in for "independent": the correspondence map
   below asserts nothing from graph identity, only from the explicit
   Correspondence pairs, so a structural proof here is still a proof that
   Ground_eval's scan grounding, [expand]'s frontier crossing and
   [Ground_expr.project]/[equal] agree for a genuine two-step recurrence
   read through the full obligation machinery -- the one piece of Milestone C
   no existing test exercised. The mutation test below is what makes this a
   proof of the MACHINERY rather than of a tautology: swapping which initial
   state feeds which gate is a real functional change hidden behind an
   unchanged shape and an unchanged correspondence map, and it must refute. *)

open Graph_ir
open Verify_fixtures

let k = 1 (* hidden_size *)
let isz = 1 (* input width *)
let seq = 2 (* sequence length *)
let batch = 1
let mat_shape ~rows ~cols = Vec6.shape ~n:rows ~t:1 ~d:1 ~h:1 ~w:1 ~c:cols
let vec_shape ~n = Vec6.shape ~n ~t:1 ~d:1 ~h:1 ~w:1 ~c:1
let state_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:batch ~c:k
let seq_shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:seq ~w:batch ~c:isz

let params : Lstm.Lstm.params =
  { hidden_size = k; input_size = isz; batch_first = false }

(* [swap_state] mutates the graph WITHOUT changing any shape or declaration
   order: [h0]/[c0] are still the 6th/7th declared inputs either way, so the
   positional correspondence built below stays valid on both sides and the
   mutation is purely "which cell feeds which role in the recurrence". *)
let build_lstm_graph ~name ~swap_state =
  Graph_builder.build ~name
    ~outputs:(fun (out, hn, cn) -> [ out; hn; cn ])
    Graph_builder.(
      let* input_id = input ~shape:seq_shape ~name:"input" () in
      let* weight_ih_id =
        input ~shape:(mat_shape ~rows:(4 * k) ~cols:isz) ~name:"weight_ih" ()
      in
      let* weight_hh_id =
        input ~shape:(mat_shape ~rows:(4 * k) ~cols:k) ~name:"weight_hh" ()
      in
      let* bias_ih_id =
        input ~shape:(vec_shape ~n:(4 * k)) ~name:"bias_ih" ()
      in
      let* bias_hh_id =
        input ~shape:(vec_shape ~n:(4 * k)) ~name:"bias_hh" ()
      in
      let* h0_id = input ~shape:state_shape ~name:"h0" () in
      let* c0_id = input ~shape:state_shape ~name:"c0" () in
      let layer : Lstm.Lstm.Layer.t =
        {
          forward =
            {
              weight_ih = weight_ih_id;
              weight_hh = weight_hh_id;
              bias = Some (bias_ih_id, bias_hh_id);
            };
          reverse = None;
        }
      in
      let h0, c0 = if swap_state then (c0_id, h0_id) else (h0_id, c0_id) in
      Graph_builder.lstm params ~input:input_id ~layers:[ layer ] ~h0 ~c0 ())
  |> Err.or_raise ~pp_error:Graph_builder.pp_error

let identical_map ~src ~dst src_graph dst_graph =
  let pair src_id dst_id =
    Correspondence.pair
      (Option.get (Snapshot.edge src src_id))
      (Option.get (Snapshot.edge dst dst_id))
      Correspondence.Identical
  in
  List.map2 pair src_graph.Graph.inputs dst_graph.Graph.inputs
  @ List.map2 pair src_graph.Graph.outputs dst_graph.Graph.outputs

let%expect_test
    "Map_verify proves a small scan-backed LSTM stage against a separately \
     built, structurally identical one" =
  let graph_a = build_lstm_graph ~name:"lstm_a" ~swap_state:false in
  let graph_b = build_lstm_graph ~name:"lstm_b" ~swap_state:false in
  let module A = (val Version_fixture.of_graph graph_a) in
  let module B = (val Version_fixture.of_graph graph_b) in
  verify_map
    (hand_map ~src:A.snapshot ~dst:B.snapshot
       (identical_map ~src:A.snapshot ~dst:B.snapshot graph_a graph_b)
       [])
    ~src:A.snapshot ~dst:B.snapshot;
  [%expect
    {|
    {t0} -> {t0} identical: proved (structural) [exhaustive]
    {t1} -> {t1} identical: proved (structural) [exhaustive]
    {t2} -> {t2} identical: proved (structural) [exhaustive]
    {t3} -> {t3} identical: proved (structural) [exhaustive]
    {t4} -> {t4} identical: proved (structural) [exhaustive]
    {t5} -> {t5} identical: proved (structural) [exhaustive]
    {t6} -> {t6} identical: proved (structural) [exhaustive]
    {t7} -> {t7} identical: proved (structural) [exhaustive]
    {t8} -> {t8} identical: proved (structural) [exhaustive]
    {t9} -> {t9} identical: proved (structural) [exhaustive] |}]

(* The rejection half of the flagship test: an unchanged shape, an unchanged
   correspondence map, and a mutation entirely inside the scan's own
   recurrence (which initial-state cell feeds [h]/which feeds [c]). Every
   input still corresponds Identical, so [out]/[h_n]/[c_n] read the SAME free
   cells on both sides and a probe over them is a genuine counterexample, not
   a manufactured one -- matching .ai/native_transform_verify.md §8b's
   "settled frontier" precondition, since an LSTM stage's own inputs have no
   producer to expand through in the first place. *)
let%expect_test
    "Map_verify refutes an LSTM stage whose initial state is wired to the \
     wrong gate role, at an unchanged shape" =
  let graph_a = build_lstm_graph ~name:"lstm_a" ~swap_state:false in
  let graph_c = build_lstm_graph ~name:"lstm_c" ~swap_state:true in
  let module A = (val Version_fixture.of_graph graph_a) in
  let module C = (val Version_fixture.of_graph graph_c) in
  verify_map
    (hand_map ~src:A.snapshot ~dst:C.snapshot
       (identical_map ~src:A.snapshot ~dst:C.snapshot graph_a graph_c)
       [])
    ~src:A.snapshot ~dst:C.snapshot;
  [%expect
    {|
    {t0} -> {t0} identical: proved (structural) [exhaustive]
    {t1} -> {t1} identical: proved (structural) [exhaustive]
    {t2} -> {t2} identical: proved (structural) [exhaustive]
    {t3} -> {t3} identical: proved (structural) [exhaustive]
    {t4} -> {t4} identical: proved (structural) [exhaustive]
    {t5} -> {t5} identical: proved (structural) [exhaustive]
    {t6} -> {t6} identical: proved (structural) [exhaustive]
    {t7} -> {t7} identical: refuted: value at (0): src.t7 vs dst.t7 under {v0(0)=-0x1.c49ba5e353f7dp+0, v1(0)=0x1.ef1a9fbe76c8bp+1, v1(1,0,0,0,0,0)=-0x1.7df3b645a1cacp+1, v1(2,0,0,0,0,0)=-0x1.d16872b020c4ap+1, v1(3,0,0,0,0,0)=0x1p+1, v2(0)=-0x1.3f7ced916872bp-1, v2(1,0,0,0,0,0)=-0x1.e3d70a3d70a3dp+1, v2(2,0,0,0,0,0)=-0x1.73b645a1cac08p+0, v2(3,0,0,0,0,0)=0x1.4ed916872b021p+1, v3(0)=-0x1.26e978d4fdf3bp-5, v3(1,0,0,0,0,0)=-0x1.f6c8b43958106p+0, v3(2,0,0,0,0,0)=0x1.f4bc6a7ef9db2p+1, v3(3,0,0,0,0,0)=-0x1.4f5c28f5c28f6p+1, v4(0)=-0x1.eb851eb851eb8p-5, v4(1,0,0,0,0,0)=-0x1.4624dd2f1a9fcp+1, v4(2,0,0,0,0,0)=-0x1.c395810624dd3p+1, v4(3,0,0,0,0,0)=-0x1.ed916872b020cp-1, v5(0)=0x1.49ba5e353f7cfp-1, v6(0)=-0x1.d78d4fdf3b646p+1} [exhaustive]
    {t8} -> {t8} identical: refuted: value at (0): src.t8 vs dst.t8 under {v0(0)=-0x1.c49ba5e353f7dp+0, v0(1,0,0)=-0x1.a1cac083126e9p-1, v1(0)=0x1.ef1a9fbe76c8bp+1, v1(1,0,0,0,0,0)=-0x1.7df3b645a1cacp+1, v1(2,0,0,0,0,0)=-0x1.d16872b020c4ap+1, v1(3,0,0,0,0,0)=0x1p+1, v2(0)=-0x1.3f7ced916872bp-1, v2(1,0,0,0,0,0)=-0x1.e3d70a3d70a3dp+1, v2(2,0,0,0,0,0)=-0x1.73b645a1cac08p+0, v2(3,0,0,0,0,0)=0x1.4ed916872b021p+1, v3(0)=-0x1.26e978d4fdf3bp-5, v3(1,0,0,0,0,0)=-0x1.f6c8b43958106p+0, v3(2,0,0,0,0,0)=0x1.f4bc6a7ef9db2p+1, v3(3,0,0,0,0,0)=-0x1.4f5c28f5c28f6p+1, v4(0)=-0x1.eb851eb851eb8p-5, v4(1,0,0,0,0,0)=-0x1.4624dd2f1a9fcp+1, v4(2,0,0,0,0,0)=-0x1.c395810624dd3p+1, v4(3,0,0,0,0,0)=-0x1.ed916872b020cp-1, v5(0)=0x1.49ba5e353f7cfp-1, v6(0)=-0x1.d78d4fdf3b646p+1} [exhaustive]
    {t9} -> {t9} identical: refuted: value at (0): src.t9 vs dst.t9 under {v0(0)=0x1p+0, v0(1,0,0)=0x1p+1, v1(0)=0x1.8p+1, v1(1,0,0,0,0,0)=0x1p+2, v1(2,0,0,0,0,0)=0x1.4p+2, v1(3,0,0,0,0,0)=0x1.8p+2, v2(0)=0x1.cp+2, v2(1,0,0,0,0,0)=0x1p+3, v2(2,0,0,0,0,0)=0x1.2p+3, v2(3,0,0,0,0,0)=0x1.4p+3, v3(0)=0x1.6p+3, v3(1,0,0,0,0,0)=0x1.8p+3, v3(2,0,0,0,0,0)=0x1.ap+3, v3(3,0,0,0,0,0)=0x1.cp+3, v4(0)=0x1.ep+3, v4(1,0,0,0,0,0)=0x1p+4, v4(2,0,0,0,0,0)=0x1.1p+4, v4(3,0,0,0,0,0)=0x1.2p+4, v5(0)=0x1.3p+4, v6(0)=0x1.4p+4} [exhaustive] |}]
