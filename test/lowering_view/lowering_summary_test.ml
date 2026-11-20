(* Lowering_view.Summary on fixture nodes: totality, the metric vocabulary. *)

module A = Lowering_view.Artifact
module F = Native_test.Graph_fixtures
module Lv = Lowering_view

let build g =
  Err.or_raise ~pp_error:Loop_ir.Loop_bundle.pp_error
    (Loop_ir.Loop_bundle.build g)

let relu () =
  F.build "relu"
    Graph_builder.(
      let* x = input ~shape:(F.nhwc ~h:2 ~w:3 ~c:2) () in
      relu x)

let pool () =
  F.build2 "pool"
    Graph_builder.(
      let* x = input ~shape:(F.nhwc ~h:4 ~w:4 ~c:2) () in
      let params : Pool.MaxPool2dWithIndices.params =
        {
          ceil_mode = false;
          kernel = { h = Dim.extent 2; w = Dim.extent 2 };
          stride = { h = Op_config.Pos.of_int 2; w = Op_config.Pos.of_int 2 };
          pad = { h = Op_config.Nonneg.of_int 0; w = Op_config.Nonneg.of_int 0 };
        }
      in
      max_pool2d_with_indices params x)

let rms_norm () =
  F.build "rms_norm"
    Graph_builder.(
      let* x = input ~shape:(F.s 1 1 1 3 4 5) () in
      rms_norm { Norm.RmsNorm.dims = [ Axis.C ]; eps = 1e-5 } ~x ())

let nobody _ = false
let only g = List.hd (build g).Loop_ir.Loop_bundle.invocations

module S = Lv.Summary

let print_summary (s : S.t) =
  Fmt.pr
    "precision=%s vectorized=%b vec=%d scalar=%d ssa_refusal=%s cycles=%s@."
    (match s.S.precision with
    | None -> "none"
    | Some p -> Loop_ir.Loop_numerics.Precision.name p)
    s.S.vectorized s.S.vector_loops s.S.scalar_loops
    (match s.S.ssa_refusal with
    | None -> "none"
    | Some r -> Fmt.str "%a" A.Refusal.pp r)
    (match s.S.coral_cycles with
    | None -> "none"
    | Some c -> string_of_bool (c > 0.))

let%expect_test
    "a summary per invocation; a refusal in one does not fail the others" =
  let ok = only (relu ()) in
  let bad =
    {
      ok with
      Loop_ir.Loop_bundle.placed =
        Fusion_plan.default Loop_ir_test.Loop_programs.i64_load_of_f32_kernel;
    }
  in
  let b = build (relu ()) in
  let b = { b with Loop_ir.Loop_bundle.invocations = [ ok; bad; ok ] } in
  List.iter print_summary (S.of_bundle ~granted:nobody b);
  let granted = Graph_ir.Node_id.equal ok.Loop_ir.Loop_bundle.node in
  List.iter print_summary (S.of_bundle ~granted b);
  [%expect
    {|
    precision=f64 vectorized=true vec=1 scalar=0 ssa_refusal=none cycles=true
    precision=none vectorized=false vec=0 scalar=0 ssa_refusal=Loop_lower: t1: load of format f32 is not lowered cycles=none
    precision=f64 vectorized=true vec=1 scalar=0 ssa_refusal=none cycles=true
    precision=none vectorized=false vec=0 scalar=0 ssa_refusal=node n0 is realized only by a grant: its bundle body is a placeholder cycles=none
    precision=none vectorized=false vec=0 scalar=0 ssa_refusal=node n0 is realized only by a grant: its bundle body is a placeholder cycles=none
    precision=none vectorized=false vec=0 scalar=0 ssa_refusal=node n0 is realized only by a grant: its bundle body is a placeholder cycles=none |}]

let%expect_test "the pool's two outputs and a Region node are summarized" =
  List.iter
    (fun g -> List.iter print_summary (S.of_bundle ~granted:nobody (build g)))
    [ pool (); rms_norm () ];
  [%expect
    {|
    precision=f64 vectorized=false vec=0 scalar=5 ssa_refusal=none cycles=true
    precision=f64 vectorized=false vec=0 scalar=5 ssa_refusal=none cycles=true
    precision=f64 vectorized=true vec=1 scalar=0 ssa_refusal=none cycles=true |}]

let%expect_test "metric names round-trip and are in order" =
  List.iter
    (fun m ->
      assert (S.Metric.of_name (S.Metric.name m) = Some m);
      Fmt.pr "%s@." (S.Metric.name m))
    S.Metric.all;
  Fmt.pr "unknown: %b@." (S.Metric.of_name "nope" = None);
  let s = List.hd (S.of_bundle ~granted:nobody (build (relu ()))) in
  List.iter
    (fun m ->
      Fmt.pr "%s -> %s@." (S.Metric.name m)
        (match S.metric s m with None -> "none" | Some v -> Fmt.str "%g" v))
    S.Metric.all;
  [%expect
    {|
    coral_cycles
    scalar_loops
    vector_loops
    unknown: true
    coral_cycles -> 201.336
    scalar_loops -> 0
    vector_loops -> 1 |}]
