(* Under node: [Loop_region_executor.make] wired in as [Eval_direct.run]'s
   own [~region_executor], on the SDPA graph, once at a toy shape and once at [fastvit_sa12]'s exact measured
   shape ([H=16 W=49 C=32]). Proves the executor agrees with the
   reference AND that the coverage counter shows it, not the fallback, ran. *)

open Graph_ir

let sdpa_graph ~query_shape ~key_shape =
  let mask_shape = Attention.Sdpa.score_shape ~query_shape ~key_shape in
  let materialize shape scale =
    Tensor.materialize shape (fun coord ->
        (scale *. float_of_int (Dim.to_int (Vec6.get coord Axis.W)))
        +. float_of_int (Dim.to_int (Vec6.get coord Axis.C))
        +. 1.)
  in
  let query = materialize query_shape 10. in
  let key = materialize key_shape 10. in
  let value = materialize key_shape 100. in
  let mask = Tensor.materialize mask_shape (fun _ -> 0.) in
  let g =
    Err.or_raise ~pp_error:Graph_builder.pp_error
      Graph_builder.(
        build ~name:"sdpa_region_executor_js" ~outputs:(fun output ->
            [ output ])
        @@
        let* qi = input ~shape:query_shape ~name:"query" () in
        let* ki = input ~shape:key_shape ~name:"key" () in
        let* vi = input ~shape:key_shape ~name:"value" () in
        let* mi = input ~shape:mask_shape ~name:"mask" () in
        sdpa
          { Attention.Sdpa.scale = Attention.Sdpa.Scale.Default }
          ~query:qi ~key:ki ~value:vi ~mask:mi ())
  in
  let inputs =
    match g.Graph.inputs with
    | [ qid; kid; vid; mid ] ->
        [ (qid, query); (kid, key); (vid, value); (mid, mask) ]
    | _ -> assert false
  in
  (g, inputs)

let check ?limits ~query_shape ~key_shape () =
  let g, inputs = sdpa_graph ~query_shape ~key_shape in
  let output_id = List.hd g.Graph.outputs in
  let reference =
    Err.or_raise ~pp_error:Eval_direct.pp_error (Eval_direct.run ~inputs g)
  in
  let coverage = Loop_region_executor.Coverage.create () in
  let region_executor =
    Loop_region_executor.make ?limits ~on_fallback:ignore coverage
  in
  let result =
    Err.or_raise ~pp_error:Eval_direct.pp_error
      (Eval_direct.run ~region_executor ~inputs g)
  in
  let agree =
    Tensor.equal_bits
      (Tensor_id.Map.find output_id reference)
      (Tensor_id.Map.find output_id result)
  in
  Fmt.pr "agree=%b generated_js=%d fallback=%d@." agree
    coverage.Loop_region_executor.Coverage.generated_js
    coverage.Loop_region_executor.Coverage.fallback

let%expect_test "toy SDPA shape: generated JS agrees with the reference" =
  check
    ~query_shape:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:3)
    ~key_shape:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:3 ~c:3)
    ();
  [%expect {| agree=true generated_js=1 fallback=0 |}]

let%expect_test
    "fastvit_sa12-scale SDPA shape (H=16 W=49 C=32): generated JS agrees with \
     the reference" =
  check
    ~query_shape:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:16 ~w:49 ~c:32)
    ~key_shape:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:16 ~w:49 ~c:32)
    ();
  [%expect {| agree=true generated_js=1 fallback=0 |}]

(* T2.2/T2.3's own mutation, kept as a permanent regression test: an
   impossibly tight [max_size] forces [Loop_region_program.lower] to
   refuse, proving the fallback path is reachable AND still correct (the
   result must still [agree] -- [Region_executor.default] is the reference
   materializer). *)
let%expect_test
    "an impossible limit forces the fallback path, which still agrees" =
  check
    ~limits:
      (Err.or_raise ~pp_error:Kernel.Limits.pp_error
         (Kernel.Limits.create ~max_size:1
            ~max_depth:Kernel.Limits.default.max_depth
            ~max_values:Kernel.Limits.default.max_values
            ~max_dep_depth:Kernel.Limits.default.max_dep_depth
            ~max_inputs:Kernel.Limits.default.max_inputs
            ~max_outputs:Kernel.Limits.default.max_outputs
            ~max_extent:Kernel.Limits.default.max_extent
            ~max_numel:Kernel.Limits.default.max_numel
            ~max_bytes:Kernel.Limits.default.max_bytes
            ~max_local_slots:Kernel.Limits.default.max_local_slots
            ~max_scan_state:Kernel.Limits.default.max_scan_state
            ~max_scan_updates_per_key:
              Kernel.Limits.default.max_scan_updates_per_key
            ~max_scan_updates_total:Kernel.Limits.default.max_scan_updates_total))
    ~query_shape:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:2 ~c:3)
    ~key_shape:(Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:3 ~c:3)
    ();
  [%expect {| agree=true generated_js=0 fallback=1 |}]

(* T3.3's own mutation, kept as a fast unit test rather than a run of the
   whole (expensive, real-archive) pipeline: a threshold above what a
   coverage value actually reaches must fail with a coverage message. *)
let%expect_test
    "Coverage.check: a threshold above what was reached fails with a coverage \
     message" =
  let coverage = Loop_region_executor.Coverage.create () in
  coverage.Loop_region_executor.Coverage.generated_js <- 2;
  coverage.Loop_region_executor.Coverage.fallback <- 0;
  let show min_generated_js =
    match Loop_region_executor.Coverage.check ~min_generated_js coverage with
    | Ok () -> Fmt.pr "ok@."
    | Error msg -> Fmt.pr "error: %s@." msg
  in
  show 2;
  show 3;
  [%expect
    {|
    ok
    error: coverage: generated_js=2 fallback=0, expected generated_js >= 3
    |}]
