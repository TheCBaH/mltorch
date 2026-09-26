(* Under node: [Loop_region_executor.make_group] wired in as [Eval_direct.
   run]'s own [~region_group_executor], on the Lstm walk's own subject
   (design §1's "the Lstm group's executor was deferred", plan T7.2) --
   Lstm is the one op [Region_computation.is_region_authored] AND
   multi-output, so it is the only real subject this executor ever runs.
   Reuses [Native_op_walk_js.Native_op_walk]'s own "lstm" walk rather than
   hand-building weights/biases for every layer/direction, the same
   subject the walked-subject sweeps already trust. Proves every one of
   Lstm's three outputs (out/h_n/c_n) agrees with the reference AND that
   the coverage counter shows the generated path, not the fallback, ran. *)

let lstm_subject () =
  match Native_op_walk_js.Native_op_walk.find "lstm" with
  | None -> failwith "no lstm walk registered"
  | Some m ->
      let module M =
        (val m
            : Walk_core.Walk.Op
            with type subject = Native_op_walk_js.Native_subject.t)
      in
      let cfg = M.cascade M.initial in
      let subject, _pcg = M.build (Walk_core.Pcg.seed ~seed:0L ~seq:1L) cfg in
      subject

let check ?limits () =
  let subject = lstm_subject () in
  let g = subject.Native_op_walk_js.Native_subject.graph in
  let inputs = subject.Native_op_walk_js.Native_subject.inputs in
  let reference =
    Err.or_raise ~pp_error:Eval_direct.pp_error (Eval_direct.run ~inputs g)
  in
  let coverage = Loop_region_executor.Coverage.create () in
  let region_group_executor =
    Loop_region_executor.make_group ?limits ~on_fallback:ignore coverage
  in
  let result =
    Err.or_raise ~pp_error:Eval_direct.pp_error
      (Eval_direct.run ~region_group_executor ~inputs g)
  in
  let agree =
    List.for_all
      (fun oid ->
        Tensor.equal_bits
          (Tensor_id.Map.find oid reference)
          (Tensor_id.Map.find oid result))
      g.Graph_ir.Graph.outputs
  in
  Fmt.pr "outputs=%d agree=%b generated_js=%d fallback=%d@."
    (List.length g.Graph_ir.Graph.outputs)
    agree coverage.Loop_region_executor.Coverage.generated_js
    coverage.Loop_region_executor.Coverage.fallback

let%expect_test "lstm: generated JS agrees with the reference on every output" =
  check ();
  [%expect {| outputs=3 agree=true generated_js=3 fallback=0 |}]

(* The group twin of [region_executor_test.ml]'s own "impossible limit"
   mutation: [Loop_region_program.lower_group] must refuse just as reliably
   as the solo [lower] does, and the fallback ([Region_executor.
   default_group]) must still agree. *)
let%expect_test
    "an impossible limit forces the group fallback path, which still agrees" =
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
    ();
  [%expect {| outputs=3 agree=true generated_js=0 fallback=1 |}]
