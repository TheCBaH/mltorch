(* Under node: [Loop_node_executor]'s [Node_executor.t] wired in as
   [Eval_direct.run]'s own [~node_executor]. Design §8's table plus the plan's
   named mutations (S4/T4.1-T4.5), each shown red then reverted except where
   noted "kept". *)

open Graph_ir

let relu_graph () =
  let shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:4 in
  let x =
    Tensor.materialize shape (fun coord ->
        -2. +. float_of_int (Dim.to_int (Vec6.get coord Axis.C)))
  in
  let g =
    Err.or_raise ~pp_error:Graph_builder.pp_error
      Graph_builder.(
        build ~name:"node_executor_js_relu" ~outputs:(fun r -> [ r ])
        @@
        let* xi = input ~shape ~name:"x" () in
        relu xi)
  in
  (g, List.combine g.Graph.inputs [ x ])

let run ?node_executor (g, inputs) =
  Err.or_raise ~pp_error:Eval_direct.pp_error
    (Eval_direct.run ?node_executor ~inputs g)

let%expect_test "relu: generated JS agrees with the reference, coverage counted"
    =
  let subject = relu_graph () in
  let g, _ = subject in
  let output = List.hd g.Graph.outputs in
  let reference = run subject in
  let executor = Loop_node_executor.create () in
  let result =
    run ~node_executor:(Loop_node_executor.node_executor executor) subject
  in
  let agree =
    Tensor.equal_bits
      (Tensor_id.Map.find output reference)
      (Tensor_id.Map.find output result)
  in
  Fmt.pr "agree=%b generated_js=%d@." agree
    (Loop_node_executor.Coverage.total
       executor.Loop_node_executor.coverage.generated_js);
  [%expect {| agree=true generated_js=1 |}]

(* Mutation 3 (T4.3): an impossible [max_size] makes every node refuse to
   adapt/lower, so every node falls back -- [generated_js = 0] -- and results
   are UNCHANGED, since the direct fallback is always correct. *)
let%expect_test "impossible max_size: every node falls back, results unchanged"
    =
  let subject = relu_graph () in
  let g, _ = subject in
  let output = List.hd g.Graph.outputs in
  let reference = run subject in
  let d = Kernel.Limits.default in
  let impossible_limits =
    Err.or_raise ~pp_error:Kernel.Limits.pp_error
      (Kernel.Limits.create ~max_size:1 ~max_depth:d.Kernel.Limits.max_depth
         ~max_values:d.Kernel.Limits.max_values
         ~max_dep_depth:d.Kernel.Limits.max_dep_depth
         ~max_inputs:d.Kernel.Limits.max_inputs
         ~max_outputs:d.Kernel.Limits.max_outputs
         ~max_extent:d.Kernel.Limits.max_extent
         ~max_numel:d.Kernel.Limits.max_numel
         ~max_bytes:d.Kernel.Limits.max_bytes
         ~max_local_slots:d.Kernel.Limits.max_local_slots
         ~max_scan_state:d.Kernel.Limits.max_scan_state
         ~max_scan_updates_per_key:d.Kernel.Limits.max_scan_updates_per_key
         ~max_scan_updates_total:d.Kernel.Limits.max_scan_updates_total)
  in
  let executor =
    Loop_node_executor.create ~limits:impossible_limits ~on_fallback:ignore ()
  in
  let result =
    run ~node_executor:(Loop_node_executor.node_executor executor) subject
  in
  let agree =
    Tensor.equal_bits
      (Tensor_id.Map.find output reference)
      (Tensor_id.Map.find output result)
  in
  let coverage = executor.Loop_node_executor.coverage in
  Fmt.pr "agree=%b generated_js=%d fallback=%d@." agree
    (Loop_node_executor.Coverage.total coverage.generated_js)
    (Loop_node_executor.Coverage.total coverage.fallback);
  (match Loop_node_executor.Coverage.check ~min_generated_js:1 coverage with
  | Ok () -> Fmt.pr "check: unexpectedly ok@."
  | Error msg -> Fmt.pr "check: %s@." msg);
  [%expect
    {|
    agree=true generated_js=0 fallback=1
    check: coverage: generated_js=0 fallback=1 pending=0, expected generated_js >= 1
    |}]

(* Mutation 4 (T4.3): a threshold above reach fails with a coverage message;
   kept as a permanent test alongside a threshold within reach succeeding, so
   the assertion is proven load-bearing both ways. *)
let%expect_test
    "coverage check: threshold above reach fails, within reach passes" =
  let executor = Loop_node_executor.create () in
  ignore
    (run
       ~node_executor:(Loop_node_executor.node_executor executor)
       (relu_graph ()));
  let coverage = executor.Loop_node_executor.coverage in
  (match Loop_node_executor.Coverage.check ~min_generated_js:1 coverage with
  | Ok () -> Fmt.pr "within reach: ok@."
  | Error msg -> Fmt.pr "within reach: unexpectedly failed: %s@." msg);
  (match Loop_node_executor.Coverage.check ~min_generated_js:999 coverage with
  | Ok () -> Fmt.pr "above reach: unexpectedly ok@."
  | Error msg -> Fmt.pr "above reach: %s@." msg);
  [%expect
    {|
    within reach: ok
    above reach: coverage: generated_js=1 fallback=0 pending=0, expected generated_js >= 999
    |}]

(* [check_parity]'s allow-list mechanism itself, exercised directly on a
   [Coverage.t] rather than through a real excluded graph: every op kind
   [scope] once excluded (M1-era Bool/I64/index outputs, T6.*'s unwalked
   factories, T7.1's Unbind/Split_with_sizes) has since routed, so there is
   no longer a real subject that lands in [pending] to build a graph
   around -- exactly the state design §4.4 calls "closure". A synthetic
   bump proves the mechanism the same way a real one would, and stays
   meaningful regardless of which op kinds are still open at any given
   milestone. *)
let%expect_test
    "check_parity: an unnamed pending op kind fails, naming it passes" =
  let coverage = Loop_node_executor.Coverage.create () in
  Loop_node_executor.Coverage.bump coverage.Loop_node_executor.Coverage.pending
    "SomeFutureOp";
  (match Loop_node_executor.Coverage.check_parity ~allow:[] coverage with
  | Ok () -> Fmt.pr "allow=[]: unexpectedly ok@."
  | Error msg -> Fmt.pr "allow=[]: %s@." msg);
  (match
     Loop_node_executor.Coverage.check_parity ~allow:[ "SomeFutureOp" ] coverage
   with
  | Ok () -> Fmt.pr "allow=[SomeFutureOp]: ok@."
  | Error msg -> Fmt.pr "allow=[SomeFutureOp]: unexpectedly failed: %s@." msg);
  [%expect
    {|
    allow=[]: check_parity: outside the allow-list: SomeFutureOp=1
    allow=[SomeFutureOp]: ok
    |}]

(* Mutation 5 (T4.4): a sabotaged kernel (the real emitted source for
   [relu], hand-perturbed the way design §6's own example does -- a binary
   op's operator swapped) must be caught by shadow mode: it still runs
   (no exception, no refusal), produces a WRONG but plausible value, and
   [~shadow:true] must report the disagreement and return the trusted
   (direct) result rather than the sabotaged one. *)
let%expect_test "shadow mode catches a sabotaged kernel" =
  let subject = relu_graph () in
  let g, _ = subject in
  let node = List.hd g.Graph.nodes in
  let output = Output_ordinal.zero in
  let oid = List.hd g.Graph.outputs in
  let reference_tensor = Tensor_id.Map.find oid (run subject) in
  let program =
    Err.or_raise ~pp_error:Loop_node_program.pp_error
      (Loop_node_program.lower g node ~output)
  in
  let real_source = Loop_js.emit program in
  (* [relu]'s body is the ternary [x < 0 ? 0 : x]: sabotage by flipping the
     comparison to [x > 0 ? 0 : x] -- the same class of "still runs, silently
     wrong" perturbation as design §6's own "Mul printed as +" example. *)
  let sabotaged_source =
    let pattern = "< 0 ?" and replacement = "> 0 ?" in
    let len_s = String.length real_source and len_p = String.length pattern in
    let rec find i =
      if i + len_p > len_s then None
      else if String.sub real_source i len_p = pattern then Some i
      else find (i + 1)
    in
    match find 0 with
    | None ->
        Fmt.pr "SOURCE: %s@." real_source;
        failwith "relu's emitted source no longer contains '< 0 ?'"
    | Some i ->
        String.sub real_source 0 i ^ replacement
        ^ String.sub real_source (i + len_p) (len_s - i - len_p)
  in
  let sabotaged =
    Err.or_raise ~pp_error:Loop_js_exec.pp_error
      (Loop_js_exec.compile_as program sabotaged_source)
  in
  let executor =
    Loop_node_executor.create ~shadow:true ~on_fallback:ignore ()
  in
  Hashtbl.replace executor.Loop_node_executor.table oid
    (Loop_node_executor.Compiled sabotaged);
  let result =
    run ~node_executor:(Loop_node_executor.node_executor executor) subject
  in
  let matches_direct =
    Tensor.equal_bits reference_tensor (Tensor_id.Map.find oid result)
  in
  let coverage = executor.Loop_node_executor.coverage in
  Fmt.pr "returned_direct_result=%b generated_js=%d fallback=%d@."
    matches_direct
    (Loop_node_executor.Coverage.total coverage.generated_js)
    (Loop_node_executor.Coverage.total coverage.fallback);
  [%expect {| returned_direct_result=true generated_js=0 fallback=1 |}]

(* Mutation 6 (T4.2): [precompile] fills the table shape-only, before any
   weight is bound; a later evaluation must hit that same table rather than
   growing it, which is the whole point of precompiling ahead of inference. *)
let%expect_test "precompile fills the table; running afterward does not grow it"
    =
  let subject = relu_graph () in
  let g, _ = subject in
  let executor = Loop_node_executor.create () in
  Loop_node_executor.precompile executor g;
  let after_precompile = Hashtbl.length executor.Loop_node_executor.table in
  ignore
    (run ~node_executor:(Loop_node_executor.node_executor executor) subject);
  let after_run = Hashtbl.length executor.Loop_node_executor.table in
  Fmt.pr "after_precompile=%d after_run=%d@." after_precompile after_run;
  [%expect {| after_precompile=1 after_run=1 |}]

(* T4.5: every walked subject, through the production seam, shadow on. This
   is walk-scale parity for M1 -- the model-scale version is T5.2/T5.4,
   manual against real archives. The allow-list is exactly the M2/M3 kinds
   the walked ops exercise -- named by [Graph_ir.op_name], which is each op
   module's own [.name] field and does not always match its OCaml
   constructor (["MaxDim"], not ["Max_dim"]): [Unbind] (M3, excluded by
   [scope] unconditionally). T6.0 (an I64-declared [Kernel.Output.t] can now
   resolve its signature from [values_i64]) plus [Loop_node_program]'s own
   sibling-i64-stage pruning closed every other former gap in one step:
   [MaxDim]/[Max_pool2d_with_indices]'s index ordinal (T6.5) and
   [Add]/[Mul]/[Permute]/[Reshape]/[Sub]'s I64-operand arm (T6.1) all now
   agree, so [op_name] no longer needs a carve-out for any of them. [Lstm]
   needs no entry: it is Region-authored, so [Eval_direct] never calls
   [Node_executor] for it at all -- it is not merely allowed, it is
   invisible to this coverage table by construction. *)
let%expect_test "every walked subject: node_executor + shadow, M1 allow-list" =
  (* A FRESH executor (and so a fresh compile table) per subject: the table is
     keyed by bare [Tensor_id.t], unique only within the graph an executor was
     built for (design §4.2, "per executor instance") -- sharing one across
     every walked op's own graph let a compiled kernel from one graph answer
     for an unrelated node in another whose id happened to collide, which is
     what the shadow's own bitwise check is there to catch (and did: see the
     Findings this test's first, broken version produced). *)
  let coverage = Loop_node_executor.Coverage.create () in
  let silent = Format.make_formatter (fun _ _ _ -> ()) (fun () -> ()) in
  let verify _ppf (s : Native_op_walk_js.Native_op_walk.Subject.t) =
    let executor =
      Loop_node_executor.create ~shadow:true ~on_fallback:ignore ()
    in
    ignore
      (Err.or_raise ~pp_error:Eval_direct.pp_error
         (Eval_direct.run
            ~node_executor:(Loop_node_executor.node_executor executor)
            ~inputs:s.Native_op_walk_js.Native_op_walk.Subject.inputs
            s.Native_op_walk_js.Native_op_walk.Subject.graph));
    Loop_node_executor.Coverage.merge_into ~into:coverage
      executor.Loop_node_executor.coverage;
    true
  in
  List.iteri
    (fun index (m : Native_op_walk_js.Native_op_walk.op) ->
      ignore
        (Walk_core.Walk.run m ~verify ~ppf:silent
           ~pcg:(Walk_core.Pcg.seed ~seed:(Int64.of_int index) ~seq:1L)
           ~steps:5))
    Native_op_walk_js.Native_op_walk.all_walks;
  (match
     Loop_node_executor.Coverage.check_parity ~allow:[ "Unbind" ] coverage
   with
  | Ok () -> Fmt.pr "check_parity (M1 allow-list): ok@."
  | Error msg -> Fmt.pr "check_parity (M1 allow-list): %s@." msg);
  (match Loop_node_executor.Coverage.check_parity ~allow:[] coverage with
  | Ok () -> Fmt.pr "check_parity (empty allow-list): unexpectedly ok@."
  | Error msg -> Fmt.pr "check_parity (empty allow-list): %s@." msg);
  [%expect
    {|
    check_parity (M1 allow-list): ok
    check_parity (empty allow-list): unexpectedly ok |}]

(* T4.5's remaining piece: one test per [admit] rejection, with the executor
   installed, proving the error is IDENTICAL to the no-executor case --
   [admit] runs before [Node_executor] is ever consulted (design §4.2), so a
   rejected node reaches neither path and both must report the same thing by
   construction, not by coincidence. *)
let error_of subject ~node_executor =
  let g, inputs = subject in
  match Err.payload (Eval_direct.run ?node_executor ~inputs g) with
  | Ok _ -> "unexpectedly ok"
  | Error e -> Fmt.str "%a" Eval_direct.pp_error e

let%expect_test "admit rejections: identical error with the executor installed"
    =
  let shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:4 in
  let executor = Loop_node_executor.create ~on_fallback:ignore () in
  let with_executor = Some (Loop_node_executor.node_executor executor) in
  let bool_arithmetic =
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"admit_bool_arithmetic" ~outputs:(fun r -> [ r ])
          @@
          let* xb = input ~shape ~fmt:(Payload.Fmt Payload.Bool) () in
          let* xf = input ~shape () in
          add xb xf)
    in
    let xb = Tensor.materialize_bool shape (fun _ -> false) in
    let xf = Tensor.materialize shape (fun _ -> 1.) in
    (g, List.combine g.Graph_ir.Graph.inputs [ xb; xf ])
  in
  let mixed_dtype =
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"admit_mixed_dtype" ~outputs:(fun r -> [ r ])
          @@
          let* xi = input ~shape ~fmt:(Payload.Fmt Payload.I64) () in
          let* xf = input ~shape () in
          add xi xf)
    in
    let xi = Tensor.materialize_i64 shape (fun _ -> 1L) in
    let xf = Tensor.materialize shape (fun _ -> 1.) in
    (g, List.combine g.Graph_ir.Graph.inputs [ xi; xf ])
  in
  let bool_scalar_arithmetic =
    let g =
      Err.or_raise ~pp_error:Graph_builder.pp_error
        Graph_builder.(
          build ~name:"admit_bool_scalar_arithmetic" ~outputs:(fun r -> [ r ])
          @@
          let* xb = input ~shape ~fmt:(Payload.Fmt Payload.Bool) () in
          add_scalar 1.0 xb)
    in
    let xb = Tensor.materialize_bool shape (fun _ -> false) in
    (g, List.combine g.Graph_ir.Graph.inputs [ xb ])
  in
  List.iter
    (fun (name, subject) ->
      let without = error_of subject ~node_executor:None in
      let with_ = error_of subject ~node_executor:with_executor in
      Fmt.pr "%s: identical=%b (%s)@." name (String.equal without with_) without)
    [
      ("bool_arithmetic", bool_arithmetic);
      ("mixed_dtype", mixed_dtype);
      ("bool_scalar_arithmetic", bool_scalar_arithmetic);
    ];
  [%expect
    {|
    bool_arithmetic: identical=true (add: arithmetic on a Bool operand is not supported, a=bool b=f32)
    mixed_dtype: identical=true (add: unsupported mixed dtype, a=i64 b=f32)
    bool_scalar_arithmetic: identical=true (add_scalar: arithmetic on a Bool operand is not supported, x=bool)
    |}]
