(* [Me_classify]: partiality versus defect (.ai/model_explorer_design.md).

   The classifiers are TOTAL matches over the upstream error rows, so the
   compiler is what keeps them complete. What a test can add is the other half:
   that every row is classified the way the matrix says, enumerated rather than
   sampled, so the matrix and the code cannot drift apart — which is exactly
   what happened once already and is why the plan calls for both a payload case
   and a domain case.

   The distinction matters because the two answers tell a user different things.
   Reporting a defect of ours as "outside the dialect" tells them to change
   their model to work around our bug. *)

module MC = Me_classify
module C = Me_session.Capability

let nid = Graph_ir.Node_id.of_int
let tid = Graph_ir.Tensor_id.of_int
let show label v = Format.printf "%-38s %a@." label MC.pp_verdict v

(* --- Native4D --- *)

let%expect_test "every Native4D row, classified" =
  let shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1 in
  let rows : (string * Native4d.Error.t) list =
    [
      ("Missing_constant_payload", `Missing_constant_payload (nid 0, tid 0));
      ("Axis_outside_dialect", `Axis_outside_dialect (nid 0, Axis.N));
      ( "Batch_norm_extent",
        `Batch_norm_extent (nid 0, tid 0, Dim.extent 1, Dim.extent 2) );
      ("Dynamic_batch_norm", `Dynamic_batch_norm (nid 0));
      ("Live_max_pool_indices", `Live_max_pool_indices (nid 0, tid 0));
      ("Lossy_bmm_operand", `Lossy_bmm_operand (nid 0, tid 0));
      ( "Non_four_dimensional_tensor",
        `Non_four_dimensional_tensor (tid 0, shape) );
      ("Unsupported_bmm_batch", `Unsupported_bmm_batch (nid 0, Dim.extent 3));
      ( "Unsupported_grouped_transposed_conv",
        `Unsupported_grouped_transposed_conv (nid 0, 2) );
      ( "Unsupported_op",
        `Unsupported_op (nid 0, Graph_ir.Relu { Pointwise.Relu.x = tid 0 }) );
      ("Bad_constant_payload", `Bad_constant_payload (tid 0));
    ]
  in
  List.iter (fun (n, e) -> show n (MC.native4d e)) rows;
  [%expect
    {|
    Missing_constant_payload               unavailable requires_payloads
    Axis_outside_dialect                   unavailable outside_dialect_domain
    Batch_norm_extent                      unavailable outside_dialect_domain
    Dynamic_batch_norm                     unavailable outside_dialect_domain
    Live_max_pool_indices                  unavailable outside_dialect_domain
    Lossy_bmm_operand                      unavailable outside_dialect_domain
    Non_four_dimensional_tensor            unavailable outside_dialect_domain
    Unsupported_bmm_batch                  unavailable outside_dialect_domain
    Unsupported_grouped_transposed_conv    unavailable outside_dialect_domain
    Unsupported_op                         unavailable outside_dialect_domain
    Bad_constant_payload                   fatal |}]

let%expect_test "payloads remove exactly one failure mode" =
  (* The reason Native4D is conditional for a .pt2 as well as for a bare
     model.json: nine of the ten rejections above have nothing to do with
     payloads, so having them does not put a graph inside the dialect. *)
  let outside =
    [
      MC.native4d (`Unsupported_grouped_transposed_conv (nid 0, 2));
      MC.native4d
        (`Non_four_dimensional_tensor
           (tid 0, Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:1));
    ]
  in
  Printf.printf "domain rejections still unavailable with payloads: %b\n"
    (List.for_all
       (fun v -> v = MC.Unavailable C.Outside_dialect_domain)
       outside);
  [%expect {| domain rejections still unavailable with payloads: true |}]

(* --- Lowering --- *)

let%expect_test "every Native_interp row, classified" =
  (* The row every other capability's fate follows from: it decides whether
     there is a Native graph at all. The two recoverable rows are recoverable
     for a .pt2 and a bare model.json alike, because [lower] takes an
     ExportedProgram and gains no operator support from the archive payload. *)
  let rows : (string * Native_interp.error) list =
    [
      ("Unsupported_operator", `Unsupported_operator "aten.bogus");
      ("Unsupported_input", `Unsupported_input `Non_tensor);
      ("Malformed_graph", `Undefined_ssa "y");
      (* The per-node output ceiling, in BOTH spellings a caller can meet: the
         lowerer's own bound on a serialized name list, and the same limit
         enforced inside shape inference, arriving wrapped as a builder error.
         Both must be recoverable, and the nested one must be matched ahead of
         the generic [`Build] row below -- which is [fatal], and is the row it
         would otherwise fall into. Telling a user their model is too big is a
         different answer from telling them we have a bug. *)
      ( "Output_count_over_limit",
        `Output_count_over_limit
          { Shape_error.Output_count.limit = 4096; observed = At_least 4096 } );
      ( "Build Output_count_over_limit",
        `Build
          (`Output_count_over_limit
             { Shape_error.Output_count.limit = 4096; observed = Exact 5000 })
      );
      ("Tensor_bridge", `Tensor_bridge (`Unsupported_dtype Pt2_dtype.Int64));
      ("Eval", `Eval (`Missing_input (tid 0)));
      ("Build", `Build (`Missing_tensor_sig (tid 0)));
      ("Provenance", `Provenance (`Unknown_node_id (nid 0)));
      ("Transform", `Transform (`Not_converged "fold"));
      ("Verify", `Verify (`Missing_signature (tid 0)));
      ("Lens", `Lens `Sidecar_graph_mismatch);
    ]
  in
  List.iter (fun (n, e) -> show n (MC.lowering e)) rows;
  [%expect
    {|
    Unsupported_operator                   unavailable unsupported_operator
    Unsupported_input                      unavailable unsupported_input
    Malformed_graph                        fatal
    Output_count_over_limit                unavailable over_limit
    Build Output_count_over_limit          unavailable over_limit
    Tensor_bridge                          fatal
    Eval                                   fatal
    Build                                  fatal
    Provenance                             fatal
    Transform                              fatal
    Verify                                 fatal
    Lens                                   fatal |}]

let%expect_test "every reason has a diagnostic code, and it is not assumed" =
  (* The two closed vocabularies meet in one place. They agree in spelling,
     which is exactly why the map is written out: a code added for the worker
     protocol must not silently become a capability reason. *)
  List.iter
    (fun r ->
      Format.printf "%-26s %s@."
        (Core.Pretty.to_string MC.pp_verdict (MC.Unavailable r))
        (Me_limits.Diagnostic.Code.to_string (MC.diagnostic_code r)))
    [
      C.Unsupported_operator;
      C.Unsupported_input;
      C.Unsupported_graph_shape;
      C.Outside_dialect_domain;
      C.Over_limit;
      C.Requires_payloads;
      C.Prerequisite_unavailable;
      C.Not_implemented;
    ];
  [%expect
    {|
    unavailable unsupported_operator unsupported_operator
    unavailable unsupported_input unsupported_input
    unavailable unsupported_graph_shape unsupported_graph_shape
    unavailable outside_dialect_domain outside_dialect_domain
    unavailable over_limit     over_limit
    unavailable requires_payloads requires_payloads
    unavailable prerequisite_unavailable prerequisite_unavailable
    unavailable not_implemented not_implemented |}]

(* --- Kernel --- *)

let%expect_test "every Kernel_adapt row, classified" =
  let rows : (string * Kernel_adapt.error) list =
    [
      ("Passthrough_output", `Passthrough_output (tid 0));
      ("Too_many_values", `Too_many_values 4096);
      ("Too_many_inputs", `Too_many_inputs 1024);
      ("Too_many_outputs", `Too_many_outputs 1024);
      ("Dependency_too_deep", `Dependency_too_deep 1024);
      ("Eval_too_deep", `Eval_too_deep 2048);
      ("Numel_too_large", `Numel_too_large (tid 0));
      ("Missing_live_output", `Missing_live_output (tid 0));
      ("Unknown_program_output", `Unknown_program_output (tid 0));
      ("Output_not_selected", `Output_not_selected (tid 0));
      ("Unknown_selection", `Unknown_selection (tid 0));
      ("Duplicate_id", `Duplicate_id (tid 0));
      ("Unknown_output", `Unknown_output (tid 0));
      ("Unreachable_value", `Unreachable_value (tid 0));
      ("Quant_contract", `Quant_contract (tid 0));
    ]
  in
  List.iter (fun (n, e) -> show n (MC.kernel e)) rows;
  [%expect
    {|
    Passthrough_output                     unavailable unsupported_graph_shape
    Too_many_values                        unavailable over_limit
    Too_many_inputs                        unavailable over_limit
    Too_many_outputs                       unavailable over_limit
    Dependency_too_deep                    unavailable over_limit
    Eval_too_deep                          unavailable over_limit
    Numel_too_large                        unavailable over_limit
    Missing_live_output                    fatal
    Unknown_program_output                 fatal
    Output_not_selected                    fatal
    Unknown_selection                      fatal
    Duplicate_id                           fatal
    Unknown_output                         fatal
    Unreachable_value                      fatal
    Quant_contract                         fatal |}]

let%expect_test "a limit is not a defect, and a defect is not a limit" =
  (* The two classes the Kernel table exists to separate: a real model can be
     too big, which is a bound doing its job; a structural failure in a
     repository-generated stage program is ours. *)
  Printf.printf "over-limit recoverable: %b\ninternal fatal:        %b\n"
    (MC.kernel (`Too_many_values 1) = MC.Unavailable C.Over_limit)
    (MC.kernel (`Missing_live_output (tid 0)) = MC.Fatal);
  [%expect
    {|
    over-limit recoverable: true
    internal fatal:        true |}]

(* --- prerequisite propagation --- *)

(* Starts from a vector the capability table ACCEPTS: loop_ir and codegen are
   permanently unimplemented, so a producer never offers them anything else,
   and a fixture that set them Available would show [propagate] carrying
   forward a vector [Session.validate] rejects. *)
let caps status =
  List.map
    (fun k ->
      let status =
        match k with
        | C.Feature C.Loop_ir | C.Feature C.Codegen ->
            C.Unavailable { reason = C.Not_implemented; detail = None }
        | _ -> status
      in
      { C.key = k; C.status })
    C.all_keys

let render caps =
  List.iter
    (fun (c : C.t) ->
      Printf.printf "%-28s %s\n" (C.key_name c.C.key)
        (match c.C.status with
        | C.Not_requested -> "not_requested"
        | C.Available _ -> "available"
        | C.Unavailable { reason = C.Prerequisite_unavailable; _ } ->
            "unavailable prerequisite_unavailable"
        | C.Unavailable { reason = C.Not_implemented; _ } ->
            "unavailable not_implemented"
        | C.Unavailable _ -> "unavailable other"))
    caps

let%expect_test "the COMPLETE vector when lowering is unavailable" =
  (* The whole vector, not just the failing key: a test that checked only
     [Initial_native] would let every downstream row drift, which is what the
     unsupported-model golden exists to prevent. *)
  render (MC.propagate ~lowering_available:false (caps (C.Available C.Present)));
  [%expect
    {|
    stage:source                 available
    stage:initial_native         unavailable prerequisite_unavailable
    stage:canonical              unavailable prerequisite_unavailable
    stage:native4d               unavailable prerequisite_unavailable
    stage:stage_program          unavailable prerequisite_unavailable
    stage:kernel                 unavailable prerequisite_unavailable
    stage:fusion                 unavailable prerequisite_unavailable
    feature:flow                 unavailable prerequisite_unavailable
    feature:verification         unavailable prerequisite_unavailable
    feature:pass_audits          unavailable prerequisite_unavailable
    feature:fold                 unavailable prerequisite_unavailable
    feature:expression_detail    available
    feature:loop_ir              unavailable not_implemented
    feature:codegen              unavailable not_implemented |}]

let%expect_test "not_requested takes precedence over blocked" =
  (* A feature nobody asked for was not blocked by anything, and reporting it as
     blocked invents a dependency the caller never exercised. *)
  render (MC.propagate ~lowering_available:false (caps C.Not_requested));
  [%expect
    {|
    stage:source                 not_requested
    stage:initial_native         not_requested
    stage:canonical              not_requested
    stage:native4d               not_requested
    stage:stage_program          not_requested
    stage:kernel                 not_requested
    stage:fusion                 not_requested
    feature:flow                 not_requested
    feature:verification         not_requested
    feature:pass_audits          not_requested
    feature:fold                 not_requested
    feature:expression_detail    not_requested
    feature:loop_ir              unavailable not_implemented
    feature:codegen              unavailable not_implemented |}]

let%expect_test "with lowering available nothing is rewritten" =
  let before = caps (C.Available C.Present) in
  let after = MC.propagate ~lowering_available:true before in
  Printf.printf "unchanged: %b\n" (before = after);
  [%expect {| unchanged: true |}]

let%expect_test "which keys depend on lowering" =
  List.iter
    (fun k ->
      Printf.printf "%-28s %b\n" (C.key_name k) (MC.depends_on_lowering k))
    C.all_keys;
  [%expect
    {|
    stage:source                 false
    stage:initial_native         true
    stage:canonical              true
    stage:native4d               true
    stage:stage_program          true
    stage:kernel                 true
    stage:fusion                 true
    feature:flow                 true
    feature:verification         true
    feature:pass_audits          true
    feature:fold                 true
    feature:expression_detail    false
    feature:loop_ir              false
    feature:codegen              false |}]
