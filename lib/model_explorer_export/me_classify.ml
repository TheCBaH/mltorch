(* Two exhaustive classifiers. See the .mli. *)

module C = Me_session.Capability

type verdict = Unavailable of C.reason | Fatal

let pp_verdict fmt = function
  | Fatal -> Fmt.pf fmt "fatal"
  | Unavailable r ->
      Fmt.pf fmt "unavailable %s"
        (match r with
        | C.Unsupported_operator -> "unsupported_operator"
        | C.Unsupported_input -> "unsupported_input"
        | C.Unsupported_graph_shape -> "unsupported_graph_shape"
        | C.Outside_dialect_domain -> "outside_dialect_domain"
        | C.Over_limit -> "over_limit"
        | C.Requires_payloads -> "requires_payloads"
        | C.Prerequisite_unavailable -> "prerequisite_unavailable"
        | C.Not_implemented -> "not_implemented")

(* Total over [Native4d.Error.t]. No wildcard: a row added upstream must be
   classified here rather than defaulting to whichever answer happened to be
   written last, and the two answers differ in what they tell the user to do. *)
let native4d : [< Native4d.Error.t ] -> verdict = function
  (* The one failure mode payloads remove. *)
  | `Missing_constant_payload _ -> Unavailable C.Requires_payloads
  (* The ten domain rejections. Having payloads does not put a graph inside the
     dialect, which is why Native4D stays conditional for both input kinds. *)
  | `Axis_outside_dialect _ | `Batch_norm_extent _ | `Dynamic_batch_norm _
  | `Live_max_pool_indices _ | `Lossy_bmm_operand _
  | `Non_four_dimensional_tensor _ | `Unsupported_bmm_batch _
  | `Unsupported_grouped_conv _ | `Unsupported_grouped_transposed_conv _
  | `Unsupported_op _ ->
      Unavailable C.Outside_dialect_domain
  (* A payload that WAS supplied and is wrong, and a map or view invariant
     failure, are defects. Reporting either as "outside the dialect" tells the
     user to change their model to work around our bug. *)
  | `Bad_constant_payload _ | `Map _ | `View _ -> Fatal

let kernel : [< Kernel_adapt.error ] -> verdict = function
  (* Recoverable, and the kernel suite builds a graph its own comment calls
     legal that produces it: a graph input used directly as a graph output has
     no stage to name. *)
  | `Passthrough_output _ -> Unavailable C.Unsupported_graph_shape
  (* [Kernel]'s limit rows: a real model can be too big, and that is a bound
     doing its job rather than a defect. *)
  | `Too_many_values _ | `Too_many_inputs _ | `Too_many_outputs _
  | `Dependency_too_deep _ | `Eval_too_deep _ | `Extent_too_large _
  | `Numel_too_large _ ->
      Unavailable C.Over_limit
  (* Everything else is a defect. The stage program is repository-generated, so
     a structural failure in it is ours; and the two selection rows are
     reachable only through [?select], which whole-program export never
     passes. *)
  | `Program_invalid _ | `Unknown_stage_source _ | `Missing_live_output _
  | `Unknown_program_output _ | `Output_not_selected _ | `Unknown_selection _
  | `Duplicate_id _ | `Signature_id_mismatch _ | `Unresolved_source _
  | `Forward_reference _ | `Unknown_output _ | `Unreachable_value _
  | `Not_materializable _ | `Quant_contract _ | `Body _ ->
      Fatal

let requires_payloads_without_them = C.Requires_payloads

(* Which keys have no meaning without a Native graph.

   [Feature Flow] is in the set: with no Native state the spine holds only
   [s/pt2/000] and no transitions, and rendering a one-node flow graph asserts a
   navigability that does not exist. [Source] is not — decoding succeeded, which
   is the whole of what it claims. [Loop_ir] and [Codegen] are not, because they
   are permanently unimplemented and nothing about this input changed that. *)
let depends_on_lowering (k : C.key) =
  match k with
  | C.Graph_stage C.Source -> false
  | C.Graph_stage _ -> true
  | C.Feature C.Loop_ir | C.Feature C.Codegen -> false
  | C.Feature C.Expression_detail -> false
  | C.Feature _ -> true

let propagate ~lowering_available caps =
  if lowering_available then caps
  else
    List.map
      (fun (c : C.t) ->
        match c.C.status with
        (* [Not_requested] takes precedence: a feature nobody asked for was not
           blocked by anything, and reporting it as blocked invents a
           dependency the caller never exercised. *)
        | C.Not_requested -> c
        | _ when not (depends_on_lowering c.C.key) -> c
        | _ ->
            {
              c with
              C.status =
                C.Unavailable
                  { reason = C.Prerequisite_unavailable; detail = None };
            })
      caps
