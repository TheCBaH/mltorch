(* Direct (concrete) evaluation of a native graph. Walks the topo-ordered nodes
   threading an immutable env, runs each op through [Eval_op.Make (Direct)] and
   [Schedule.evaluate]. Structural groups do not affect evaluation. See
   .ai/native_graph_design.md.

   [?retain] decides what the result holds, keyed by edge id:
   - [Release_schedule.Retain.All], the default: EVERY edge's tensor (inputs,
     intermediates, outputs), so callers can print any intermediate.
   - [Only s]: exactly [g.outputs] and those of [s] that were bound. Every other
     edge leaves the env right after its last reader runs, so its payload is
     garbage from then on rather than at the end of the run. Outputs and errors
     are identical to an [All] run.
   [?arena] backs each eligible intermediate with its planned slot instead of a
   fresh allocation. Outputs and retained edges are never eligible, so the
   result holds no arena memory. A plan runs only on the run it was built from:
   at entry this dry-runs the graph under the effective [retain] and requires the
   plan's script to equal it ([`Arena_script_mismatch] otherwise), before any node
   runs. That dry run is under the plan's own alignment policy, the one its
   script was made under. An arena is used by one run at a time ([`Arena_busy]). Results are equal
   bit for bit to a run without one.

   [?trace] is a test hook: it receives the script the run actually follows, read
   from what the run holds rather than from the schedule, so a test can compare it
   with [dry_run]'s.

   [?on_format_mismatch] is a test hook too: it is called for each result whose
   shape, format or quantization differs from its edge's declared signature. The
   run itself is unchanged.

   Neither setting ever holds an index output nothing reads: it is not
   allocated. See .ai/ (tensor release). *)

open Graph_ir

type context = Operand | Sig_shape
type missing_tensor = { context : context; id : Tensor_id.t }
type arity_mismatch = { expected : int; actual : int }

type mixed_dtype = {
  mixed_op : string;
  a_fmt : Payload.packed_fmt;
  b_fmt : Payload.packed_fmt;
}

type scalar_op = { scalar_op : string; fmt : Payload.packed_fmt }

(* An embedding table must be float32 and its indices int64: the only pair the
   gather reads exactly. *)
type embedding_dtype = {
  weight_fmt : Payload.packed_fmt;
  indices_fmt : Payload.packed_fmt;
}

type error =
  [ Arena.error
  | Eval_direct_compute.error
  | Graph_shape.error
  | `Missing_constant of Tensor_id.t
  | `Missing_input of Tensor_id.t
  | `Missing_tensor of missing_tensor
  | `Output_arity_mismatch of arity_mismatch
  | `Quant_missing of Tensor_id.t
  | `Region_construction of Region_computation.error
  | `Region_execution of Region_eval.error
  | `Unsupported_bool_arithmetic of mixed_dtype
  | `Unsupported_bool_scalar_arithmetic of scalar_op
  | `Unsupported_embedding_dtype of embedding_dtype
  | `Unsupported_mixed_dtype of mixed_dtype ]

(** A result whose shape, format or quantization is not its edge's declared
    signature. *)
module Format_mismatch : sig
  type t = {
    node : Node_id.t;
    op : string;
    output : Tensor_id.t;
    declared : Tensor_sig.t;
    actual : Tensor.packed;
  }
end

type hooks =
  | Hooks : { on_start : node -> 'a; on_end : node -> 'a -> unit } -> hooks

val pp_error : Format.formatter -> [< error ] -> unit

val dry_run :
  ?alignment:Alignment_policy.t ->
  ?retain:Release_schedule.Retain.t ->
  graph ->
  (Alloc_script.t, error) Err.t
(** The allocation script [run ?retain] would follow, without computing
    anything: shares [run]'s fold, so which outputs are allocated and when each
    edge is released are decided in one place. Graph inputs and constants are
    bound by the caller and never appear in it. *)

val fresh_synthetic_ids :
  graph -> (Region_computation.synthetic_role * Tensor_id.t) list
(** One id per synthetic-default role, disjoint from every graph tensor: the ids
    a Region-authored node's optional-operand defaults are bound under. *)

val storage_script :
  ?alignment:Alignment_policy.t ->
  ?retain:Release_schedule.Retain.t ->
  Storage_script.Config.t ->
  graph ->
  (Storage_script.t, error) Err.t
(** Every block a run under [config] touches, by role and arena, and when it is
    allocated and freed: [dry_run]'s fold, with the used constants and graph
    inputs added and the run's boundaries marked. See [Storage_script]. *)

val run_storage :
  arenas:Arena.t list ->
  script:Storage_script.t ->
  ?hooks:hooks ->
  ?on_format_mismatch:(Format_mismatch.t -> unit) ->
  ?region_counters:Region_execution.counters Tensor_id.Map.t ->
  ?region_executor:Region_executor.t ->
  ?region_group_executor:Region_executor.group ->
  ?node_executor:Node_executor.t ->
  ?limits:Kernel.Limits.t ->
  ?retain:Release_schedule.Retain.t ->
  ?constants:(Tensor_id.t * Tensor.packed) list ->
  graph ->
  inputs:(Tensor_id.t * Tensor.packed) list ->
  (Tensor.packed Tensor_id.Map.t, error) Err.t
(** A run whose outputs are computed into the slots [arenas] place (the arenas
    of a [Storage_plan] built from [script]), [`Storage_script_mismatch] before
    any node runs when this run's own storage script under [retain] is not
    [script]. Constants and inputs are bound as given: filling their slots is
    the caller's ([Storage_run]), and so is holding the arenas. Unlike [run], a
    result may be a view into an arena. *)

val run :
  ?arena:Arena.t ->
  ?hooks:hooks ->
  ?trace:(Alloc_script.Event.t -> unit) ->
  ?on_format_mismatch:(Format_mismatch.t -> unit) ->
  ?region_counters:Region_execution.counters Tensor_id.Map.t ->
  ?region_executor:Region_executor.t ->
  ?region_group_executor:Region_executor.group ->
  ?node_executor:Node_executor.t ->
  ?limits:Kernel.Limits.t ->
  ?retain:Release_schedule.Retain.t ->
  ?constants:(Tensor_id.t * Tensor.packed) list ->
  graph ->
  inputs:(Tensor_id.t * Tensor.packed) list ->
  (Tensor.packed Tensor_id.Map.t, error) Err.t
