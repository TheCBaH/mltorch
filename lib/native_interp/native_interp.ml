(* Pure lowering of the static tensor subset of an ExportedProgram.

   This file is now a thin facade: the actual implementation lives in
   native_interp_error.ml (error payload types and printers),
   native_interp_decode.ml (argument decoding and shape/config helpers,
   internal to [lower]), native_interp_lower.ml ([lower]/[lower_archive]/
   [tensor_of_pt2]), and native_interp_exec.ml ([run] and the
   transform/verify/evaluate pipeline)..

   native_interp.mli is unchanged by the split: every item below is a
   manifest alias, so the public [Native_interp.*] surface is exactly what it
   was before. *)

type arg_kind = Native_interp_error.arg_kind

module Expected_rank = Native_interp_error.Expected_rank

type dim_fault = Native_interp_error.dim_fault
type metadata_role = Native_interp_error.metadata_role
type hw_param = Native_interp_error.hw_param
type config_param = Native_interp_error.config_param
type config_fault = Native_interp_error.config_fault
type unsupported_option = Native_interp_error.unsupported_option
type unsupported_input = Native_interp_error.unsupported_input

module Missing_arg = Native_interp_error.Missing_arg
module Wrong_arg_kind = Native_interp_error.Wrong_arg_kind
module Unresolved_sym_arg = Native_interp_error.Unresolved_sym_arg
module Bad_dimension = Native_interp_error.Bad_dimension
module Missing_metadata = Native_interp_error.Missing_metadata
module Axis_out_of_range = Native_interp_error.Axis_out_of_range
module Bad_arity = Native_interp_error.Bad_arity
module Adaptive_pool_rank = Native_interp_error.Adaptive_pool_rank
module Bad_config = Native_interp_error.Bad_config
module Normalized_rank = Native_interp_error.Normalized_rank
module Normalized_shape = Native_interp_error.Normalized_shape
module Live_layer_norm_stats = Native_interp_error.Live_layer_norm_stats
module Unsupported_option = Native_interp_error.Unsupported_option
module Output_arity = Native_interp_error.Output_arity
module Bad_view = Native_interp_error.Bad_view
module Bad_slice = Native_interp_error.Bad_slice
module Bad_select = Native_interp_error.Bad_select
module Concat_rank_mismatch = Native_interp_error.Concat_rank_mismatch
module Bad_upsample_size = Native_interp_error.Bad_upsample_size

type malformed = Native_interp_error.malformed

module Rank_mismatch = Native_interp_error.Rank_mismatch
module Storage_range = Native_interp_error.Storage_range

type tensor_bridge = Native_interp_error.tensor_bridge
type error = Native_interp_error.error

type hooks = Native_interp_exec.hooks =
  | Hooks : {
      on_start : Pt2_native_graph.t -> Graph_ir.node -> 'a;
      on_end : Pt2_native_graph.t -> Graph_ir.node -> 'a -> unit;
    }
      -> hooks

let pp_error = Native_interp_error.pp_error
let pp_malformed = Native_interp_error.pp_malformed
let pp_tensor_bridge = Native_interp_error.pp_tensor_bridge
let lower = Native_interp_lower.lower
let lower_archive = Native_interp_lower.lower_archive
let run = Native_interp_exec.run

type transformed = Native_interp_exec.transformed =
  | Transformed : {
      constants : Tensor.packed Graph_ir.Tensor_id.Map.t;
      constant_store : Constant_store.t;
      derived : (Graph_ir.Tensor_id.t * string list) list;
      graph : Graph_ir.graph;
      lens : 'b Pt2_native_graph.lens;
      nodes_before : int;
      audits : Pass.Audit_log.t;
      composed : Map_verify.Report.t option;
    }
      -> transformed

let preload = Native_interp_exec.preload
let transform = Native_interp_exec.transform
let transform_lowered = Native_interp_exec.transform_lowered

type loaded = Native_interp_exec.loaded = {
  from_state : int;
  from_archive : int;
  from_plan : int;
}

let evaluate = Native_interp_exec.evaluate
let tensor_of_pt2 = Native_interp_lower.tensor_of_pt2
let capture_resolver = Native_interp_exec.capture_resolver
