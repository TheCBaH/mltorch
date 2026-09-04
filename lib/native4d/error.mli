(* Why a Native graph is outside the Native4D dialect. A row, not a closed
   variant, because the set is built PROGRESSIVELY: this stage owns the
   domain-check cases, and the later stages union their own rows in as the
   modules that name them arrive (Shape4 in stage 2, the builder in stage 3, the
   lowerer and Graph_map in stage 5). Writing the whole of
   .ai/native4d_design.md §10 here would reference three modules that do not
   exist yet. See .ai/native4d_plan.md.

   Each module owns its own row and callers widen, the idiom [Graph_shape.widen]
   already uses — so [Shape4.of_vec6] will return ITS row rather than this
   aggregate. The reverse would make [Error] depend on [Shape4] and [Shape4]
   depend on [Error].

   These are not failures of Native4D; a graph outside the domain is a graph the
   partial conversion does not claim to handle (design §1). *)

open Graph_ir

type t =
  [ `Axis_outside_dialect of Node_id.t * Axis.t
  | `Bad_constant_payload of Tensor_id.t
  | `Batch_norm_extent of
    Node_id.t * Tensor_id.t * Dim.extent Dim.t * Dim.extent Dim.t
  | `Batched_matmul_batch_axis of Node_id.t
  | `Constant_store of Constant_store.error
  | `Dynamic_batch_norm of Node_id.t
  | `Live_max_pool_indices of Node_id.t * Tensor_id.t
  | `Map of Graph_map.error
  | `Missing_constant_payload of Node_id.t * Tensor_id.t
  | `Non_four_dimensional_tensor of Tensor_id.t * Vec6.shape
  | `Sdpa_batch_axis of Node_id.t
  | `Unsupported_grouped_transposed_conv of Node_id.t * int
  | `Unsupported_op of Node_id.t * op
  | `View of Framework.View4.error ]
(* Carries the op, per design §10, not just its name: an op is rejected for
       what it is, and the payload printed through [Graph_ir.pp_op_with] shows
       the parameters that put it outside the dialect. *)

(* [< t] so a caller holding a narrower row can print it without widening. *)
val pp : Format.formatter -> [< t ] -> unit
