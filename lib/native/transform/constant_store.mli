(* The graph-facing part of deferred constants.  Plan value ids remain stable
   while graph tensor ids may be compacted or removed. *)

open Graph_ir

type error =
  [ `Binding_overwrite of Tensor_id.t
  | Const_ssa.error
  | `Missing_export of Const_ssa.Value_id.t ]

type t

val empty : t
val materialized : t -> Tensor.packed Tensor_id.Map.t
val plan : t -> Const_ssa.t
val binding : t -> Tensor_id.t -> Const_ssa.Value_id.t option
val bindings : t -> (Tensor_id.t * Const_ssa.Value_id.t) list
val is_effective_constant : t -> Tensor_id.t -> bool

val bind_captured :
  t -> tensor:Tensor_sig.t -> Const_ssa.Capture.t -> (t, error) Err.t

val bind_literal : t -> tensor:Tensor_sig.t -> Tensor.packed -> (t, error) Err.t
val bind_apply : t -> tensor:Tensor_sig.t -> Graph_ir.op -> (t, error) Err.t

val bind_materialized :
  t -> tensor:Tensor_sig.t -> Tensor.packed -> (t, error) Err.t

val restrict_and_rename_exports :
  t -> (Tensor_id.t -> Tensor_id.t option) -> (t, error) Err.t

val pp_error : Format.formatter -> [< error ] -> unit
val pp : Format.formatter -> t -> unit
