type error =
  [ `Missing_capture of Const_ssa.Capture.t
  | `Direct of Eval_direct.error
  | `Signature_mismatch of Const_ssa.Value_id.t
  | Constant_store.error ]

type report = { captures : int; applies : int; cache_hits : int }
type resolver = Const_ssa.Capture.t -> (Tensor.packed, error) Err.t

val materialize :
  ?needed:Tensor_id.Set.t ->
  resolver ->
  Constant_store.t ->
  (Constant_store.t * report, error) Err.t

val pp_error : Format.formatter -> [< error ] -> unit
