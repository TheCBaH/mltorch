(* Immutable, validated deferred-constant SSA.  This module deliberately has no
   dependency on rewrite state or an archive implementation. *)

open Graph_ir

module Value_id : sig
  type t

  val of_tensor_id : Tensor_id.t -> t
  val to_tensor_id : t -> Tensor_id.t
  val compare : t -> t -> int
  val equal : t -> t -> bool
  val pp : Format.formatter -> t -> unit
end

module Capture : sig
  type t

  val of_string : string -> t
  val to_string : t -> string
  val compare : t -> t -> int
  val equal : t -> t -> bool
  val pp : Format.formatter -> t -> unit
end

type leaf =
  | Captured of Capture.t
  | Literal of Tensor.packed
  | Opaque_materialized of Tensor.packed

type definition =
  | Apply of { op : Graph_ir.op; output : Tensor_sig.t }
  | Leaf of { leaf : leaf; output : Tensor_sig.t }

type error =
  [ `Duplicate_definition of Value_id.t
  | `Invalid_literal of Value_id.t
  | `Missing_operand of Value_id.t * Value_id.t
  | `Output_id_mismatch of Value_id.t * Tensor_id.t
  | `Output_signature_mismatch of Value_id.t
  | `Shape of Graph_shape.error
  | `Unsupported_op of string ]

type t

val empty : t
val add : t -> id:Value_id.t -> definition -> (t, error) Err.t
val find : t -> Value_id.t -> definition option
val sig_of : t -> Value_id.t -> Tensor_sig.t option
val operands : t -> Value_id.t -> Value_id.t list
val validate : t -> (unit, error) Err.t
val bindings : t -> (Value_id.t * definition) list

(* Whether an operation is admitted by the closed Const-SSA language. The
   materialized-fold trace uses this to report the first operation family that
   still needs a symbolic implementation. *)
val allows : Graph_ir.op -> bool
val pp_error : Format.formatter -> [< error ] -> unit
val pp : Format.formatter -> t -> unit
