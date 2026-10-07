(** A mutable builder for generic functions. Every value, block and instruction
    id comes from a bounded counter; result types and order threading are
    derived from the opcode, so a builder cannot declare a store pure, drop an
    operand or invent a result type. The program it returns is unverified: pass
    it to {!Mir_verify.generic}. *)

type t

val create : unit -> t

type block

val block_id : block -> Mir_id.Block.t

val param : block -> Mir_value.t list
(** The block's value parameters, in order. *)

val new_block : t -> Mir_type.t list -> block
(** A block with fresh parameters of these types and a fresh order parameter.
    The first block made is a function's entry unless {!func} says otherwise. *)

val op :
  ?origin:Mir_origin.t ->
  t ->
  block ->
  signature:(Mir_op.Callee.t -> Mir_typing.Signature.t option) ->
  Mir_op.t ->
  (Mir_value.t list, Mir_typing.Error.t) result
(** Appends an operation: its results get fresh ids at the types
    {!Mir_typing.check} derives, and an ordered operation consumes the block's
    current order state and produces the next. *)

val emit : ?origin:Mir_origin.t -> t -> block -> Mir_op.t -> Mir_value.t
(** {!op} for an operation with exactly one result and no callee; raises
    [Invalid_argument] on a typing error (a fixture bug). *)

val emit_unit : ?origin:Mir_origin.t -> t -> block -> Mir_op.t -> unit
(** {!op} for an operation with no result and no callee. *)

val order : block -> Mir_value.t
(** The block's current order state. *)

val jump : block -> block -> Mir_value.t list -> unit

val branch :
  block ->
  Mir_value.t ->
  block * Mir_value.t list ->
  block * Mir_value.t list ->
  unit

val return : block -> Mir_value.t list -> unit

val fail :
  ?origin:Mir_origin.t -> block -> Mir_failure.t -> Mir_value.t list -> unit
(** [origin]: the guard the failure ends, inherited by its expansion *)

val func :
  t ->
  id:Mir_id.Func.t ->
  name:string ->
  entry:block ->
  results:Mir_type.t list ->
  (Mir_op.t, Mir_terminator.t) Mir_func.t
(** The function over every block made since the previous {!func}, entry first.
    Raises [Invalid_argument] if a block has no terminator. *)

val program :
  ?regions:Mir_region.t list ->
  ?views:Mir_view.t list ->
  ?helpers:Mir_helper.t list ->
  ?planning:Mir_planning.t ->
  (Mir_op.t, Mir_terminator.t) Mir_func.t list ->
  main:Mir_id.Func.t ->
  Mir_program.generic
