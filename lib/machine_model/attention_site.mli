(** A selected attention: the descriptor and the identity of each role's tensor
    in the graph. A bundle keeps neither an operation's name nor its operand
    order (a kernel's sources are an id-ordered set, a repeated operand appears
    once), so the descriptor alone cannot tell the query from the key; the roles
    travel with it and are resolved through the invocation's edges. *)

type t = {
  block : Attention_block.t;
  query : Tensor_id.t;
  key : Tensor_id.t;
  value : Tensor_id.t;
  mask : Tensor_id.t option;
  output : Tensor_id.t;
}

(** The SSA buffer each role binds in one invocation. *)
module Buffers : sig
  type t = {
    q : Ssa_ir.Ssa_id.Buffer.t;
    k : Ssa_ir.Ssa_id.Buffer.t;
    v : Ssa_ir.Ssa_id.Buffer.t;
    mask : Ssa_ir.Ssa_id.Buffer.t option;
    o : Ssa_ir.Ssa_id.Buffer.t;
  }

  val pp : t Fmt.t
end

(** Why a role has no buffer. Alphabetical. *)
module Unbound : sig
  type t =
    | Not_an_output of Tensor_id.t
        (** the output role names a tensor the program reads *)
    | Role of { role : string; tensor : Tensor_id.t }
        (** a role whose tensor no buffer of the invocation binds *)

  val pp : t Fmt.t
end

val buffers :
  t -> Loop_ir.Loop_bundle.invocation -> (Buffers.t, Unbound.t) result
(** Each role's buffer: the one whose position in the program's buffers holds
    the role's tensor among the invocation's edges. Several roles may resolve to
    one buffer (a query that is also the key). *)

val program :
  t ->
  Buffers.t ->
  Ssa_ir.Ssa_program.t ->
  (Ssa_ir.Ssa_program.t, string) result
(** The program with its body replaced by one {!Ssa_ir.Ssa_op.Attend} over the
    buffers, its buffers unchanged. A program that does not take and return one
    effect is refused. *)
