(** Instruction scheduling inside a selected block, checked against the order it
    replaced. *)

open Machine_ir

module Policy : sig
  type t =
    | Reverse
        (** the latest ready instruction first: the legal order farthest from
            the source's, for the evidence suite *)
    | Sink
        (** a pure instruction whose results stay in its block moved down to
            just before its first dependent, when that lengthens no operand's
            life *)
    | Source  (** the selected order, unchanged *)

  val name : t -> string
end

(** Fault injection for the evidence suite: each drops one class of dependence
    from the graph the scheduler orders by. No consumer passes one. *)
module Mutation : sig
  type t =
    | Data  (** a use not ordered after its definition *)
    | Flags  (** a flags writer free to enter a [Flags] value's range *)
    | Order  (** the order chain *)
end

(** Where a scheduled block breaks the order it replaced. *)
module Violation : sig
  type t =
    | Not_permutation of Mir_id.Block.t
    | Reversed of {
        block : Mir_id.Block.t;
        first : Mir_id.Instr.t;  (** what must come first *)
        second : Mir_id.Instr.t;
      }

  val pp : Format.formatter -> t -> unit
end

module Refusal : sig
  type t = Dependence of Violation.t | Invalid of Mir_diagnostic.t

  val pp : Format.formatter -> t -> unit
end

module Make (T : Mir_sel.TARGET) : sig
  module S : module type of struct
    include Mir_sel.Make (T)
  end

  val reorder : ?mutation:Mutation.t -> Policy.t -> S.Verified.t -> S.selected
  (** The reordered program, unchecked. *)

  val schedule :
    ?mutation:Mutation.t ->
    Policy.t ->
    S.Verified.t ->
    (S.Verified.t, Refusal.t) result
end
