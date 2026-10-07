(** What native assembly receives: a realized physical program that has passed
    the physical verifier and the symbolic checker again at publication, with
    everything a printer, assembler and loader need that the instructions do not
    say — the target and features, the planning identity, the helpers and their
    native bindings, every data and function symbol, every relocation an
    instruction encodes, and each instruction's origin. Printing, assembling and
    loading stay downstream; an artifact exists only through {!Make}'s
    [publish]. *)

open Machine_ir

(** Where a data symbol's bytes come from. *)
module Section : sig
  type t =
    | Bound
        (** supplied by the host at each invocation: a symbol the artifact
            references and the host defines *)
    | Bss  (** the artifact's own storage, undefined at each invocation *)
    | Rodata of string  (** the artifact's own constant bytes *)

  val name : t -> string
end

module Symbol : sig
  type kind =
    | Data of {
        region : Mir_id.Region.t;
        size : int64;
        align : int64;
        section : Section.t;
      }
    | External_function of string
        (** a helper's native binding: the C library symbol it links *)
    | Function of Mir_id.Func.t  (** defined by the artifact's text *)

  type t = { name : string; kind : kind }
end

module Relocation : sig
  type t = {
    func : Mir_id.Func.t;
    block : Mir_id.Block.t;
    instr : Mir_id.Instr.t option;  (** [None] for a late form *)
    reference : Mir_target.Reference.t;
    symbol : string;
    addend : int64;  (** the view's offset in its region *)
  }
end

module Identity : sig
  type t = {
    target : string;
    source : Mir_target.Source.t;
    features : Mir_target.Feature.t list;
    planning : Mir_planning.t;
    helpers : Mir_helper.t list;
  }
end

type ('op, 'test) t

val identity : (_, _) t -> Identity.t
val symbols : (_, _) t -> Symbol.t list
val relocations : (_, _) t -> Relocation.t list
val program : ('op, 'test) t -> ('op, 'test) Mir_phys.Program.t

val origins : (_, _) t -> (Mir_id.Instr.t * Mir_origin.t) list
(** every executed instruction's origin, in program order *)

val pp_summary : Format.formatter -> (_, _) t -> unit
(** The identity, symbols and relocation counts, for a report. *)

module Make (T : Mir_sel.TARGET) : sig
  val publish :
    planning:Mir_planning.t ->
    Mir_sel.Make(T).Verified.t ->
    (T.op, T.test) Mir_phys.Program.t ->
    ((T.op, T.test) t, string) result
  (** Refuses a program with an unrealized frame, one the physical verifier or
      the checker (against its selected program) rejects, and one calling a
      helper with no native binding: interpreter-only helper support cannot
      complete a native artifact. *)
end
