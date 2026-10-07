(** Generic Machine IR to selected AArch64. Each generic value keeps its id and
    type; temporaries are fresh. A generic [fail] becomes explicit stores of the
    model failure record into a runtime view and a nonzero status return; a
    return becomes a zero status. The main function's results gain a trailing
    [i32] status — every function's results do. A call becomes [bl] with its
    arguments and results in fixed registers; a call that may fail is followed
    by a branch on its status, whose taken side returns that status with the
    callee's record untouched. Vectors are first split into register-wide slices
    ({!Mir_vsplit}) and selected as Advanced SIMD forms on Q (4S, 2D) and D (2S)
    registers. Anything outside the admitted slice is a typed refusal. *)

open Machine_ir

module Refusal : sig
  type t =
    | Fallible_with_results of Mir_op.Callee.t
        (** a call that may fail returning results, or from a function that has
            results: neither result can be left undefined in this slice *)
    | Invalid_selection of Mir_diagnostic.t
        (** the selected program its verifier rejects: a selection defect *)
    | Missing_site of Mir_failure.t
        (** a reachable site-bearing failure with no compatible table entry *)
    | Operation of string  (** a generic operation, by name, not admitted *)
    | Vector of Mir_vsplit.Refusal.t
        (** a vector the split into register-wide slices refuses *)
    | Width of Mir_type.t  (** a value type this slice keeps in no register *)

  val pp : Format.formatter -> t -> unit
end

(** Fault injection for the evidence suite: each is one deliberate selection
    defect a comparison must detect. No consumer passes one. *)
module Mutation : sig
  type t =
    | Contiguous_lanes  (** a strided vector load read contiguously *)
    | Contract  (** a separate multiply and add selected as one [fmadd] *)
    | Dropped_half  (** a narrowing's upper half left zero *)
    | Fcmp_lt_cond
        (** ordered less-than tested with [lt] (true when unordered) *)
    | Missing_failure_word
        (** the last word a record's kind defines not stored *)
    | Signed_compare  (** an unsigned compare tested with a signed condition *)
end

type result = {
  selected : A64_stage.Sel.Verified.t;
  record : Mir_id.View.t;  (** the failure record's view *)
}

val program :
  ?mutation:Mutation.t ->
  ?sites:Mir_failure.Site_entry.t array ->
  ?unlisted:Mir_failure.Unlisted.t ->
  Mir_verify.Generic.t ->
  (result, Refusal.t) Err.t
(** [unlisted] (default [Refused]) is what a site-bearing failure [sites] has no
    entry for means: a refusal, or the sentinel site. *)
