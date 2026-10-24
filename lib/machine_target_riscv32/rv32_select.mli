(** Generic Machine IR to selected RV32IMF. The generic program keeps its LP64
    model; the selected one is ILP32: a pointer is one 32-bit register, and a
    [ptr.add]'s i64 offset contributes its low word (a pointer plus a word wraps
    modulo 2^32, which is the signed offset for every address in an object below
    2 GiB). Every other i64 value becomes a pair of i32 values — named before
    selection so a use finds them in any block order — and every i64 operation a
    sequence on the halves; a half nothing reads is pruned. Each other generic
    value keeps its id and type; temporaries are fresh. A generic [fail] becomes
    stores of the model failure record into a runtime view and a status return
    of 1; a return gains a zero status. Vectors are cut into slices of
    [register_bytes] ({!Mir_vsplit}; default 16, four lanes, or 64, sixteen
    lanes in a group of four registers) and selected as Zve32f forms; a full i64
    converts to binary32 with one rounding in word arithmetic. Binary64, an i64
    division whose operands are not sign-extended words by a constant, and an
    i64 call argument are typed refusals. *)

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
    | Contract  (** a separate multiply and add selected as one [fmadd.s] *)
    | Dropped_carry  (** an i64 addition's carry into the high word dropped *)
    | Missing_failure_word
        (** the last word a record's kind defines not stored *)
    | Number_max  (** IEEE maximum selected as [fmax.s] (maximumNumber) alone *)
    | Pruned_live  (** values only a terminator reads pruned as unread *)
    | Signed_compare  (** an i64 compare's low words compared as signed *)
    | Unguarded_trunc
        (** truncation through a 32-bit integer at every magnitude *)
end

type result = {
  selected : Rv32_stage.Sel.Verified.t;
  record : Mir_id.View.t;  (** the failure record's view *)
}

val program :
  ?mutation:Mutation.t ->
  ?register_bytes:int64 ->
  ?sites:Mir_failure.Site_entry.t array ->
  ?unlisted:Mir_failure.Unlisted.t ->
  Mir_verify.Generic.t ->
  (result, Refusal.t) Err.t
