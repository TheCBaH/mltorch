(** Generic Machine IR to selected x86-64, through the shared builder: every
    generic value keeps its id and type; a failure becomes the model record's
    stores and status 1; a return status 0; a call that may fail is followed by
    a test of its status. Baseline is SSE2: [Ffma] needs the FMA feature and a
    truncation SSE4.1, otherwise each is a typed refusal. *)

open Machine_ir

module Refusal : sig
  type t =
    | Fallible_with_results of Mir_op.Callee.t
    | Feature of Mir_target.Feature.t
        (** an operation only an extension expresses *)
    | Invalid_selection of Mir_diagnostic.t
    | Missing_site of Mir_failure.t
    | Operation of string
    | Vector of Mir_vsplit.Refusal.t
        (** a vector the split into register-wide slices refuses *)
    | Width of Mir_type.t

  val pp : Format.formatter -> t -> unit
end

(** Fault injection for the evidence suite. No consumer passes one. *)
module Mutation : sig
  type t =
    | Commuted_sub
        (** a subtraction's left constant folded as if it were the right *)
    | Contract  (** a separate multiply and add selected as one FMA *)
    | Division_swap  (** IDIV's dividend and divisor exchanged *)
    | Fused_float_eq
        (** a branch on ordered float equality fused as one [je] *)
    | Max_no_nan  (** IEEE maximum without its NaN repair *)
    | Missing_failure_word
        (** the last word a record's kind defines not stored *)
    | No_parity  (** ordered equality tested by ZF alone *)
    | Pruned_live  (** values only a terminator reads pruned as unread *)
    | Scaled_address  (** a folded index scaled by twice its element size *)
    | Signed_compare  (** an unsigned compare tested with a signed condition *)
end

type result = { selected : X64_stage.Sel.Verified.t; record : Mir_id.View.t }

val program :
  ?mutation:Mutation.t ->
  ?sites:Mir_failure.Site_entry.t array ->
  ?unlisted:Mir_failure.Unlisted.t ->
  ?features:Mir_target.Feature.t list ->
  Mir_verify.Generic.t ->
  (result, Refusal.t) Err.t
(** [features] defaults to the baseline, [[Sse2]]. [unlisted] (default
    [Refused]) is what a site-bearing failure [sites] has no entry for means: a
    refusal, or the sentinel site. *)
