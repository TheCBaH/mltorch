(* Packaging a rewrite as a reusable pass, and running a pipeline of them.
   See .ai/native_transform_design.md §5.

   [t.run] is explicitly polymorphic because a pass must work at every version:
   the state's tag is minted per [Rewrite.origin] and rebound by every [Step], so
   a pass fixed to one version could be used exactly once. *)

open Graph_ir

(* --- Dialect-independent, and therefore OUTSIDE the functor.

   [Make] is applied twice — as [Make (Native_side)] below, and in
   native4d/framework.ml as [Pass4] — and records declared inside a functor are
   DISTINCT TYPES in each application. A counter declared inside could not merge
   a Native audit with a Native4D one, which is what a summary over a whole
   pipeline has to do. [Map_verify] is the precedent in this dependency:
   [Report], [Entry], [Outcome], [Verdict], [Coverage], [Tally] and [Policy] all
   sit outside [Make_pair], which is why [Audit.t] can carry a
   [Map_verify.Report.t] at all.

   [Pass4] may re-export ALIASES of these for ergonomics — never fresh
   declarations, which would reintroduce exactly the split this removes. *)

module Frame : sig
  (* NESTED, not a path plus one scalar: pipeline.ml's [relayout] is
     [fixpoint (sequence "relayout" [ fixpoint chain_permute; ... ])], so a leaf
     runs under a stack of composites and an iteration number belongs to the
     composite that iterates, not to the leaf. *)
  type t = { name : string; iteration : int option }

  val pp : Format.formatter -> t -> unit
end

module Count_overflow : sig
  type counter = Audit_reports | Outcome_bucket of string | Execution_index
  type t = { counter : counter }

  val pp : Format.formatter -> t -> unit
end

type count_error = [ `Count_overflow of Count_overflow.t ]

module Outcome_counts : sig
  (* Counts by outcome label, sharing ONE vocabulary with [Map_verify.Tally]:
     both key on [Map_verify.Outcome.label]. A four-counter shape could not
     represent [Tested Agrees] or [Tested Disagrees], and folding a [Sampled n]
     proof into "proved" is the exact overstatement coverage exists to prevent.

     [int64] counts with a merge, neither of which [Tally] has — which is why
     this lives here rather than as an extension of [Map_verify]. *)
  type t

  val empty : t

  val bindings : t -> (string * int64) list
  (** Canonically ordered: base verdict in [Verdict.labels] order, then sample
      count numerically. Deterministic on both backends. *)

  val add : t -> Map_verify.Outcome.t -> (t, [> count_error ]) Err.t
  val of_report : Map_verify.Report.t -> (t, [> count_error ]) Err.t
  val merge : t -> t -> (t, [> count_error ]) Err.t

  module Invalid : sig
    type kind =
      | Negative_count
      | Duplicate_label
      | Malformed_label  (** not a [Verdict] label, optionally [sampled <n>] *)
      | Label_too_long

    type t = { label : string; kind : kind }
  end

  type invalid = [ `Invalid_counts of Invalid.t ]

  val of_bindings : (string * int64) list -> (t, [> invalid ]) Err.t
  (** THE way back in, and why [t] can be abstract while crossing a wire.
      Replaying [add] is not an inverse: the wire carries labels and counts, not
      [Map_verify.Outcome.t] values, and a separate compilation unit seeing only
      this signature could not rebuild one. It is also what lets a test seed a
      bucket at [Int64.max_int] — without which neither overflow check above
      could ever be shown to fire.

      Accepts a label iff it is [v], or [v ^ " [sampled " ^ d ^ "]"], for [v] in
      [Map_verify.Verdict.labels] and [d] a decimal in [1 .. Int64.max_int] with
      no leading zero. [d] is validated LEXICALLY and never converted: parsing
      through [int] would wrap under js_of_ocaml before the comparison. *)

  val pp_invalid : Format.formatter -> [< invalid ] -> unit

  val jsont : t Jsont.t
  (** Encodes and decodes THROUGH [of_bindings], so a malformed binding cannot
      become an unchecked map. An ARRAY of [{label, count}], because the
      canonical order is part of the determinism claim and a JSON object would
      be reordered by whatever map rebuilt it; [count] is
      [Jsont.int64_as_string], never the adaptive [Jsont.int64].

      A [Jsont.t] has NO typed error channel ([Jsont_bytesrw.decode_string]
      returns [('a, string) result]), so decoding surfaces a Jsont message
      rather than [`Invalid_counts]. OCaml callers get the typed row from
      [of_bindings]. *)
end

(** UNIQUE WITHIN ONE TRACE SCOPE ONLY. [Make] is applied per dialect and each
    application seeds its ordinals from zero, while [per_node ~name] takes an
    arbitrary string — so a Native leaf and a Native4D leaf can hold EQUAL
    values. Hoisting this out of the functor made the two comparable; it did not
    make them distinct. Any cross-dialect index must qualify by scope. *)
module Exec_id : sig
  (* Authoritative PASS-EXECUTION provenance. NOT a state identity: [N] changed
     steps produce [N + 1] states, and there is no execution at all for an
     import, a cross-dialect conversion, or a run in which no pass changed
     anything. *)
  type t = { frames : Frame.t list; leaf : string; index : int64 }

  val pp : Format.formatter -> t -> unit

  val next_ordinal : int64 -> (int64, [> count_error ]) Err.t
  (** The ONLY way an ordinal advances: checks the boundary BEFORE adding, so no
      caller is left with [Int64.add index 1L] and a look at the result. *)
end

module Audit : sig
  (* What verifying one pass's step found. Only produced for an ACCEPTED report
     — a rejected one becomes a [`Verification] error instead.

     Keyed by [Exec_id.t] rather than a bare pass name: a [fixpoint] contributes
     one audit per iteration and a [sequence] may hold the same leaf twice, so a
     name alone does not say which execution a report belongs to. *)
  type t = { id : Exec_id.t; report : Map_verify.Report.t }
end

module Audit_summary : sig
  type t = { omitted_reports : int64; counts : Outcome_counts.t }

  val empty : t

  val add : t -> Map_verify.Report.t -> (t, [> count_error ]) Err.t
  (** RESULT-VALUED: the aggregate is [int64] and every increment is checked
      BEFORE the addition, so an overflow is reported rather than inspected
      after the fact. [omitted_reports] and each outcome bucket are checked
      independently, so whichever overflowed names itself. *)
end

module Audit_log : sig
  (* GENUINELY bounded: at most [max_reports] cells plus at most ONE aggregate.
     Retaining a summary per overflowing leaf would shrink the payload while
     leaving the CARDINALITY unbounded — one cell, one id and one tally per
     changed leaf — which is not a bound. *)
  type t = { reports : Audit.t list; overflow : Audit_summary.t option }

  val empty : t
  val retained : t -> int

  val omitted : t -> int64
  (** [0L] when nothing overflowed. *)

  val push : max_reports:int -> t -> Audit.t -> (t, [> count_error ]) Err.t
  (** Enforced BEFORE retention: a limit applied while PROJECTING an
      already-built log arrives after the memory is gone. *)

  val omit : t -> (t, [> count_error ]) Err.t
  (** Record one deliberately skipped audit. Its outcome counts are unknown,
      unlike an audit dropped only after verification completed. *)

  val concat : max_reports:int -> t -> t -> (t, [> count_error ]) Err.t
end

module Make (S : Side.S) : sig
  type node = S.op Graph_common.Node.t

  module Verification : sig
    (* A verifier that errors and a verifier whose report the policy rejects are
       different failures, and both name the pass, since verifying per step exists
       to say WHICH rewrite is at fault. *)
    type problem = Error of Map_verify.error | Rejected of Map_verify.Report.t
    type t = { pass : string; problem : problem }

    val pp : Format.formatter -> t -> unit
  end

  (* [count_error] and [`Malformed_outcome] widen a row that is otherwise
     closed: without them a [let*] over [Audit_summary.add] or
     [Exec_id.next_ordinal] inside [run] would not typecheck against it.
     [Native_interp] already carries [`Transform of Pass.error], so this reaches
     the interpreter's result with no change there. *)
  type error =
    [ Rewrite.Make(S).error
    | `Not_converged of string
    | `Verification of Verification.t
    | count_error
    | `Malformed_outcome of string ]

  val pp_error : Format.formatter -> [< error ] -> unit

  (* What verification a run is under. Threaded THROUGH the pass tree rather than
     applied at its boundary, because a composite verified only at the boundary is
     barely verified: a [fixpoint] iteration or a [sequence] member can be wrong
     and cancelled by a later one, and the error would name the composite rather
     than the pass at fault. *)
  module Trace : sig
    (* Per-execution detail: the state a pass ran against and the step it
       produced. INSIDE the functor, unlike everything above, because the
       payload mentions [Rewrite.Make(S).t] — this is the one part of an outcome
       that is genuinely dialect-specific, and the exporter consumes it through
       two already-functorised instantiations rather than making the two
       traces meet. *)
    module Entry : sig
      module Payload : sig
        (* PARAMETERISED so it can be a named module outside the construct that
           quantifies over it: ['v] is bound at [t] below, so the existential is
           exactly as opaque as an inline record would make it. *)
        type 'v t = {
          id : Exec_id.t;
          before : 'v Rewrite.Make(S).t;
          step : 'v Rewrite.Make(S).step;
        }
      end

      type t = Entry : 'v Payload.t -> t
    end

    type t = { entries : Entry.t list; truncated : bool }
    (** [List.length entries <= max_trace_entries]. [truncated] so a reader is
        never handed a partial trace that looks whole. *)

    val empty : t
    val length : t -> int
  end

  (* [frames] and [index] are the traversal position, threaded DOWN; the advance
     comes back up through [outcome.next_index], because this record is
     immutable and [run_with] hands the same one to every pass.

     Both budgets are enforced BEFORE retention. Transformation continues past
     either — the final graph and the composed report stay available, and only
     the per-step detail is lost — so exhausting one degrades what can be shown
     rather than failing the run. *)
  type ctx = {
    budget : Map_verify.Budget.t option;
    policy : Map_verify.Policy.t option;
    probe : int option;
    max_verified_steps : int option;
    trace : bool;
    max_trace_entries : int;
    max_audit_reports : int;
    frames : Frame.t list;
    index : int64;
  }

  val no_verification : ctx

  (* A step plus its audits. They are carried through the pass tree rather than
     pushed to a callback, so observing a run needs no mutable state; [run_all]
     drops them for the callers that do not want them.

     [next_index] is the ordinal AFTER this pass. A leaf that changed something
     consumes one; a converged identity sweep consumes none, which keeps
     ordinals dense over the executions that happened rather than the attempts.

     The trace and verification guarantees hold only for passes built through
     [per_node], [of_pattern], [sequence] and [fixpoint]. [run_with] receives a
     [t] as [{name; run}] and cannot tell a leaf from a composite — a composite
     legitimately advances by many leaves, and a hand-built pass can fabricate a
     plausible prefix — so it performs STRUCTURAL SANITY ONLY (the ordinal must
     not go backwards) and reports [`Malformed_outcome], never a claim about
     intent. The per-leaf and exact-frame rules live in the private
     [of_sweep]. *)
  type 'v outcome = {
    audits : Audit_log.t;
    trace : Trace.t;
    next_index : int64;
    step : 'v Rewrite.Make(S).step;
  }

  type t = {
    name : string;
    run : 'v. ctx -> 'v Rewrite.Make(S).t -> ('v outcome, error) Err.t;
  }

  (* What a pass sees. Both halves come from the state, and the payloads are the
     state's CUMULATIVE map, so a fold running under a [fixpoint] sees what earlier
     iterations produced. A callback handed only a graph could not express constant
     folding at all — it has to evaluate. *)
  type env = {
    constant_store : Constant_store.t;
    constants : Tensor.packed Tensor_id.Map.t;
    view : Graph_view.Make(S.Dialect).t;
  }

  (* Visit each node and offer a rewrite. The callback returns a BUILDER rather
     than a recipe, so the driver — not the pass — threads the allocator and keeps
     allocation sequential and contiguous across the matches it accepts. *)
  (* RANK-2, for the same reason [t.run] is: the version a recipe is planned at
     comes from the state the driver holds, and a pass has to work at whichever one
     that is. It is also what stops a pass from closing over a source edge resolved
     against some other version. *)
  type per_node = {
    on_node : 'v. env -> node -> ('v, unit) Recipe.Make(S).t option;
  }

  val per_node : name:string -> per_node -> t

  type 'a builder = {
    build : 'v. 'a -> Region.Make(S.Dialect).t -> ('v, unit) Recipe.Make(S).t;
  }

  (* Match with a pattern anchored at each node output; [build] turns an accepted
     match into a builder. Neither takes the env: [Pattern] is defined over the
     view alone, and no pattern-based pass has needed payloads. One that does
     should grow the env here rather than reach around the driver. *)
  val of_pattern :
    name:string ->
    pattern:(Tensor_id.t -> 'a Pattern.Make(S.Dialect).t) ->
    build:'a builder ->
    t

  (* Re-run until a sweep changes nothing. [max_iters] bounds a pass that keeps
     finding work — a rewrite that reintroduces its own match would otherwise spin
     — and exhausting it is an error rather than a silent stop, since the result
     would depend on the bound. *)
  val fixpoint : ?max_iters:int -> t -> t

  (* Compose a fixed list of passes into one, so a caller can [fixpoint] the
     whole group as a unit — needed when the passes unlock each other, and no
     single non-interleaved round through them all is enough. *)
  val sequence : name:string -> t list -> t

  (* [verify] checks each step against the state it came from, as it is applied —
     including every [fixpoint] iteration and every [sequence] member, not just
     the composite they hand back — so the first offending pass stops the pipeline
     and the error names it. An identity step is skipped: its map is empty, and on
     a real graph checking every cluster of a no-op sweep dominates the cost.
     Reports for accepted steps are dropped; a caller that wants every report
     calls [Map_verify.step] itself. *)
  val run_all :
    ?verify:Map_verify.Policy.t ->
    ?verify_budget:Map_verify.Budget.t ->
    ?verify_probe:int ->
    ?max_verified_steps:int ->
    'v Rewrite.Make(S).t ->
    t list ->
    ('v Rewrite.Make(S).step, error) Err.t

  (* [run_all] keeping the audits, one per pass that actually rewrote something —
     a pass whose sweep matched nothing produces an identity step, which has
     nothing to verify. Order follows execution, and a [fixpoint] contributes one
     per iteration, so a caller can show where verification did and did not
     reach. *)
  val run_reporting :
    ?verify:Map_verify.Policy.t ->
    ?verify_budget:Map_verify.Budget.t ->
    ?verify_probe:int ->
    ?max_verified_steps:int ->
    ?trace:bool ->
    ?max_trace_entries:int ->
    ?max_audit_reports:int ->
    'v Rewrite.Make(S).t ->
    t list ->
    ('v outcome, error) Err.t
end

include module type of Make (Native_side)
