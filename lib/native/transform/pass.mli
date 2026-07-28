(* Packaging a rewrite as a reusable pass, and running a pipeline of them.
   See .ai/native_transform_design.md §5.

   [t.run] is explicitly polymorphic because a pass must work at every version:
   the state's tag is minted per [Rewrite.origin] and rebound by every [Step], so
   a pass fixed to one version could be used exactly once. *)

open Graph_ir

module Verification : sig
  (* A verifier that errors and a verifier whose report the policy rejects are
     different failures, and both name the pass, since verifying per step exists
     to say WHICH rewrite is at fault. *)
  type problem = Error of Map_verify.error | Rejected of Map_verify.Report.t
  type t = { pass : string; problem : problem }

  val pp : Format.formatter -> t -> unit
end

type error =
  [ Rewrite.error | `Not_converged of string | `Verification of Verification.t ]

val pp_error : Format.formatter -> [< error ] -> unit

(* What verification a run is under. Threaded THROUGH the pass tree rather than
   applied at its boundary, because a composite verified only at the boundary is
   barely verified: a [fixpoint] iteration or a [sequence] member can be wrong
   and cancelled by a later one, and the error would name the composite rather
   than the pass at fault. *)
type ctx = {
  budget : Map_verify.Budget.t option;
  policy : Map_verify.Policy.t option;
  probe : int option;
}

val no_verification : ctx

module Audit : sig
  (* What verifying one pass's step found. Only produced for an ACCEPTED
     report — a rejected one becomes a [`Verification] error instead. *)
  type t = { pass : string; report : Map_verify.Report.t }
end

(* A step plus its audits. They are carried through the pass tree rather than
   pushed to a callback, so observing a run needs no mutable state; [run_all]
   drops them for the callers that do not want them. *)
type 'v outcome = { audits : Audit.t list; step : 'v Rewrite.step }

type t = {
  name : string;
  run : 'v. ctx -> 'v Rewrite.t -> ('v outcome, error) Core.result;
}

(* What a pass sees. Both halves come from the state, and the payloads are the
   state's CUMULATIVE map, so a fold running under a [fixpoint] sees what earlier
   iterations produced. A callback handed only a graph could not express constant
   folding at all — it has to evaluate. *)
type env = { constants : Tensor.packed Tensor_id.Map.t; view : Graph_view.t }

(* Visit each node and offer a rewrite. The callback returns a BUILDER rather
   than a recipe, so the driver — not the pass — threads the allocator and keeps
   allocation sequential and contiguous across the matches it accepts. *)
type per_node = { on_node : env -> node -> unit Recipe.t option }

val per_node : name:string -> per_node -> t

(* Match with a pattern anchored at each node output; [build] turns an accepted
   match into a builder. Neither takes the env: [Pattern] is defined over the
   view alone, and no pattern-based pass has needed payloads. One that does
   should grow the env here rather than reach around the driver. *)
val of_pattern :
  name:string ->
  pattern:(Tensor_id.t -> 'a Pattern.t) ->
  build:('a -> Region.t -> unit Recipe.t) ->
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
  'v Rewrite.t ->
  t list ->
  ('v Rewrite.step, error) Core.result

(* [run_all] keeping the audits, one per pass that actually rewrote something —
   a pass whose sweep matched nothing produces an identity step, which has
   nothing to verify. Order follows execution, and a [fixpoint] contributes one
   per iteration, so a caller can show where verification did and did not
   reach. *)
val run_reporting :
  ?verify:Map_verify.Policy.t ->
  ?verify_budget:Map_verify.Budget.t ->
  ?verify_probe:int ->
  'v Rewrite.t ->
  t list ->
  ('v outcome, error) Core.result
