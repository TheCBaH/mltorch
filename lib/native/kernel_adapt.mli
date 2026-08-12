(* [Stage_program.t] -> [Kernel.t], whole-program or over a selected subset.

   [Stage_program.t] is a public record, so this treats it as UNTRUSTED: it
   validates the whole definition table and every body's budget before it runs a
   single [Expr.Fold] query of its own. That ordering is not cosmetic. Deriving
   boundary inputs and liveness requires [Fold.sources], which recurses over a
   whole tree, and [Kernel.create]'s own budgets-first rule runs far too late to
   help — by then the adapter has already walked the body.

   Every rule here reads EVERY source, never only the loads. [Symbolic.max_pool]
   names its input solely inside the intrinsic descriptor, with no [Load] in the
   body, so a loads-only boundary rule would fail to create a synthetic input
   for a selected max-pool's outside producer and — worse — would fail to mark a
   producer live when its max-pool consumer sits outside the selection, letting
   a later fusion plan drop a store that consumer still needs. *)

(* WHAT THE LIMITS BOUND: both the raw [Stage_program.t] lists and the derived
   kernel interface, at two distinct points.

   Raw first, before any list is traversed — [stages] against [max_values],
   [inputs] + [consts] together against [max_inputs] (both become boundary
   inputs), and [outputs] against [max_outputs]. Bounding only [stages] left the
   adapter doing unbounded work on lists just as public as that one.

   Derived second: [of_stage_program] through [Kernel.create], which counts
   synthetic inputs the raw lists do not contain, and [required_outputs] against
   [max_outputs] itself. The latter is not redundant — that entry point never
   reaches [Kernel.create], so without it the same [limits] would mean different
   things depending on which function a caller used. *)

module Unknown_stage : sig
  type t = { at : Tensor_id.t; source : Expr.Source.t }
end

module Program_error : sig
  type kind = Duplicate_definition | Signature_id | Forward_source
  type t = { at : Tensor_id.t; kind : kind }
end

type error =
  [ Kernel.error
  | `Unknown_stage_source of Unknown_stage.t
  | `Missing_live_output of Tensor_id.t
  | `Output_not_selected of Tensor_id.t
  | `Unknown_selection of Tensor_id.t
  | `Passthrough_output of Tensor_id.t
  | `Unknown_program_output of Tensor_id.t
  | `Program_invalid of Program_error.t ]

val pp_error : Format.formatter -> [< error ] -> unit

val required_outputs :
  ?limits:Kernel.Limits.t ->
  ?select:Tensor_id.Set.t ->
  Stage_program.t ->
  (Tensor_id.t list, error) Err.t
(** The outputs a kernel over this selection MUST declare, ordered:

    1. selected occurrences of [Stage_program.outputs], in their original order
    and INCLUDING repeats — that list is the program's signature and its arity
    is meaningful; 2. then selected values any unselected stage depends on, in
    stage order, skipping ids already present; 3. then otherwise-dead selected
    terminals, likewise.

    Dead terminals are promoted rather than sliced away. A [Stage_program] may
    legitimately contain a stage nothing consumes — [Eval_symbolic.run] emits
    one per node output while a [Discard] node emits none, so in
    [Graph_fixtures.multi_output] the max-pool index stage has no consumer — and
    [Stage_program.ground] materialises every stage. Dropping them would change
    behaviour under the guise of adapting it; eliminating dead stages is a
    graph/DCE concern. *)

val of_stage_program :
  ?limits:Kernel.Limits.t ->
  ?select:Tensor_id.Set.t ->
  ?outputs:Tensor_id.t list ->
  Stage_program.t ->
  (Kernel.t, error) Err.t
(** [select] absent adapts the whole program; [outputs] absent takes
    [required_outputs], so the ordinary caller does not run the analysis twice.
    When supplied, [outputs] must BEGIN with the required list — order and
    repeats included — after which extras naming selected values are allowed and
    simply force a store. Membership alone would be too weak: for a program
    output list [v; v] a caller could pass [v] and silently drop an output
    position.

    Boundary inputs are derived from the selected bodies' free sources, not
    copied wholesale, so a kernel cut from one branch of a graph does not demand
    bindings for another branch's inputs.

    A graph output that is a boundary input rather than a stage result is
    rejected as [`Passthrough_output] rather than filtered away: [Kernel.Output]
    names a value, and silently returning a kernel that computes fewer outputs
    than its source program is the one failure no downstream test would catch.
*)
