(* Packaging a rewrite as a reusable pass, and running a pipeline of them.
   See .ai/native_transform_design.md §5.

   [t.run] is explicitly polymorphic because a pass must work at every version:
   the state's tag is minted per [Rewrite.origin] and rebound by every [Step], so
   a pass fixed to one version could be used exactly once. *)

open Graph_ir

type error = [ Rewrite.error | `Not_converged of string ]

val pp_error : Format.formatter -> [< error ] -> unit

type t = {
  name : string;
  run : 'v. 'v Rewrite.t -> ('v Rewrite.step, error) Core.result;
}

(* Visit each node and offer a rewrite. The callback takes a view, so it too has
   to be explicitly polymorphic; it returns a BUILDER rather than a recipe, so
   the driver — not the pass — threads the allocator and keeps allocation
   sequential and contiguous across the matches it accepts. *)
type per_node = { on_node : Graph_view.t -> node -> unit Recipe.t option }

val per_node : name:string -> per_node -> t

(* Match with a pattern anchored at each node output; [build] turns an accepted
   match into a builder. Needs no rank-2 annotation, [Recipe.t] being
   version-free. *)
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
val run_all : 'v Rewrite.t -> t list -> ('v Rewrite.step, error) Core.result
