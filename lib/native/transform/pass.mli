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
val run_all : 'v Rewrite.t -> t list -> ('v Rewrite.step, error) Core.result
