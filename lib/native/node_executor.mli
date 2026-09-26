(* The pluggable seam between [Eval_direct]'s non-Region computing arms
   (design §4.2, plan S2) and however one node's output actually gets
   computed. [default] always calls [direct], so an [Eval_direct.run] caller
   that never passes [?node_executor] gets the exact same result as before
   this seam existed -- the same convention [Region_executor.t]/[default]
   already establish for the Region-authored arm.

   [run]'s error row is universally quantified ([Tensor.packed, 'e) Err.t]
   for every 'e, not fixed to [Eval_direct.error]: this module sits BELOW
   [Eval_direct] in the dependency order (mirroring [Region_executor]'s own
   placement) and so cannot name [Eval_direct.error]. The executor never
   invents an error of its own -- it only ever returns what [direct ()]
   returned, so whatever error row the caller's own [direct] thunk produces
   passes through unchanged. This is why the field must be a record, not a
   bare function-type alias: only a record (or object) field can carry an
   explicit rank-2 [~'e.] quantifier in OCaml. *)

open Graph_ir

type t = {
  run :
    'e.
    graph ->
    node ->
    output:Output_ordinal.t ->
    out_shape:Vec6.shape ->
    operands:Tensor.packed Tensor_id.Map.t ->
    direct:(unit -> (Tensor.packed, 'e) Err.t) ->
    (Tensor.packed, 'e) Err.t;
}

val default : t
(** [{ run = fun _ _ ~output:_ ~out_shape:_ ~operands:_ ~direct -> direct () }]:
    every node is [pending] and falls straight back to the direct path. *)
