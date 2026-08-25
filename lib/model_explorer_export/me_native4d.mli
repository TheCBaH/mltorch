(* The Native4D instantiation of {!Me_build}.

   This is why that module is a functor at all: Native4D is projected through
   the SAME body with a different op type. The alternative is two projections
   that have to be kept saying the same thing — and a divergence between them
   would show as "the 4D view renders differently", which reads as a dialect
   difference rather than as the exporter defect it would be.

   One instantiation, at one path, for the reason [Native4d.Framework] gives for
   its own: applying the functor per call site would work, since OCaml's named
   functors are applicative, but it scatters the ownership of which
   specialization is THE Native4D one. *)

val graph :
  limits:Me_limits.Limits.t ->
  id:string ->
  ?labels:(Graph_ir.Node_id.t -> string) ->
  ?group_attrs:(string * (string * string) list) list ->
  ?constant_store:Constant_store.t ->
  Native4d.Graph.graph ->
  (Model_explorer.Graph.t, [> Me_build.error ]) Err.t
