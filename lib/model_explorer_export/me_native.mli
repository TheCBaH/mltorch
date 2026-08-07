(* The Native instantiation of {!Me_build}. See there for what the projection
   does; this file exists so that there is exactly ONE Native projection in the
   program rather than a functor application per call site. *)

val graph :
  limits:Me_limits.Limits.t ->
  id:string ->
  ?labels:(Graph_ir.Node_id.t -> string) ->
  Graph_ir.graph ->
  (Model_explorer.Graph.t, [> Me_build.error ]) Core.result
