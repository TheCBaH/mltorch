(* An extracted body of a Native-family graph: the sub-graph, the extraction's
   graph key and the scope its tensors are named in. *)

open Inference_compare_schema

type 'op t = {
  graph : 'op Graph_common.Graph.t;
  key : Graph_key.t;
  scope : Scope.t;
}
