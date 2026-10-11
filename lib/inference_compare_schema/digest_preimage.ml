(* The exact bytes hashed for a record's digest. Every domain has its own tag,
   so equal canonical text under two domains hashes differently and a graph
   digest can never be reused as a plan digest. *)

type domain =
  | Comparison_policy
  | Contract
  | Frontier_mapping
  | Graph
  | Logical_value
  | Mapping
  | Plan
  | Policy_bindings
  | Policy_selection

let tag = function
  | Comparison_policy -> "comparison_policy"
  | Contract -> "contract"
  | Frontier_mapping -> "frontier_mapping"
  | Graph -> "graph"
  | Logical_value -> "logical_value"
  | Mapping -> "mapping"
  | Plan -> "plan"
  | Policy_bindings -> "policy_bindings"
  | Policy_selection -> "policy_selection"

let version = "inference-compare/v1"

let of_json domain json =
  version ^ "/" ^ tag domain ^ "\n" ^ Wire.canonical_string json
