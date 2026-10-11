open Inference_compare_schema

type absent_from = First | Second

type t =
  | Ambiguous_route of { destination : Tensor_ref.t; source : Tensor_ref.t }
  | Conflicting_constant of Id.Canonical_value.t
  | Intermediate_graph_mismatch of {
      first_right : Graph_key.t;
      second_left : Graph_key.t;
    }
  | Invalid_identifier of string
  | Lossy_comparison of Id.Member_pair.t
  | Missing_binding of Tensor_ref.t
  | Missing_check of Id.Canonical_value.t
  | Missing_intermediate_member of {
      absent_from : absent_from;
      node : Node_ref.t;
    }
  | Missing_route of { tensor : Tensor_ref.t; toward : Side.t }
  | Output_mismatch of { position : int }

let pp ppf = function
  | Ambiguous_route { destination; source } ->
      Fmt.pf ppf "ambiguous route %a -> %a" Tensor_ref.pp source Tensor_ref.pp
        destination
  | Conflicting_constant k ->
      Fmt.pf ppf "conflicting constant %a" Id.Canonical_value.pp k
  | Intermediate_graph_mismatch { first_right; second_left } ->
      Fmt.pf ppf "intermediate graph differs: %a vs %a" Graph_key.pp first_right
        Graph_key.pp second_left
  | Invalid_identifier s -> Fmt.pf ppf "invalid identifier %s" s
  | Lossy_comparison p ->
      Fmt.pf ppf "comparison of pair %a cannot be composed" Id.Member_pair.pp p
  | Missing_binding t -> Fmt.pf ppf "no binding for %a" Tensor_ref.pp t
  | Missing_check k ->
      Fmt.pf ppf "no input check for %a" Id.Canonical_value.pp k
  | Missing_intermediate_member { absent_from; node } ->
      Fmt.pf ppf "intermediate node %a is absent from the %s map" Node_ref.pp
        node
        (match absent_from with First -> "first" | Second -> "second")
  | Missing_route { tensor; toward } ->
      Fmt.pf ppf "no route for %a toward the %s side" Tensor_ref.pp tensor
        (Side.to_string toward)
  | Output_mismatch { position } ->
      Fmt.pf ppf "final output %d does not line up" position

let fail e = Err.fail ~pp_error:pp e
