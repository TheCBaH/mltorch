open Inference_compare_schema

module Limit = struct
  type t = Canonical_values | Edges | Nodes | Recipe_nodes | Units
end

module Edge_reason = struct
  type t =
    | Canonical of Id.Canonical_value.t
    | Control of { side : Side.t; node : Node_ref.t }
    | Data of { side : Side.t; tensor : Tensor_ref.t }
end

module Cycle_edge = struct
  type t = {
    from_unit : Id.Unit.t;
    to_unit : Id.Unit.t;
    reason : Edge_reason.t;
  }
end

type detail =
  | Cycle of Cycle_edge.t list
  | Expected_actual_count of { expected : int; actual : int }
  | Limit of { limit : Limit.t; bound : int64; actual : int64 }
  | Node_cycle of Node_ref.t list
  | No_detail
  | Position of int
  | Recipe_fault of Inference_compare_codecs.Codec_error.t
  | Spec_pair of { expected : Tensor_spec.t; actual : Tensor_spec.t }

type t = {
  code : Failure.code;
  side : Side.t option;
  unit : Id.Unit.t option;
  node : Node_ref.t option;
  tensor : Tensor_ref.t option;
  canonical : Id.Canonical_value.t option;
  detail : detail;
}

let make ?side ?unit ?node ?tensor ?canonical ?(detail = No_detail) code =
  { code; side; unit; node; tensor; canonical; detail }

let code_name (c : Failure.code) =
  match Failure.encode_code c with Jsont.String (s, _) -> s | _ -> "?"

let pp ppf e =
  Fmt.pf ppf "%s" (code_name e.code);
  Option.iter (Fmt.pf ppf " side=%a" Side.pp) e.side;
  Option.iter (Fmt.pf ppf " unit=%a" Id.Unit.pp) e.unit;
  Option.iter (Fmt.pf ppf " node=%a" Node_ref.pp) e.node;
  Option.iter (Fmt.pf ppf " tensor=%a" Tensor_ref.pp) e.tensor;
  Option.iter (Fmt.pf ppf " canonical=%a" Id.Canonical_value.pp) e.canonical;
  match e.detail with
  | No_detail -> ()
  | Position i -> Fmt.pf ppf " position=%d" i
  | Expected_actual_count { expected; actual } ->
      Fmt.pf ppf " expected=%d actual=%d" expected actual
  | Limit { bound; actual; _ } ->
      Fmt.pf ppf " limit=%Ld actual=%Ld" bound actual
  | Cycle edges -> Fmt.pf ppf " cycle_length=%d" (List.length edges)
  | Node_cycle nodes -> Fmt.pf ppf " node_cycle_length=%d" (List.length nodes)
  | Recipe_fault c ->
      Fmt.pf ppf " recipe_fault=%a" Inference_compare_codecs.Codec_error.pp c
  | Spec_pair _ -> Fmt.string ppf " spec_mismatch"

let fail e = Err.fail ~pp_error:pp e

(* The design's failure codes have no "invalid recipe"; a recipe fault is
   reported under the code that names what the planner was checking, with the
   codec error kept whole in the detail. *)
let of_recipe_fault ?side ?unit ?tensor ?canonical
    (c : Inference_compare_codecs.Codec_error.t) =
  let code : Failure.code =
    match c with
    | Lossy_recipe | Undeclared_lossy_cast | Unsupported_cast _
    | Cast_overflow _ | Cast_value_unavailable _ ->
        Lossy_common_input
    | Unknown_boundary _ | Unknown_canonical _ | Unknown_node _ ->
        Missing_common_input
    | Duplicate_node _ | Cyclic_recipe _ -> Ambiguous_route
    | _ -> Boundary_mismatch
  in
  make ?side ?unit ?tensor ?canonical ~detail:(Recipe_fault c) code
