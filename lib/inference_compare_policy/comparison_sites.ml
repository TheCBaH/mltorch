open Inference_compare_schema
module Plan = Inference_compare_plan
module Unit_plan = Plan.Correspondence_check.Unit_plan

let ( let* ) = Err.bind

let compare_site (a : Comparison_site.t) (b : Comparison_site.t) =
  Id.Comparison.compare a.comparison b.comparison

let find sites id =
  List.find_opt
    (fun (s : Comparison_site.t) -> Id.Comparison.equal s.comparison id)
    sites

let recipe_args (r : Recipe.t) =
  r.result :: List.concat_map (fun (n : Recipe.node) -> n.arguments) r.nodes

let sort_unique_tensors ts = List.sort_uniq Tensor_ref.compare ts

(* Boundary tensors a candidate recipe reads, plus -- for a canonical argument
   it passes through -- the unit's input tensors whose bindings read that
   canonical value on this side. *)
let anchors (plan : Unit_plan.t) side (recipe : Recipe.t) =
  let direct =
    List.filter_map
      (function Recipe.Boundary t -> Some t | _ -> None)
      (recipe_args recipe)
  in
  let passthrough =
    List.filter_map
      (function Recipe.Canonical k -> Some k | _ -> None)
      (recipe_args recipe)
  in
  let traced =
    List.concat_map
      (fun (b : Input_binding.t) ->
        if
          Side.equal b.side side
          && List.exists
               (function
                 | Recipe.Canonical c ->
                     List.exists (Id.Canonical_value.equal c) passthrough
                 | _ -> false)
               (recipe_args b.from_common)
        then [ b.tensor ]
        else [])
      plan.input_bindings
  in
  sort_unique_tensors (direct @ traced)

let occurrences structural pred =
  let frontier = (Plan.Structural.mapping structural).frontier in
  List.filter_map
    (fun (f : Final_output.t) -> if pred f then Some f.occurrence else None)
    frontier.final_outputs

let build structural =
  let correspondence = Plan.Structural.correspondence structural in
  let canonical_spec k =
    Plan.Correspondence_check.canonical_spec correspondence k
  in
  let units = Plan.Correspondence_check.unit_plans correspondence in
  let candidate_sites =
    List.concat_map
      (fun (p : Unit_plan.t) ->
        List.filter_map
          (fun (b : Output_binding.t) ->
            Option.map
              (fun spec ->
                {
                  Comparison_site.comparison = b.comparison_id;
                  unit = p.unit;
                  kind = Canonical_candidate b.key;
                  spec;
                  left_anchors = anchors p Left b.from_left_outputs;
                  right_anchors = anchors p Right b.from_right_outputs;
                  final_occurrences =
                    occurrences structural (fun f ->
                        Id.Canonical_value.equal f.canonical b.key);
                })
              (canonical_spec b.key))
          p.output_bindings)
      units
  in
  let pair_sites =
    List.concat_map
      (fun (p : Unit_plan.t) ->
        List.map
          (fun (m : Member_pair.t) ->
            {
              Comparison_site.comparison = m.comparison_id;
              unit = p.unit;
              kind = Member_pair m.pair_id;
              spec = m.comparison_spec;
              left_anchors = [ m.left ];
              right_anchors = [ m.right ];
              final_occurrences =
                occurrences structural (fun f ->
                    Tensor_ref.equal f.left m.left
                    && Tensor_ref.equal f.right m.right);
            })
          p.output_pairs)
      units
  in
  let sites = List.sort compare_site (candidate_sites @ pair_sites) in
  let rec distinct = function
    | a :: (b : Comparison_site.t) :: _ when compare_site a b = 0 ->
        Policy_error.fail
          (Policy_error.make ~comparison:b.comparison ~unit:b.unit
             Unavailable_comparison_site)
    | _ :: rest -> distinct rest
    | [] -> Ok ()
  in
  let* () = distinct sites in
  Ok sites
