(* A mapping regrouped into one unit holding every node of both graphs: the
   frontier reads the graphs' sources and produces their final outputs, so
   running it executes each engine's complete original graph once, on the same
   pinned case. Nothing is invented: every binding, check and output binding
   is one the mapping already carries, re-homed on the merged unit. *)

open Inference_compare_schema

let ( let* ) = Result.bind
let whole = "whole"

let canonical_keys (r : Recipe.t) =
  List.filter_map
    (function Recipe.Canonical k -> Some k | _ -> None)
    (r.result :: List.concat_map (fun (n : Recipe.node) -> n.arguments) r.nodes)

let mem_tensor t l = List.exists (Tensor_ref.equal t) l

let merge_cluster (clusters : Cluster.t list) ~finals : Cluster.t =
  let produced = List.concat_map (fun (c : Cluster.t) -> c.outputs) clusters in
  let sort l = List.sort_uniq Tensor_ref.compare l in
  {
    Cluster.nodes =
      List.sort_uniq Node_ref.compare
        (List.concat_map (fun (c : Cluster.t) -> c.nodes) clusters);
    inputs =
      sort
        (List.filter
           (fun t -> not (mem_tensor t produced))
           (List.concat_map (fun (c : Cluster.t) -> c.inputs) clusters));
    outputs = sort (List.filter (fun t -> mem_tensor t finals) produced);
  }

let first f l = List.find_map f l

let of_mapping (m : Mapping_artifact.t) : (Mapping_artifact.t, string) result =
  let f = m.frontier in
  let* unit_id =
    Result.map_error (fun _ -> "whole-graph unit id") (Id.Unit.of_string whole)
  in
  let left_finals =
    List.map (fun (o : Final_output.t) -> o.left) f.final_outputs
  and right_finals =
    List.map (fun (o : Final_output.t) -> o.right) f.final_outputs
  in
  let left =
    merge_cluster
      (List.map (fun (u : Matched_unit.t) -> u.left) m.units)
      ~finals:left_finals
  and right =
    merge_cluster
      (List.map (fun (u : Matched_unit.t) -> u.right) m.units)
      ~finals:right_finals
  in
  let binding side t =
    first
      (fun (b : Input_binding.t) ->
        if b.side = side && Tensor_ref.equal b.tensor t then Some b else None)
      f.input_bindings
  in
  let* input_bindings =
    let side s inputs =
      List.fold_left
        (fun acc t ->
          let* acc = acc in
          match binding s t with
          | Some b -> Ok (acc @ [ { b with Input_binding.unit = unit_id } ])
          | None -> Error "a graph source has no input binding")
        (Ok []) inputs
    in
    let* l = side Left left.inputs in
    let* r = side Right right.inputs in
    Ok (l @ r)
  in
  let keys_read side =
    List.sort_uniq Id.Canonical_value.compare
      (List.concat_map
         (fun (b : Input_binding.t) ->
           if b.side = side then canonical_keys b.from_common else [])
         input_bindings)
  in
  let* input_checks =
    let side s =
      List.fold_left
        (fun acc k ->
          let* acc = acc in
          match
            first
              (fun (c : Input_check.t) ->
                if c.side = s && Id.Canonical_value.equal c.canonical k then
                  Some c
                else None)
              f.input_checks
          with
          | Some c -> Ok (acc @ [ { c with Input_check.unit = unit_id } ])
          | None -> Error "a canonical value read by the graph has no check")
        (Ok []) (keys_read s)
    in
    let* l = side Side.Left in
    let* r = side Side.Right in
    Ok (l @ r)
  in
  let final_keys =
    List.sort_uniq Id.Canonical_value.compare
      (List.map (fun (o : Final_output.t) -> o.canonical) f.final_outputs)
  in
  let* output_bindings =
    List.fold_left
      (fun acc k ->
        let* acc = acc in
        match
          first
            (fun (o : Output_binding.t) ->
              if Id.Canonical_value.equal o.key k then Some o else None)
            f.output_bindings
        with
        | Some o -> Ok (acc @ [ { o with Output_binding.unit = unit_id } ])
        | None -> Error "a final output has no output binding")
      (Ok []) final_keys
  in
  let referenced =
    keys_read Left @ keys_read Right @ final_keys
    @ List.map (fun (c : Input_check.t) -> c.canonical) input_checks
  in
  let used k = List.exists (Id.Canonical_value.equal k) referenced in
  let comparisons =
    List.map (fun (o : Output_binding.t) -> o.comparison_id) output_bindings
  in
  Ok
    {
      m with
      units = [ { Matched_unit.unit_id; left; right; output_pairs = [] } ];
      dispositions =
        List.map
          (function
            | Member_disposition.Internal_helper h ->
                Member_disposition.Internal_helper
                  { h with owner_unit = unit_id }
            | d -> d)
          m.dispositions;
      tensor_relations =
        List.map
          (fun (r : Tensor_relation.t) ->
            {
              r with
              member_pairs =
                List.filter
                  (fun (p : Member_pair.t) ->
                    List.exists
                      (Id.Comparison.equal p.comparison_id)
                      comparisons)
                  r.member_pairs;
            })
          m.tensor_relations;
      frontier =
        {
          f with
          canonical_values =
            List.filter
              (fun (c : Canonical_spec.t) -> used c.key)
              f.canonical_values;
          initial_values =
            List.filter
              (fun (i : Initial_binding.t) -> used i.key)
              f.initial_values;
          input_bindings;
          input_checks;
          output_bindings;
        };
    }
