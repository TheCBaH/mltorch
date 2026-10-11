open Inference_compare_schema
open Util
open Units

let build ctx ~(first : Mapping_artifact.t) ~(second : Mapping_artifact.t)
    ~units ~relations =
  (* member pairs: a first pair (a, b) and a second pair (b, c) make (a, c),
     kept only where both ends are outputs of the same composed unit *)
  let first_pairs =
    List.concat_map
      (fun (r : Tensor_relation.t) -> r.member_pairs)
      first.tensor_relations
  and second_pairs =
    List.concat_map
      (fun (r : Tensor_relation.t) -> r.member_pairs)
      second.tensor_relations
  in
  let* pairs =
    all
      (fun (cu : composed_unit) ->
        let candidates =
          List.concat_map
            (fun (p1 : Member_pair.t) ->
              List.filter_map
                (fun (p2 : Member_pair.t) ->
                  if
                    Tensor_ref.equal p1.right p2.left
                    && mem_tensor p1.left cu.left.outputs
                    && mem_tensor p2.right cu.right.outputs
                  then Some (p1, p2)
                  else None)
                second_pairs)
            first_pairs
        in
        all
          (fun ((p1 : Member_pair.t), (p2 : Member_pair.t)) ->
            if p1.right_to_comparison <> [] then
              fail (E.Lossy_comparison p1.pair_id)
            else
              let text = tkey p1.left ^ ">" ^ tkey p2.right in
              let* pair_id = pair_id ctx text in
              let* comparison_id = comparison_id ctx text in
              Ok
                ( cu.id,
                  {
                    Member_pair.pair_id;
                    comparison_id;
                    left = p1.left;
                    right = p2.right;
                    claim = Claim.join p1.claim p2.claim;
                    left_to_comparison =
                      p1.left_to_comparison @ p2.left_to_comparison;
                    right_to_comparison = p2.right_to_comparison;
                    comparison_spec = p2.comparison_spec;
                  } ))
          candidates)
      units
  in
  let pairs = List.concat pairs in
  let relations =
    List.map
      (fun (r : Tensor_relation.t) ->
        {
          r with
          member_pairs =
            List.filter_map
              (fun (_, (p : Member_pair.t)) ->
                if
                  mem_tensor p.left r.left_members
                  && mem_tensor p.right r.right_members
                then Some p
                else None)
              pairs;
        })
      relations
  in
  let matched_units : Matched_unit.t list =
    List.map
      (fun cu ->
        {
          Matched_unit.unit_id = cu.id;
          left = cu.left;
          right = cu.right;
          output_pairs =
            List.filter_map
              (fun (u, (p : Member_pair.t)) ->
                if Id.Unit.equal u cu.id then Some p.pair_id else None)
              pairs;
        })
      units
  in
  Ok (relations, matched_units)
