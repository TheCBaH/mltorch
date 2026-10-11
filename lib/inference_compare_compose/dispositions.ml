open Inference_compare_schema
open Util
open Components
open Units

let build ~(first : Mapping_artifact.t) ~(second : Mapping_artifact.t) ~comps
    ~units =
  (* dispositions *)
  let unit_of_second id =
    find_first
      (fun cu ->
        if has_unit id (unit_ids cu.comp.second) then Some cu.id else None)
      units
  in
  let reason text =
    match Prose.Description.of_string text with
    | Ok d -> Ok d
    | Error _ -> fail (E.Invalid_identifier text)
  in
  let reason_of dispositions node default =
    List.find_map
      (function
        | Member_disposition.Eliminated { node = n; reason }
        | Created { node = n; reason }
          when Node_ref.equal n node ->
            Some reason
        | _ -> None)
      dispositions
    |> function
    | Some r -> Ok r
    | None -> reason default
  in
  let* dispositions =
    let* per_component =
      all
        (fun comp ->
          if comp.a <> [] && comp.c = [] then
            all
              (fun node ->
                let* reason =
                  reason_of first.dispositions node "removed by a later pass"
                in
                Ok (Member_disposition.Eliminated { node; reason }))
              comp.a
          else if comp.a = [] && comp.c <> [] then
            all
              (fun node ->
                let* reason =
                  reason_of second.dispositions node "made by a later pass"
                in
                Ok (Member_disposition.Created { node; reason }))
              comp.c
          else Ok [])
        comps
    in
    let helpers =
      List.filter_map
        (function
          | Member_disposition.Internal_helper { node; owner_unit } ->
              Option.map
                (fun owner_unit ->
                  Member_disposition.Internal_helper { node; owner_unit })
                (unit_of_second owner_unit)
          | _ -> None)
        second.dispositions
    and uncompared =
      List.filter
        (function
          | Member_disposition.Tensor_not_compared { tensor; _ } ->
              Graph_key.equal tensor.Tensor_ref.graph first.left_graph
              || Graph_key.equal tensor.Tensor_ref.graph second.right_graph
          | _ -> false)
        (first.dispositions @ second.dispositions)
    in
    Ok (List.concat per_component @ helpers @ uncompared)
  in
  Ok dispositions
