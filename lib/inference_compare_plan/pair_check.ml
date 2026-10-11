open Inference_compare_schema
module Codecs = Inference_compare_codecs
module Recipe_eval = Codecs.Recipe_eval
open Recipe_scope
module Str_set = Set.Make (String)

(* Member pairs and tensor relations. Returns the lookup from a pair id to its
   declaration once every pair is referenced by exactly one unit, resolves on
   both sides' unit outputs, converts to its comparison spec through the codec
   checker, and every relation member exists in an endpoint graph. *)
let check ~options ~left ~right ~partition (m : Mapping_artifact.t) =
  let frontier = m.frontier in
  let idx = function Side.Left -> left | Side.Right -> right in
  let tensor_spec side t = Graph_index.spec (idx side) t in
  (* Member pairs. *)
  let all_pairs =
    List.concat_map
      (fun (r : Tensor_relation.t) -> r.member_pairs)
      m.tensor_relations
  in
  let* pair_table =
    let rec go tbl = function
      | [] -> Ok tbl
      | (p : Member_pair.t) :: rest ->
          let k = Id.Member_pair.to_string p.pair_id in
          if Str_set.mem k (fst tbl) then
            fail (Plan_error.make ~tensor:p.left Duplicate_owner)
          else go (Str_set.add k (fst tbl), (k, p) :: snd tbl) rest
    in
    go (Str_set.empty, []) all_pairs
  in
  let pair_of id =
    List.assoc_opt (Id.Member_pair.to_string id) (snd pair_table)
  in
  let* () =
    (* Comparison ids are the sites' identities: each is used once. *)
    let ids =
      List.map
        (fun (b : Output_binding.t) -> b.comparison_id)
        frontier.output_bindings
      @ List.map (fun (p : Member_pair.t) -> p.comparison_id) all_pairs
    in
    let rec go seen = function
      | [] -> Ok ()
      | c :: rest ->
          let k = Id.Comparison.to_string c in
          if Str_set.mem k seen then fail (Plan_error.make Duplicate_owner)
          else go (Str_set.add k seen) rest
    in
    go Str_set.empty ids
  in
  let* referenced =
    let rec per_unit used = function
      | [] -> Ok used
      | (u : Matched_unit.t) :: rest ->
          let rec per_pair used = function
            | [] -> Ok used
            | pid :: more -> (
                match pair_of pid with
                | None ->
                    fail (Plan_error.make ~unit:u.unit_id Missing_tensor_pair)
                | Some (p : Member_pair.t) ->
                    let k = Id.Member_pair.to_string pid in
                    if Str_set.mem k used then
                      fail
                        (Plan_error.make ~unit:u.unit_id ~tensor:p.left
                           Duplicate_owner)
                    else
                      let side_ok side tensor steps =
                        match
                          Partition_check.boundary partition u.unit_id side
                        with
                        | None ->
                            fail
                              (Plan_error.make ~side ~unit:u.unit_id
                                 Unknown_member)
                        | Some bd ->
                            if
                              not
                                (List.exists (Tensor_ref.equal tensor)
                                   bd.outputs)
                            then
                              fail
                                (Plan_error.make ~side ~unit:u.unit_id ~tensor
                                   Boundary_mismatch)
                            else
                              let* source =
                                match tensor_spec side tensor with
                                | Some s -> Ok s
                                | None ->
                                    fail
                                      (Plan_error.make ~side ~unit:u.unit_id
                                         ~tensor Unknown_member)
                              in
                              let recipe =
                                chain_recipe
                                  ~id:("pair-" ^ Id.Member_pair.to_string pid)
                                  tensor steps p.comparison_spec
                              in
                              let* checked =
                                check_recipe ~options
                                  ~canonical:(fun _ -> None)
                                  ~boundary:(tensor_env [ (tensor, source) ])
                                  ~side ~unit:u.unit_id ~tensor recipe
                              in
                              ignore checked;
                              Ok ()
                      in
                      let* () = side_ok Side.Left p.left p.left_to_comparison in
                      let* () =
                        side_ok Side.Right p.right p.right_to_comparison
                      in
                      per_pair (Str_set.add k used) more)
          in
          let* used = per_pair used u.output_pairs in
          per_unit used rest
    in
    per_unit Str_set.empty m.units
  in
  let* () =
    let rec go = function
      | [] -> Ok ()
      | (k, (p : Member_pair.t)) :: rest ->
          if Str_set.mem k referenced then go rest
          else fail (Plan_error.make ~tensor:p.left Unknown_member)
    in
    go (snd pair_table)
  in
  (* Relations name members of the endpoint graphs. *)
  let* () =
    let member side t =
      match Graph_index.producer (idx side) t with
      | Some _
        when Graph_key.equal t.Tensor_ref.graph (Graph_index.key (idx side)) ->
          Ok ()
      | _ -> fail (Plan_error.make ~side ~tensor:t Unknown_member)
    in
    let rec go = function
      | [] -> Ok ()
      | (r : Tensor_relation.t) :: rest ->
          let rec all side = function
            | [] -> Ok ()
            | t :: more ->
                let* () = member side t in
                all side more
          in
          let* () = all Side.Left r.left_members in
          let* () = all Side.Right r.right_members in
          let rec routes = function
            | [] -> Ok ()
            | (v : Value_route.t) :: more ->
                let either t =
                  match member Side.Left t with
                  | Ok () -> Ok ()
                  | Error _ -> member Side.Right t
                in
                let rec srcs = function
                  | [] -> either v.destination
                  | t :: more ->
                      let* () = either t in
                      srcs more
                in
                let* () = srcs v.sources in
                routes more
          in
          let* () = routes r.routes in
          go rest
    in
    go m.tensor_relations
  in
  Ok pair_of
