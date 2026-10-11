open Inference_compare_schema
open Util
open Components
open Routes

let build ctx ~(first : Mapping_artifact.t) ~(second : Mapping_artifact.t) =
  (* tensor relations *)
  let tuf = Uf.create () in
  let tmembers = ref [] in
  let tkey_l layer t = layer_tag layer ^ tkey t in
  let link_tensors layers =
    List.iter
      (fun (l, t) ->
        tmembers := (l, t) :: !tmembers;
        ignore (Uf.find tuf (tkey_l l t)))
      layers;
    match layers with
    | [] -> ()
    | (l0, t0) :: rest ->
        List.iter (fun (l, t) -> Uf.union tuf (tkey_l l0 t0) (tkey_l l t)) rest
  in
  let tag layer = List.map (fun t -> (layer, t)) in
  List.iter
    (fun (r : Tensor_relation.t) ->
      link_tensors (tag A r.left_members @ tag B r.right_members))
    first.tensor_relations;
  List.iter
    (fun (r : Tensor_relation.t) ->
      link_tensors (tag B r.left_members @ tag C r.right_members))
    second.tensor_relations;
  let groups : (string, Tensor_ref.t list * Tensor_ref.t list) Hashtbl.t =
    Hashtbl.create 64
  in
  let group_order = ref [] in
  List.iter
    (fun (layer, t) ->
      let root = Uf.find tuf (tkey_l layer t) in
      let a, c =
        match Hashtbl.find_opt groups root with
        | Some g -> g
        | None ->
            group_order := root :: !group_order;
            ([], [])
      in
      match layer with
      | A -> Hashtbl.replace groups root (t :: a, c)
      | C -> Hashtbl.replace groups root (a, t :: c)
      | B -> Hashtbl.replace groups root (a, c))
    (List.rev !tmembers);
  let first_fwd, first_rev = directed_routes first
  and second_fwd, second_rev = directed_routes second in
  let* fwd = join_routes ~inner:first_fwd ~outer:second_fwd in
  let* rev = join_routes ~inner:second_rev ~outer:first_rev in
  let in_group a c r = mem_tensor r.source a && mem_tensor r.destination c in
  let* relations =
    all
      (fun root ->
        let a, c = Hashtbl.find groups root in
        if a = [] || c = [] then Ok None
        else
          let a = sort_tensors a and c = sort_tensors c in
          let* relation_id =
            relation_id ctx
              (String.concat "\n"
                 (List.map (fun t -> "A|" ^ tkey t) a
                 @ List.map (fun t -> "C|" ^ tkey t) c))
          in
          let make_route dir (d : directed) =
            let* route_id =
              route_id ctx (dir ^ tkey d.source ^ ">" ^ tkey d.destination)
            in
            Ok
              {
                Value_route.route_id;
                sources = [ d.source ];
                destination = d.destination;
                steps = d.steps;
                claim = d.claim;
                lossy = d.lossy;
              }
          in
          let* routes_f =
            all (make_route "f|") (List.filter (in_group a c) fwd)
          in
          let* routes_r =
            all (make_route "r|")
              (List.filter
                 (fun r -> mem_tensor r.source c && mem_tensor r.destination a)
                 rev)
          in
          Ok
            (Some
               {
                 Tensor_relation.relation_id;
                 left_members = a;
                 right_members = c;
                 member_pairs = [];
                 routes = routes_f @ routes_r;
               }))
      (List.rev !group_order)
  in
  let relations = List.filter_map Fun.id relations in
  Ok (relations, fwd, rev)
