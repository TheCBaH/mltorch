open Inference_compare_schema
open Util

(* ---- node components ----------------------------------------------------- *)

type layer = A | B | C

let layer_tag = function A -> "A|" | B -> "B|" | C -> "C|"

type component = {
  mutable a : Node_ref.t list;
  mutable b : Node_ref.t list;
  mutable c : Node_ref.t list;
  mutable first : Matched_unit.t list;
  mutable second : Matched_unit.t list;
}

let created_nodes (m : Mapping_artifact.t) =
  List.filter_map
    (function Member_disposition.Created { node; _ } -> Some node | _ -> None)
    m.dispositions

let eliminated_nodes (m : Mapping_artifact.t) =
  List.filter_map
    (function
      | Member_disposition.Eliminated { node; _ } -> Some node | _ -> None)
    m.dispositions

let components (first : Mapping_artifact.t) (second : Mapping_artifact.t) =
  let* () =
    let b_first =
      Node_ref.Set.of_list
        (List.concat_map (fun (u : Matched_unit.t) -> u.right.nodes) first.units
        @ created_nodes first)
    and b_second =
      Node_ref.Set.of_list
        (List.concat_map (fun (u : Matched_unit.t) -> u.left.nodes) second.units
        @ eliminated_nodes second)
    in
    match
      ( Node_ref.Set.choose_opt (Node_ref.Set.diff b_first b_second),
        Node_ref.Set.choose_opt (Node_ref.Set.diff b_second b_first) )
    with
    | Some node, _ ->
        fail (E.Missing_intermediate_member { absent_from = Second; node })
    | None, Some node ->
        fail (E.Missing_intermediate_member { absent_from = First; node })
    | None, None -> Ok ()
  in
  let uf = Uf.create () in
  let members = ref [] in
  let key layer n = layer_tag layer ^ nkey n in
  let reg layer n =
    members := (layer, n) :: !members;
    ignore (Uf.find uf (key layer n))
  in
  let link layers =
    List.iter (fun (l, n) -> reg l n) layers;
    match layers with
    | [] -> ()
    | (l0, n0) :: rest ->
        List.iter (fun (l, n) -> Uf.union uf (key l0 n0) (key l n)) rest
  in
  let tagged layer = List.map (fun n -> (layer, n)) in
  List.iter
    (fun (u : Matched_unit.t) ->
      link (tagged A u.left.nodes @ tagged B u.right.nodes))
    first.units;
  List.iter
    (fun (u : Matched_unit.t) ->
      link (tagged B u.left.nodes @ tagged C u.right.nodes))
    second.units;
  List.iter (fun n -> link [ (A, n) ]) (eliminated_nodes first);
  List.iter (fun n -> link [ (B, n) ]) (created_nodes first);
  List.iter (fun n -> link [ (B, n) ]) (eliminated_nodes second);
  List.iter (fun n -> link [ (C, n) ]) (created_nodes second);
  let table : (string, component) Hashtbl.t = Hashtbl.create 64 in
  let order = ref [] in
  let comp root =
    match Hashtbl.find_opt table root with
    | Some c -> c
    | None ->
        let c = { a = []; b = []; c = []; first = []; second = [] } in
        Hashtbl.add table root c;
        order := root :: !order;
        c
  in
  List.iter
    (fun (layer, n) ->
      let c = comp (Uf.find uf (key layer n)) in
      match layer with
      | A -> c.a <- n :: c.a
      | B -> c.b <- n :: c.b
      | C -> c.c <- n :: c.c)
    (List.rev !members);
  let first_root (u : Matched_unit.t) =
    match (u.left.nodes, u.right.nodes) with
    | n :: _, _ -> Some (Uf.find uf (key A n))
    | [], n :: _ -> Some (Uf.find uf (key B n))
    | [], [] -> None
  and second_root (u : Matched_unit.t) =
    match (u.left.nodes, u.right.nodes) with
    | n :: _, _ -> Some (Uf.find uf (key B n))
    | [], n :: _ -> Some (Uf.find uf (key C n))
    | [], [] -> None
  in
  List.iter
    (fun u ->
      Option.iter
        (fun r ->
          let c = comp r in
          c.first <- c.first @ [ u ])
        (first_root u))
    first.units;
  List.iter
    (fun u ->
      Option.iter
        (fun r ->
          let c = comp r in
          c.second <- c.second @ [ u ])
        (second_root u))
    second.units;
  let dedupe l = List.sort_uniq Node_ref.compare l in
  Ok
    (List.rev_map
       (fun root ->
         let c = Hashtbl.find table root in
         { c with a = dedupe c.a; b = dedupe c.b; c = dedupe c.c })
       !order)
