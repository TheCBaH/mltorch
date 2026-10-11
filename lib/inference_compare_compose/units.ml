open Inference_compare_schema
open Util
open Components

type composed_unit = {
  id : Id.Unit.t;
  comp : component;
  left : Cluster.t;
  right : Cluster.t;
}

(* A cluster merged from whole units: what its members read from outside the
   set and what anything outside the set (or the graph's result list) reads of
   what it makes. *)
let merged_cluster ~(units : Matched_unit.t list)
    ~(side : Matched_unit.t -> Cluster.t) ~(all : Matched_unit.t list) ~finals
    ~nodes : Cluster.t =
  let ids = unit_ids units in
  let clusters = List.map side units in
  let produced = List.concat_map (fun (c : Cluster.t) -> c.outputs) clusters in
  let inputs =
    sort_tensors
      (List.filter
         (fun t -> not (mem_tensor t produced))
         (List.concat_map (fun (c : Cluster.t) -> c.inputs) clusters))
  in
  let consumed_outside t =
    List.exists
      (fun (u : Matched_unit.t) ->
        (not (has_unit u.unit_id ids)) && mem_tensor t (side u).inputs)
      all
  in
  let outputs =
    sort_tensors
      (List.filter
         (fun t -> mem_tensor t finals || consumed_outside t)
         produced)
  in
  { Cluster.nodes = List.sort_uniq Node_ref.compare nodes; inputs; outputs }

let build ctx ~(first : Mapping_artifact.t) ~(second : Mapping_artifact.t) comps
    =
  let a_finals =
    List.map (fun (f : Final_output.t) -> f.left) first.frontier.final_outputs
  and c_finals =
    List.map (fun (f : Final_output.t) -> f.right) second.frontier.final_outputs
  in
  (* units *)
  let* units =
    all
      (fun comp ->
        if comp.a = [] && comp.c = [] then Ok None
        else
          let terminal =
            List.sort compare
              (List.map (fun n -> "A|" ^ nkey n) comp.a
              @ List.map (fun n -> "C|" ^ nkey n) comp.c)
          in
          let* id = unit_id ctx (String.concat "\n" terminal) in
          let left =
            merged_cluster ~units:comp.first
              ~side:(fun u -> u.Matched_unit.left)
              ~all:first.units ~finals:a_finals ~nodes:comp.a
          and right =
            merged_cluster ~units:comp.second
              ~side:(fun u -> u.Matched_unit.right)
              ~all:second.units ~finals:c_finals ~nodes:comp.c
          in
          Ok (Some { id; comp; left; right }))
      comps
  in
  let units =
    List.sort
      (fun x y -> Id.Unit.compare x.id y.id)
      (List.filter_map Fun.id units)
  in
  Ok units
