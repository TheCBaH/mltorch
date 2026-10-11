open Inference_compare_schema
open Util

(* ---- routes -------------------------------------------------------------- *)

type directed = {
  source : Tensor_ref.t;
  destination : Tensor_ref.t;
  steps : Conversion_step.t list;
  claim : Claim.t;
  lossy : bool;
}

let steps_key steps =
  String.concat ";"
    (List.map (fun s -> Wire.canonical_string (Conversion_step.encode s)) steps)

(* The single-source routes of a map, by the direction they run across their
   relation. A multi-source route (a pack) is not composed: the direction it
   would give is then simply unavailable. *)
let directed_routes (m : Mapping_artifact.t) =
  let fwd = ref [] and rev = ref [] in
  List.iter
    (fun (r : Tensor_relation.t) ->
      List.iter
        (fun (v : Value_route.t) ->
          match v.sources with
          | [ s ] ->
              let d =
                {
                  source = s;
                  destination = v.destination;
                  steps = v.steps;
                  claim = v.claim;
                  lossy = v.lossy;
                }
              in
              if
                mem_tensor s r.left_members
                && mem_tensor v.destination r.right_members
              then fwd := d :: !fwd;
              if
                mem_tensor s r.right_members
                && mem_tensor v.destination r.left_members
              then rev := d :: !rev
          | _ -> ())
        r.routes)
    m.tensor_relations;
  (List.rev !fwd, List.rev !rev)

(* Concatenate [outer] routes after [inner] ones through a shared middle
   tensor. Equal steps through several middles are one route; different steps
   between the same endpoints need an external choice. *)
let join_routes ~inner ~outer =
  let candidates =
    List.concat_map
      (fun i ->
        List.filter_map
          (fun o ->
            if Tensor_ref.equal i.destination o.source then
              Some
                {
                  source = i.source;
                  destination = o.destination;
                  steps = i.steps @ o.steps;
                  claim = Claim.join i.claim o.claim;
                  lossy = i.lossy || o.lossy;
                }
            else None)
          outer)
      inner
  in
  let rec merge acc = function
    | [] -> Ok (List.rev acc)
    | d :: rest -> (
        match
          List.find_opt
            (fun e ->
              Tensor_ref.equal e.source d.source
              && Tensor_ref.equal e.destination d.destination)
            acc
        with
        | None -> merge (d :: acc) rest
        | Some e when String.equal (steps_key e.steps) (steps_key d.steps) ->
            let joined = { e with claim = Claim.join e.claim d.claim } in
            merge (joined :: List.filter (fun x -> x != e) acc) rest
        | Some _ ->
            fail
              (E.Ambiguous_route
                 { source = d.source; destination = d.destination }))
  in
  merge [] candidates
