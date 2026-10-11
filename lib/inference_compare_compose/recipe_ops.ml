(* Recipes as expression DAGs: chains of conversion steps, substitution of one
   recipe for an argument of another, and a canonical numbering. Nothing is
   cancelled or rearranged: only structure is normalized (unreachable nodes
   dropped, identical nodes shared, ids numbered in dependency order). *)

open Inference_compare_schema

let node_id i =
  match Id.Conversion_node.of_string (Printf.sprintf "n%d" i) with
  | Ok n -> n
  | Error _ -> invalid_arg "Recipe_ops.node_id"

let same_argument a b =
  match (a, b) with
  | Recipe.Boundary x, Recipe.Boundary y -> Tensor_ref.equal x y
  | Canonical x, Canonical y -> Id.Canonical_value.equal x y
  | Conversion_result x, Conversion_result y -> Id.Conversion_node.equal x y
  | _ -> false

let argument_key = function
  | Recipe.Boundary t -> "b:" ^ Wire.canonical_string (Tensor_ref.encode t)
  | Canonical k -> "c:" ^ Id.Canonical_value.to_string k
  | Conversion_result n -> "r:" ^ Id.Conversion_node.to_string n

let find_node (r : Recipe.t) id =
  List.find_opt
    (fun (n : Recipe.node) -> Id.Conversion_node.equal n.id id)
    r.nodes

let normalize (r : Recipe.t) : Recipe.t =
  let renamed = Hashtbl.create 16 and shared = Hashtbl.create 16 in
  let out = ref [] and count = ref 0 in
  let rec visit = function
    | Recipe.Conversion_result id as a -> (
        let k = Id.Conversion_node.to_string id in
        match Hashtbl.find_opt renamed k with
        | Some a' -> a'
        | None -> (
            match find_node r id with
            | None -> a
            | Some n ->
                let arguments = List.map visit n.arguments in
                let key =
                  Wire.canonical_string (Conversion_step.encode n.operation)
                  ^ "|"
                  ^ String.concat "," (List.map argument_key arguments)
                in
                let a' =
                  match Hashtbl.find_opt shared key with
                  | Some a' -> a'
                  | None ->
                      let id = node_id !count in
                      incr count;
                      out :=
                        { Recipe.id; operation = n.operation; arguments }
                        :: !out;
                      let a' = Recipe.Conversion_result id in
                      Hashtbl.add shared key a';
                      a'
                in
                Hashtbl.add renamed k a';
                a'))
    | a -> a
  in
  let result = visit r.result in
  { r with nodes = List.rev !out; result }

(* [steps] applied in order to [argument]. *)
let chain ~id ~argument ~steps ~output_spec : Recipe.t =
  let nodes =
    List.mapi
      (fun i operation ->
        {
          Recipe.id = node_id i;
          operation;
          arguments =
            [
              (if i = 0 then argument
               else Recipe.Conversion_result (node_id (i - 1)));
            ];
        })
      steps
  in
  let result =
    match steps with
    | [] -> argument
    | _ -> Conversion_result (node_id (List.length steps - 1))
  in
  { Recipe.id; nodes; result; output_spec; lossless = false }

let mentions (r : Recipe.t) argument =
  same_argument r.result argument
  || List.exists
       (fun (n : Recipe.node) ->
         List.exists (same_argument argument) n.arguments)
       r.nodes

let boundaries (r : Recipe.t) =
  let args =
    r.result :: List.concat_map (fun (n : Recipe.node) -> n.arguments) r.nodes
  in
  List.fold_left
    (fun acc -> function
      | Recipe.Boundary t when not (List.exists (Tensor_ref.equal t) acc) ->
          acc @ [ t ]
      | _ -> acc)
    [] args

(* Replace every occurrence of [argument] in [outer] by the result of [inner],
   merging [inner]'s nodes. *)
let inline ~(outer : Recipe.t) ~argument ~(inner : Recipe.t) : Recipe.t =
  let prefix p id =
    match
      Id.Conversion_node.of_string (p ^ Id.Conversion_node.to_string id)
    with
    | Ok n -> n
    | Error _ -> invalid_arg "Recipe_ops.inline"
  in
  let rename p = function
    | Recipe.Conversion_result id -> Recipe.Conversion_result (prefix p id)
    | a -> a
  in
  let inner_result = rename "i." inner.result in
  let outer_arg a =
    if same_argument a argument then inner_result else rename "o." a
  in
  let inner_nodes =
    List.map
      (fun (n : Recipe.node) ->
        {
          n with
          Recipe.id = prefix "i." n.id;
          arguments = List.map (rename "i.") n.arguments;
        })
      inner.nodes
  and outer_nodes =
    List.map
      (fun (n : Recipe.node) ->
        {
          n with
          Recipe.id = prefix "o." n.id;
          arguments = List.map outer_arg n.arguments;
        })
      outer.nodes
  in
  normalize
    {
      outer with
      Recipe.nodes = inner_nodes @ outer_nodes;
      result = outer_arg outer.result;
    }
