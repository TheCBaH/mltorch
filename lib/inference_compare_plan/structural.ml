open Inference_compare_schema

type t = {
  mapping : Mapping_artifact.t;
  left : Graph_index.t;
  right : Graph_index.t;
  partition : Partition_check.t;
  correspondence : Correspondence_check.t;
  order : Unit_order.t;
}

let ( let* ) = Err.bind
let fail = Plan_error.fail

let check ~(limits : Resource_limits.t) ~options ~left ~right
    (m : Mapping_artifact.t) =
  let m = Mapping_artifact.canonicalize m in
  let left_desc = Graph_desc.canonicalize left
  and right_desc = Graph_desc.canonicalize right in
  (* The artifact names the graphs it was built for. A mismatch is a stale
     map, whatever the graphs' contents. *)
  let* () =
    if Graph_key.equal m.left_graph left_desc.key then Ok ()
    else fail (Plan_error.make ~side:Left Stale_graph)
  in
  let* () =
    if Graph_key.equal m.right_graph right_desc.key then Ok ()
    else fail (Plan_error.make ~side:Right Stale_graph)
  in
  (* Bound the artifact before walking it: recipe expansion can dwarf the
     graphs, and composition only grows it. *)
  let recipe_nodes =
    let count (r : Recipe.t) = List.length r.nodes in
    let f = m.frontier in
    List.fold_left
      (fun a (b : Input_binding.t) -> a + count b.from_common)
      0 f.input_bindings
    + List.fold_left
        (fun a (c : Input_check.t) -> a + count c.from_imported_inputs)
        0 f.input_checks
    + List.fold_left
        (fun a (b : Output_binding.t) ->
          a + count b.from_left_outputs + count b.from_right_outputs)
        0 f.output_bindings
  in
  let* () =
    let actual = Int64.of_int recipe_nodes in
    if Int64.compare actual limits.max_recipe_nodes > 0 then
      fail
        (Plan_error.make
           ~detail:
             (Limit
                {
                  limit = Recipe_nodes;
                  bound = limits.max_recipe_nodes;
                  actual;
                })
           Resource_exhausted)
    else Ok ()
  in
  let* left_idx = Graph_index.build ~limits ~side:Left left_desc in
  let* right_idx = Graph_index.build ~limits ~side:Right right_desc in
  let* partition = Partition_check.check ~left:left_idx ~right:right_idx m in
  let* correspondence =
    Correspondence_check.check ~options ~left:left_idx ~right:right_idx
      ~partition m
  in
  let* order =
    Unit_order.compute ~limits ~left:left_idx ~right:right_idx ~partition
      ~correspondence
  in
  Ok
    {
      mapping = m;
      left = left_idx;
      right = right_idx;
      partition;
      correspondence;
      order;
    }

let mapping t = t.mapping
let left t = t.left
let right t = t.right
let partition t = t.partition
let correspondence t = t.correspondence
let order t = t.order

let units t =
  List.filter_map
    (fun u -> Correspondence_check.unit_plan t.correspondence u)
    (Unit_order.order t.order)

let coverage t = Correspondence_check.coverage t.correspondence
let final_values t = Correspondence_check.final_values t.correspondence
let mapping_preimage t = Mapping_artifact.preimage t.mapping
