open Inference_compare_schema

let ( let* ) = Err.bind
let fail = Extract_error.fail

let first_difference eq expected actual =
  let rec go i a b =
    match (a, b) with
    | [], [] -> None
    | x :: ar, y :: br -> if eq x y then go (i + 1) ar br else Some i
    | _ -> Some i
  in
  go 0 expected actual

let check (plan : Extraction.t) (actual : Graph_desc.t) =
  let side = plan.side and unit = plan.unit in
  let expected_inputs = plan.inputs
  and actual_inputs =
    List.map (fun (s : Graph_source.t) -> s.tensor) actual.sources
  in
  let* () =
    match first_difference Tensor_ref.equal expected_inputs actual_inputs with
    | None -> Ok ()
    | Some i ->
        fail
          (Extract_error.make ~side ~unit ~detail:(Position i) Boundary_mismatch)
  in
  let* () =
    match
      first_difference Tensor_ref.equal plan.outputs actual.final_outputs
    with
    | None -> Ok ()
    | Some i ->
        (* An extra result is an undeclared observable; any other difference is
           a dropped or reordered output. Both leave the inventory wrong. *)
        fail
          (Extract_error.make ~side ~unit ~detail:(Position i)
             ?tensor:(List.nth_opt (actual.final_outputs @ plan.outputs) i)
             Output_inventory_failed)
  in
  if String.equal (Graph_desc.preimage plan.body) (Graph_desc.preimage actual)
  then Ok ()
  else fail (Extract_error.make ~side ~unit Boundary_mismatch)
