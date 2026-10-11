open Inference_compare_schema
module Codecs = Inference_compare_codecs
module Recipe_eval = Codecs.Recipe_eval

type options = {
  codecs : Codecs.Element_codec.Registry.t;
  operations : Codecs.Conversion_registry.t;
}

let ( let* ) = Err.bind
let fail = Plan_error.fail

let tensor_env side_specs : Tensor_ref.t -> Tensor_spec.t option =
 fun r ->
  List.find_map
    (fun (k, s) -> if Tensor_ref.equal k r then Some s else None)
    side_specs

let spec_pair expected actual = Plan_error.Spec_pair { expected; actual }

let check_recipe ~options ~canonical ~boundary ?side ?unit ?tensor ?key recipe =
  let env =
    {
      Recipe_eval.codecs = options.codecs;
      operations = options.operations;
      canonical;
      boundary;
    }
  in
  match Recipe_eval.check env recipe with
  | Ok checked -> Ok checked
  | Error e ->
      fail
        (Plan_error.of_recipe_fault ?side ?unit ?tensor ?canonical:key
           (Err.Error.kind e))

(* A recipe that feeds the common frontier must be lossless by the codec
   layer's own computation, not merely declared so. *)
let require_lossless ?side ?unit ?tensor ?canonical checked =
  if Recipe_eval.Checked.lossless checked then Ok ()
  else
    fail
      (Plan_error.of_recipe_fault ?side ?unit ?tensor ?canonical
         Codecs.Codec_error.Lossy_recipe)

let require_spec ?side ?unit ?tensor ?canonical ~expected checked =
  let actual = Recipe_eval.Checked.result_spec checked in
  if Tensor_spec.equal expected actual then Ok ()
  else
    fail
      (Plan_error.make ?side ?unit ?tensor ?canonical
         ~detail:(spec_pair expected actual)
         Boundary_mismatch)

(* A conversion chain as a recipe over one boundary tensor. *)
let chain_recipe ~id tensor steps spec : Recipe.t =
  let node_id i =
    match Id.Conversion_node.of_string (Printf.sprintf "step%d" i) with
    | Ok n -> n
    | Error _ -> invalid_arg "Correspondence_check.chain_recipe"
  in
  let nodes =
    List.mapi
      (fun i op ->
        {
          Recipe.id = node_id i;
          operation = op;
          arguments =
            [
              (if i = 0 then Recipe.Boundary tensor
               else Recipe.Conversion_result (node_id (i - 1)));
            ];
        })
      steps
  in
  let result =
    match steps with
    | [] -> Recipe.Boundary tensor
    | _ -> Recipe.Conversion_result (node_id (List.length steps - 1))
  in
  let id =
    match Id.Recipe.of_string id with
    | Ok r -> r
    | Error _ -> invalid_arg "Correspondence_check.chain_recipe: id"
  in
  { Recipe.id; nodes; result; output_spec = spec; lossless = false }
