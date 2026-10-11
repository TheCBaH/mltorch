(* ATen dialect tooling: realize a planned extraction as the exporter nodes
   themselves (same targets, same arguments), keeping the original node values
   so the interpreter runs exactly what the model's own run would. *)

open Inference_compare_schema
module Ex = Inference_compare_extract

type graph = Aten_model.t

type body = {
  model : Aten_model.t;
  key : Graph_key.t;
  scope : Scope.t;
  nodes : (int * Pytorch_types.Node.t) list;
  inputs : string list;  (** SSA names, boundary input order. *)
  outputs : string list;
}

let dialect =
  match Id.Dialect.of_string "aten" with
  | Ok d -> d
  | Error _ -> invalid_arg "Aten_dialect"

let describe (m : graph) = m.desc
let ( let* ) = Result.bind

let efail ?(code = Failure.Boundary_mismatch) (plan : Ex.Extraction.t) =
  Ex.Extract_error.fail
    (Ex.Extract_error.make ~side:plan.side ~unit:plan.unit code)

let index_of_local (r : Node_ref.t) =
  let s = Id.Node_local.to_string r.local_id in
  int_of_string_opt (String.sub s 1 (String.length s - 1))

let local (r : Tensor_ref.t) = Id.Tensor_local.to_string r.local_id

let extract (m : graph) (plan : Ex.Extraction.t) :
    (body, Ex.Extract_error.t) Err.t =
  let nodes =
    List.fold_right
      (fun (n : Node_desc.t) acc ->
        match acc with
        | None -> None
        | Some acc -> (
            let orig =
              List.find_map
                (fun ((o : Node_ref.t), (e : Node_ref.t)) ->
                  if Node_ref.equal e n.node then Some o else None)
                plan.descriptor.original_to_extracted_nodes
            in
            match Option.bind orig index_of_local with
            | Some i when i >= 0 && i < Array.length m.nodes ->
                Some ((i, m.nodes.(i)) :: acc)
            | _ -> None))
      plan.body.nodes (Some [])
  in
  match nodes with
  | None -> efail ~code:Unknown_member plan
  | Some nodes ->
      let scope =
        match plan.body.nodes with
        | n :: _ -> n.node.scope
        | [] -> ( match plan.inputs with t :: _ -> t.scope | [] -> [])
      in
      Ok
        {
          model = m;
          key = plan.body.key;
          scope;
          nodes;
          inputs = List.map local plan.inputs;
          outputs = List.map local plan.outputs;
        }

let describe_body (b : body) : Graph_desc.t =
  let sources =
    List.map
      (fun name ->
        let spec = List.assoc name b.model.specs in
        let role =
          match Name.User_input.of_string name with
          | Ok n -> Input_role.User_input n
          | Error _ -> invalid_arg "Aten_dialect.describe_body"
        in
        { Aten_model.name; role; spec; captured = None })
      b.inputs
  in
  match
    Aten_model.describe ~key:b.key ~scope:b.scope ~specs:b.model.specs
      ~nodes:b.nodes ~sources ~outputs:b.outputs
  with
  | Ok d -> d
  | Error m -> invalid_arg ("Aten_dialect.describe_body: " ^ m)

let validate (b : body) =
  if b.nodes = [] then
    Ex.Extract_error.fail (Ex.Extract_error.make Unsupported_effect)
  else Ok ()

module Dialect : Ex.Dialect.S with type graph = graph and type body = body =
struct
  type nonrec graph = graph
  type nonrec body = body

  let dialect = dialect
  let describe = describe
  let extract = extract
  let describe_body = describe_body
  let validate = validate
end
