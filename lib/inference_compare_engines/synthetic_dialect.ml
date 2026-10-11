(* The synthetic dialect: an ordinary graph is a {!Graph_desc.t} whose
   operators are named arithmetic over int64 tensors. Extraction realizes the
   plan by copying the planned body; the optional fault models a dialect tool
   that gets the inventory wrong, which reconciliation must catch. *)

open Inference_compare_schema
module Ex = Inference_compare_extract

type fault =
  | Drop_last_output
  | Expose_extra_output  (** An undeclared internal result becomes an output. *)
  | Rename_operator
  | Reorder_outputs

let dialect_id =
  match Id.Dialect.of_string "synthetic" with
  | Ok d -> d
  | Error _ -> invalid_arg "Synthetic_dialect"

let make ?fault () :
    (module Ex.Dialect.S
       with type graph = Graph_desc.t
        and type body = Graph_desc.t) =
  (module struct
    type graph = Graph_desc.t
    type body = Graph_desc.t

    let dialect = dialect_id
    let describe g = g
    let describe_body b = b

    let extract _ (plan : Ex.Extraction.t) =
      let b = plan.body in
      let b =
        match fault with
        | None -> b
        | Some Drop_last_output ->
            let outs = List.rev (List.tl (List.rev b.final_outputs)) in
            {
              b with
              final_outputs = outs;
              output_tree =
                Tuple (List.map (fun t -> Graph_desc.Output_tree.Tensor t) outs);
            }
        | Some Expose_extra_output -> (
            match
              List.concat_map (fun (n : Node_desc.t) -> n.results) b.nodes
            with
            | extra :: _
              when not (List.exists (Tensor_ref.equal extra) b.final_outputs) ->
                let outs = b.final_outputs @ [ extra ] in
                {
                  b with
                  final_outputs = outs;
                  output_tree =
                    Tuple
                      (List.map (fun t -> Graph_desc.Output_tree.Tensor t) outs);
                }
            | _ -> b)
        | Some Rename_operator -> (
            match b.nodes with
            | n :: rest -> (
                match Id.Operator.of_string "renamed" with
                | Ok op ->
                    {
                      b with
                      nodes =
                        { n with operator = { n.operator with operation = op } }
                        :: rest;
                    }
                | Error _ -> b)
            | [] -> b)
        | Some Reorder_outputs ->
            let outs = List.rev b.final_outputs in
            {
              b with
              final_outputs = outs;
              output_tree =
                Tuple (List.map (fun t -> Graph_desc.Output_tree.Tensor t) outs);
            }
      in
      Ok b

    let validate _ = Ok ()
  end)
