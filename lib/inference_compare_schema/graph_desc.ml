module W = Wire

module Output_tree = struct
  type t = Tensor of Tensor_ref.t | Tensor_list of t list | Tuple of t list

  (* Leaves in structural order: the occurrence order of the final outputs. *)
  let rec flatten = function
    | Tensor r -> [ r ]
    | Tensor_list xs | Tuple xs -> List.concat_map flatten xs

  (* A [Tensor_list] holds tensor leaves only; a [Tuple] may nest. *)
  let rec decode : t W.decoder =
   fun j ->
    W.variant
      (fun tag o ->
        let ( let* ) = W.( let* ) in
        match tag with
        | "tensor" ->
            let* r = W.Obj.required o "tensor" Tensor_ref.decode in
            Ok (Tensor r)
        | "tensor_list" ->
            let* items = W.Obj.required o "items" (W.list decode) in
            if List.for_all (function Tensor _ -> true | _ -> false) items
            then Ok (Tensor_list items)
            else W.fail (W.Constraint Out_of_range)
        | "tuple" ->
            let* items = W.Obj.required o "items" (W.list decode) in
            Ok (Tuple items)
        | tag -> W.unknown_tag ~kind:"output_tree" tag)
      j

  let rec encode = function
    | Tensor r -> W.tagged "tensor" [ W.field "tensor" Tensor_ref.encode r ]
    | Tensor_list xs ->
        W.tagged "tensor_list" [ W.field "items" (W.enc_list encode) xs ]
    | Tuple xs -> W.tagged "tuple" [ W.field "items" (W.enc_list encode) xs ]
end

type t = {
  key : Graph_key.t;
  nodes : Node_desc.t list;
  sources : Graph_source.t list;
  final_outputs : Tensor_ref.t list;
  output_tree : Output_tree.t;
  tensor_specs : (Tensor_ref.t * Tensor_spec.t) list;
}

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* key = W.Obj.required o "key" Graph_key.decode in
      let by_encoding enc v = W.canonical_string (enc v) in
      let* nodes =
        W.Obj.required o "nodes"
          (W.unique
             ~key:(fun (n : Node_desc.t) -> by_encoding Node_ref.encode n.node)
             (W.list Node_desc.decode))
      in
      let* sources =
        W.Obj.required o "sources"
          (W.unique
             ~key:(fun (s : Graph_source.t) ->
               by_encoding Tensor_ref.encode s.tensor)
             (W.list Graph_source.decode))
      in
      let* final_outputs =
        W.Obj.required o "final_outputs" (W.list Tensor_ref.decode)
      in
      let* output_tree = W.Obj.required o "output_tree" Output_tree.decode in
      let* tensor_specs =
        W.Obj.required o "tensor_specs"
          (W.unique
             ~key:(fun (r, _) -> by_encoding Tensor_ref.encode r)
             (W.list (W.pair Tensor_ref.decode Tensor_spec.decode)))
      in
      Ok { key; nodes; sources; final_outputs; output_tree; tensor_specs })

let encode g =
  W.obj
    [
      W.field "key" Graph_key.encode g.key;
      W.field "nodes" (W.enc_list Node_desc.encode) g.nodes;
      W.field "sources" (W.enc_list Graph_source.encode) g.sources;
      W.field "final_outputs" (W.enc_list Tensor_ref.encode) g.final_outputs;
      W.field "output_tree" Output_tree.encode g.output_tree;
      W.field "tensor_specs"
        (W.enc_list (W.enc_pair Tensor_ref.encode Tensor_spec.encode))
        g.tensor_specs;
    ]

(* Node, source and tensor-spec tables are keyed by scoped identity, so their
   order is not part of the graph. Operand, result and final-output order is
   semantic and is kept. *)
let canonicalize g =
  let sort_by key items =
    List.stable_sort (fun a b -> String.compare (key a) (key b)) items
  in
  let node_key (n : Node_desc.t) =
    W.canonical_string (Node_ref.encode n.node)
  in
  let tensor_key r = W.canonical_string (Tensor_ref.encode r) in
  {
    g with
    nodes =
      sort_by node_key
        (List.map
           (fun (n : Node_desc.t) ->
             {
               n with
               Node_desc.control_predecessors =
                 sort_by
                   (fun r -> W.canonical_string (Node_ref.encode r))
                   n.control_predecessors;
             })
           g.nodes);
    sources =
      sort_by (fun (s : Graph_source.t) -> tensor_key s.tensor) g.sources;
    tensor_specs = sort_by (fun (r, _) -> tensor_key r) g.tensor_specs;
  }

let preimage g = Digest_preimage.of_json Graph (encode (canonicalize g))
