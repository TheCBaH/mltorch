(* An exported model as the ATen side sees it: the exporter graph described as
   an ordinary graph of tensor dataflow (tensors are SSA names, nodes are
   numbered by position), with the node table kept so a cluster of nodes can be
   run again by the ATen interpreter. Attributes are not in the description;
   they travel with the nodes and are folded into the extraction digest. *)

open Inference_compare_schema
open Pytorch_types
module W = Wire

type source = {
  name : string;
  role : Input_role.t;
  spec : Tensor_spec.t;
  captured : string option;  (** The archive entry holding a constant. *)
}

type t = {
  archive : Pt2_archive.t;
  graph : Graph.t;
  nodes : Node.t array;
  sources : source list;  (** Only those something reads. *)
  specs : (string * Tensor_spec.t) list;
  key : Graph_key.t;
  desc : Graph_desc.t;
}

let ( let* ) = Result.bind
let text r = Result.map_error (fun _ -> "invalid identifier") r

let tensor_names_of_arg (a : Argument.t) =
  match a with
  | Tensor t -> [ t.TensorArgument.name ]
  | Tensors ts -> List.map (fun (t : TensorArgument.t) -> t.name) ts
  | Optional_tensor (OptionalTensorArgument.Tensor t) -> [ t.TensorArgument.name ]
  | Optional_tensors ts ->
      List.filter_map
        (function OptionalTensorArgument.Tensor t -> Some t.TensorArgument.name | _ -> None)
        ts
  | _ -> []

let operands (n : Node.t) =
  List.concat_map (fun (na : NamedArgument.t) -> tensor_names_of_arg na.arg) n.inputs

let results (n : Node.t) = Interp_decode.output_names n

let is_assertion (n : Node.t) =
  n.target = "torch.ops.aten._assert_tensor_metadata.default"

(* ---- descriptions -------------------------------------------------------- *)

let tensor_ref ~key ~scope name : (Tensor_ref.t, string) result =
  let* local_id = text (Id.Tensor_local.of_string name) in
  Ok { Tensor_ref.graph = key; scope; local_id }

let node_ref ~key ~scope index : (Node_ref.t, string) result =
  let* local_id = text (Id.Node_local.of_string (Printf.sprintf "n%d" index)) in
  Ok { Node_ref.graph = key; scope; local_id }

let rec all f = function
  | [] -> Ok []
  | x :: rest ->
      let* y = f x in
      let* ys = all f rest in
      Ok (y :: ys)

let node_desc ~key ~scope (index, (n : Node.t)) : (Node_desc.t, string) result =
  let* node = node_ref ~key ~scope index in
  let* operation = text (Id.Operator.of_string n.target) in
  let* dialect = text (Id.Dialect.of_string "aten") in
  let* operands = all (tensor_ref ~key ~scope) (operands n) in
  let* results = all (tensor_ref ~key ~scope) (results n) in
  Ok
    {
      Node_desc.node;
      operator = { dialect; operation };
      operands;
      results;
      control_predecessors = [];
      effect = (if is_assertion n then Metadata_assertion else Pure);
    }

let describe ~key ~scope ~specs ~(nodes : (int * Node.t) list) ~(sources : source list)
    ~(outputs : string list) : (Graph_desc.t, string) result =
  let* node_descs = all (node_desc ~key ~scope) nodes in
  let* srcs =
    all
      (fun (s : source) ->
        let* tensor = tensor_ref ~key ~scope s.name in
        Ok { Graph_source.tensor; role = s.role; spec = s.spec })
      sources
  in
  let* final_outputs = all (tensor_ref ~key ~scope) outputs in
  let produced = List.concat_map (fun (_, n) -> results n) nodes in
  let* tensor_specs =
    all
      (fun name ->
        let* r = tensor_ref ~key ~scope name in
        match List.assoc_opt name specs with
        | Some s -> Ok (r, s)
        | None -> Error ("no spec for tensor " ^ name))
      produced
  in
  Ok
    {
      Graph_desc.key;
      nodes = node_descs;
      sources = srcs;
      final_outputs;
      output_tree = Tuple (List.map (fun t -> Graph_desc.Output_tree.Tensor t) final_outputs);
      tensor_specs;
    }

(* ---- loading ------------------------------------------------------------- *)

let spec_of_meta (m : TensorMeta.t) =
  let* dtype =
    match Dtypes.of_scalar_type m.dtype with
    | Some d -> Ok d
    | None -> Error "unsupported element type"
  in
  let* shape =
    all
      (function SymInt.Int n -> Ok (Int64.of_int n) | _ -> Error "symbolic size")
      m.sizes
  in
  match Dtypes.spec ~dtype ~shape with Some s -> Ok s | None -> Error "unsupported element type"

let fingerprint ~hashers ~algorithm text =
  match Inference_compare_codecs.Payload_hasher.digest_string hashers algorithm text with
  | Ok d -> Ok (Digest.Payload.hex d)
  | Error _ -> Error "digest backend unavailable"

let of_archive ~hashers ~algorithm ~identity_text archive : (t, string) result =
  let program = Pt2_archive.program archive in
  let graph = program.ExportedProgram.graph_module.GraphModule.graph in
  let signature = program.ExportedProgram.graph_module.GraphModule.signature in
  let nodes = Array.of_list graph.Graph.nodes in
  let* specs =
    all
      (fun (name, m) ->
        match spec_of_meta m with Ok s -> Ok (Some (name, s)) | Error _ -> Ok None)
      (Schema_runtime.String_map.bindings graph.tensor_values)
    |> Result.map (List.filter_map Fun.id)
  in
  let spec name = List.assoc_opt name specs in
  let read =
    List.concat_map operands (Array.to_list nodes)
    @ List.concat_map tensor_names_of_arg graph.outputs
  in
  let* sources =
    all
      (fun (s : InputSpec.t) ->
        let make name role captured =
          if not (List.mem name read) then Ok None
          else
            match spec name with
            | None -> Error ("source " ^ name ^ " has no supported tensor type")
            | Some spec -> Ok (Some { name; role; spec; captured })
        in
        match s with
        | InputSpec.User_input { UserInputSpec.arg = Argument.Tensor a } ->
            let* n = text (Name.User_input.of_string a.TensorArgument.name) in
            make a.TensorArgument.name (Input_role.User_input n) None
        | InputSpec.Parameter p ->
            let name = p.arg.TensorArgument.name in
            let* n = text (Name.Constant.of_string name) in
            make name (Input_role.Constant n) (Some p.parameter_name)
        | InputSpec.Buffer p ->
            let name = p.arg.TensorArgument.name in
            let* n = text (Name.Constant.of_string name) in
            make name (Input_role.Constant n) (Some p.buffer_name)
        | InputSpec.Tensor_constant p ->
            let name = p.arg.TensorArgument.name in
            let* n = text (Name.Constant.of_string name) in
            make name (Input_role.Constant n) (Some p.tensor_constant_name)
        | _ -> Error "unsupported non-tensor graph input")
      signature.GraphSignature.input_specs
    |> Result.map (List.filter_map Fun.id)
  in
  let outputs = List.concat_map tensor_names_of_arg graph.outputs in
  let* hex = fingerprint ~hashers ~algorithm identity_text in
  let* graph_digest =
    Digest.Graph.make algorithm ~hex |> Result.map_error (fun _ -> "digest")
  in
  let* dialect = text (Id.Dialect.of_string "aten") in
  let* version = text (Id.Graph_version.of_string "pt2-export") in
  let key = { Graph_key.dialect; version; graph_digest } in
  let* desc =
    describe ~key ~scope:[] ~specs
      ~nodes:(List.mapi (fun i n -> (i, n)) (Array.to_list nodes))
      ~sources ~outputs
  in
  Ok { archive; graph; nodes; sources; specs; key; desc }

(* The text that stands for one node in a digest: its target and every
   argument, so two nodes that differ only in a stride or an axis differ here
   although the dataflow description cannot tell them apart. *)
let node_attributes (n : Node.t) =
  (* The generated PT2 types decode but do not encode, and have no printer.
     [No_sharing] makes the bytes a function of the value alone, not of how
     the decoder happened to share substructure. *)
  Marshal.to_string n [ Marshal.No_sharing ]

(* Open a [.pt2] file: the archive, plus the exporter's own graph document as
   the identity of the ATen endpoint. *)
let of_file ~hashers ~algorithm path : (t, string) result =
  let* contents =
    match Pt2_archive.read_file path with
    | Ok c -> Ok c
    | Error e -> Error (Format.asprintf "%s: %a" path Pt2_archive.pp_error (Err.Error.kind e))
  in
  let* archive =
    match Pt2_archive.of_string ~name:path contents with
    | Ok a -> Ok a
    | Error e -> Error (Format.asprintf "%s: %a" path Pt2_archive.pp_error (Err.Error.kind e))
  in
  let* identity_text =
    match Pt2_zip.of_string contents with
    | Error _ -> Error "not a zip archive"
    | Ok zip -> (
        match Pt2_zip.read_rel_required zip "models/model.json" with
        | Ok t -> Ok t
        | Error _ -> Error "no models/model.json")
  in
  of_archive ~hashers ~algorithm ~identity_text archive
