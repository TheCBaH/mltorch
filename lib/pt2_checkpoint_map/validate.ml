open Schema_runtime
open Pytorch_types
open Pytorch_weights_config
open Err.Syntax

type graph = {
  artifact_id : string;
  captures : Captures.t;
  constants : ModelWeightsConfig.t;
  graph_digest : Pt2_sha256.Digest.t;
  program : ExportedProgram.t;
  weights : ModelWeightsConfig.t;
}

let signature_captures (program : ExportedProgram.t) =
  let specs = program.graph_module.signature.input_specs in
  let captured =
    List.filter_map
      (function
        | InputSpec.Parameter p -> Some (p.parameter_name, Fault.Parameter)
        | InputSpec.Buffer b -> Some (b.buffer_name, Fault.Buffer)
        | InputSpec.Tensor_constant c ->
            Some (c.tensor_constant_name, Fault.Constant_tensor)
        | InputSpec.Constant_input _ | InputSpec.Custom_obj _
        | InputSpec.Token _ | InputSpec.User_input _ ->
            None)
      specs
  in
  let seen = Hashtbl.create 64 in
  let+ () =
    Err.List.iter
      (fun (target, _) ->
        if Hashtbl.mem seen target then
          Err.fail (`Duplicate (Fault.Capture, target))
        else begin
          Hashtbl.add seen target ();
          Err.return ()
        end)
      captured
  in
  captured

let supported_conversions (doc : Document.t) =
  Err.List.iter
    (fun (target, (e : Document.Entry.t)) ->
      match e.origin with
      | Document.Origin.Checkpoint
          {
            convert =
              Document.Origin.Cast
                { from = Dtype.BF16 | Dtype.F16; to_ = Dtype.F32 };
            _;
          } ->
          Err.return ()
      | Document.Origin.Checkpoint
          { convert = Document.Origin.Cast { from; to_ }; _ } ->
          Err.fail (`Unsupported_conversion { Fault.Cast.target; from; to_ })
      | Checkpoint { convert = Identity; _ }
      | Empty | Fill _ | Inline _ | Pack _ ->
          Err.return ())
    (String_map.bindings doc.tensors)

(* [computed] is what the graph bytes hash to; [pinned] what a document says. *)
let check_graph_digest document ~pinned ~computed =
  if Pt2_sha256.Digest.equal pinned computed then Err.return ()
  else
    Err.fail
      (`Graph_digest_mismatch
         (document, { Fault.Pair.actual = computed; expected = pinned }))

module String_set = Set.Make (String)

(* The first element of [a] absent from [b]. *)
let first_missing a b =
  let b = String_set.of_list b in
  List.find_opt (fun x -> not (String_set.mem x b)) a

let sorted_keys m = List.map fst (String_map.bindings m)

let config_entry (g : graph) target =
  match
    ( String_map.find_opt target g.weights.ModelWeightsConfig.config,
      String_map.find_opt target g.constants.ModelWeightsConfig.config )
  with
  | Some _, Some _ -> Err.fail (`Duplicate (Fault.Capture, target))
  | Some e, None | None, Some e -> Err.return e
  | None, None -> Err.fail (`Config_missing target)

let int64s = List.map Int64.of_int

let check_capture (g : graph) (doc : Document.t) ~inventory (target, kind) =
  let entry = String_map.find target doc.tensors in
  let (cap : Captures.Capture.t) = String_map.find target inventory in
  let* () =
    if cap.kind = kind then Err.return ()
    else
      Err.fail
        (`Kind_clash
           {
             Fault.Clash.target;
             against = Fault.Signature;
             expected = kind;
             actual = cap.kind;
           })
  in
  let* (cfg : WeightEntry.t) = config_entry g target in
  let* tensor =
    Pt2_tensor.of_meta cfg.tensor_meta ~data:Pt2_storage.empty
    |> Err.map_error ~pos:__POS__ (fun e ->
        `Graph_tensor (target, (e : Pt2_tensor.error)))
  in
  let graph_dtype = Dtype.of_pt2 tensor.dtype in
  let* () =
    if Dtype.equal graph_dtype entry.dtype then Err.return ()
    else
      Err.fail
        (`Dtype_clash
           {
             Fault.Clash.target;
             against = Fault.Config;
             expected = graph_dtype;
             actual = entry.dtype;
           })
  in
  let* () =
    if Dtype.equal cap.dtype entry.dtype then Err.return ()
    else
      Err.fail
        (`Dtype_clash
           {
             Fault.Clash.target;
             against = Fault.Inventory;
             expected = cap.dtype;
             actual = entry.dtype;
           })
  in
  let graph_shape = int64s tensor.sizes in
  let shape_clash against expected =
    Err.fail
      (`Shape_clash
         { Fault.Clash.target; against; expected; actual = entry.shape })
  in
  let* () =
    if List.equal Int64.equal graph_shape entry.shape then Err.return ()
    else shape_clash Fault.Config graph_shape
  in
  let* () =
    if List.equal Int64.equal cap.shape entry.shape then Err.return ()
    else shape_clash Fault.Inventory cap.shape
  in
  (* [is_contiguous] folds the sizes with an overflow check that raises for a
     tensor no buffer could hold; that is a layout nothing can match either. *)
  let* () =
    match Pt2_tensor.is_contiguous tensor with
    | true -> Err.return ()
    | false -> Err.fail (`Graph_layout target)
    | exception Invalid_argument _ -> Err.fail (`Graph_layout target)
  in
  if Pt2_sha256.Digest.equal cap.value_sha256 entry.sha256 then Err.return ()
  else
    Err.fail
      (`Value_digest_clash
         {
           Fault.Clash.target;
           against = Fault.Inventory;
           expected = cap.value_sha256;
           actual = entry.sha256;
         })

let check (doc : Document.t) (g : graph) =
  let mismatch document actual expected =
    Err.fail (`Artifact_mismatch (document, { Fault.Pair.actual; expected }))
  in
  let* () =
    if String.equal doc.artifact_id g.artifact_id then Err.return ()
    else mismatch Fault.Map doc.artifact_id g.artifact_id
  in
  let* () =
    if String.equal g.captures.artifact_id g.artifact_id then Err.return ()
    else mismatch Fault.Captures_json g.captures.artifact_id g.artifact_id
  in
  let* () =
    check_graph_digest Fault.Map ~pinned:doc.graph_sha256
      ~computed:g.graph_digest
  in
  let* () =
    check_graph_digest Fault.Captures_json ~pinned:g.captures.graph_sha256
      ~computed:g.graph_digest
  in
  let* signature = signature_captures g.program in
  let signature =
    List.sort (fun (a, _) (b, _) -> String.compare a b) signature
  in
  let signature_targets = List.map fst signature in
  let inventory =
    List.fold_left
      (fun m (c : Captures.Capture.t) -> String_map.add c.target c m)
      String_map.empty g.captures.captures
  in
  let inventory_targets = sorted_keys inventory in
  let* () =
    match first_missing signature_targets inventory_targets with
    | Some t -> Err.fail (`Inventory_missing t)
    | None -> Err.return ()
  in
  let* () =
    match first_missing inventory_targets signature_targets with
    | Some t -> Err.fail (`Inventory_surplus t)
    | None -> Err.return ()
  in
  let map_targets = sorted_keys doc.tensors in
  let* () =
    match first_missing signature_targets map_targets with
    | Some t -> Err.fail (`Missing_tensor t)
    | None -> Err.return ()
  in
  let* () =
    match first_missing map_targets signature_targets with
    | Some t -> Err.fail (`Surplus_tensor t)
    | None -> Err.return ()
  in
  let config_targets =
    sorted_keys g.weights.ModelWeightsConfig.config
    @ sorted_keys g.constants.ModelWeightsConfig.config
  in
  let* () =
    match first_missing config_targets signature_targets with
    | Some t -> Err.fail (`Config_surplus t)
    | None -> Err.return ()
  in
  let* () = Err.List.iter (check_capture g doc ~inventory) signature in
  supported_conversions doc
