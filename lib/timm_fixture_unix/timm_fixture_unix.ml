open Err.Syntax
module F = Timm_fixture
module J = F.Json
module I = F.Index
module U = Pt2_fixture_unix

let ( >>= ) = Err.bind
let wrap pp r = Err.map_error (fun e -> `Timm_fixture (Fmt.str "%a" pp e)) r

let read ?(max_bytes = 8388608) path =
  U.Fetch.read ~max_bytes path |> wrap U.Fault.pp_error

let io path f =
  match f () with
  | value -> Ok value
  | exception Sys_error why -> J.bad (path ^ ": " ^ why)
  | exception Unix.Unix_error (e, _, _) ->
      J.bad (path ^ ": " ^ Unix.error_message e)

module Discovery = struct
  type t = Absent | V2 of I.t

  let classify ~status text =
    match status with
    | "404" -> Ok Absent
    | "200" ->
        let+ index = I.of_string text in
        V2 index
    | _ -> J.bad ("v2 discovery failed: " ^ status)

  (* Only an HTTP 404 permits legacy fallback. A failed request or invalid
     published document must never turn into an inferred v1 fixture. *)
  let remote ~url =
    let* dest = io url (fun () -> Filename.temp_file "timm-index" ".json") in
    Fun.protect
      ~finally:(fun () -> Sys.remove dest)
      (fun () ->
        let args =
          [|
            "curl";
            "--silent";
            "--show-error";
            "--location";
            "--proto";
            "=https";
            "--proto-redir";
            "=https";
            "--max-redirs";
            "5";
            "--max-filesize";
            "8388608";
            "--output";
            dest;
            "--write-out";
            "%{http_code}";
            "--";
            url;
          |]
        in
        let* status, process =
          io url (fun () ->
              let ic = Unix.open_process_args_in "curl" args in
              let text = In_channel.input_all ic in
              (String.trim text, Unix.close_process_in ic))
        in
        match (process, status) with
        | Unix.WEXITED 0, "404" -> classify ~status ""
        | Unix.WEXITED 0, "200" ->
            let* text = read dest in
            classify ~status text
        | _ -> J.bad ("v2 discovery failed: " ^ status))

  let local path =
    let* text = read path in
    let+ index = I.of_string text in
    V2 index
end

let limits ~entries ~member ~total =
  Pt2_zip.Limits.create ~max_entries:entries ~max_entry_bytes:member
    ~max_total_bytes:total ~max_path_bytes:256 ~max_path_depth:8
  |> wrap Pt2_zip.Limits.pp_error

let tensors path =
  let* limits = limits ~entries:1024 ~member:134217728L ~total:536870912L in
  let* tensors =
    Pt2_archive.load_pt_tensor_map ~limits ~max_bytes:536870912L path
    |> wrap Pt2_archive.pp_error
  in
  let* () =
    if tensors = [] || List.length tensors > 512 then
      J.bad "unsupported tensor count"
    else Ok ()
  in
  let* () = J.unique (List.map fst tensors) in
  Ok tensors

module Evaluation = struct
  type t = {
    cases : F.Case.t list;
    contract : F.Contract.t;
    outputs : (string * Pt2_tensor.t) list;
    preprocessing : Jsont.json;
  }

  let members =
    [
      "cases.json";
      "contract-v2.json";
      "expected.json";
      "outputs.pt";
      "preprocessing.json";
    ]

  let open_ path indexed_members =
    let* () =
      J.same "indexed evaluation members"
        (List.sort String.compare indexed_members)
        members
    in
    let* profile = limits ~entries:5 ~member:16777216L ~total:16777216L in
    let* bytes = read ~max_bytes:16777216 path in
    let* zip =
      Pt2_zip.of_string ~limits:profile bytes |> wrap Pt2_zip.pp_error
    in
    let* () =
      J.same "evaluation members"
        (List.sort String.compare (Pt2_zip.entries zip))
        members
    in
    let member name = Pt2_zip.read_required zip name |> wrap Pt2_zip.pp_error in
    let* contract = member "contract-v2.json" >>= F.Contract.of_string in
    let* cases = member "cases.json" >>= F.Case.of_string in
    let* preprocessing = member "preprocessing.json" >>= J.parse in
    let* recipe = J.member "recipe" preprocessing in
    let* () = J.version 1 recipe in
    let* output_bytes = member "outputs.pt" in
    let* tensor_limits =
      limits ~entries:1024 ~member:134217728L ~total:536870912L
    in
    let+ outputs =
      Pt2_archive.pt_tensor_map_of_string ~limits:tensor_limits
        ~name:"outputs.pt" output_bytes
      |> wrap Pt2_archive.pp_error
    in
    { cases; contract; outputs; preprocessing }
end

module Config = struct
  type t = {
    cache : U.Cache.t;
    on_transfer : (string -> int64 -> unit) option;
    source_dir : string option;
    transport : U.Transport.t option;
  }

  let create ?source_dir ?transport ?on_transfer root =
    let+ cache = U.Cache.create root |> wrap U.Fault.pp_error in
    { cache; on_transfer; source_dir; transport }
end

let ensure config ~layer pin =
  let transport =
    match config.Config.source_dir with
    | None -> config.transport
    | Some dir ->
        Some
          (fun ~url:_ ~dest ->
            let path =
              Filename.concat dir pin.Pt2_checkpoint_map.Document.Pin.name
            in
            if not (Sys.file_exists path) then
              match config.transport with
              | None -> Error ("offline: " ^ path)
              | Some transport -> transport ~url:pin.url ~dest
            else
              match
                In_channel.with_open_bin path (fun ic ->
                    Out_channel.with_open_bin dest (fun oc ->
                        let buf = Bytes.create 65536 in
                        let rec copy () =
                          let n = input ic buf 0 65536 in
                          if n > 0 then (
                            output oc buf 0 n;
                            copy ())
                        in
                        copy ()))
              with
              | () -> Ok ()
              | exception Sys_error e -> Error e)
  in
  let transport =
    Option.map
      (fun transport ~url ~dest ->
        let result = transport ~url ~dest in
        let bytes =
          try (Unix.LargeFile.stat dest).st_size with Unix.Unix_error _ -> 0L
        in
        Option.iter (fun f -> f url bytes) config.on_transfer;
        result)
      transport
  in
  U.Fetch.ensure ?transport ~layer config.cache pin |> wrap U.Fault.pp_error

let asset config index id =
  let* a = I.asset_of index id in
  let+ path =
    ensure config ~layer:(Pt2_fixture.Fault.Member id) a.I.Asset.pin
  in
  (a, path)

module Bundle = struct
  type t = {
    evaluation : Evaluation.t;
    graph_bytes : string;
    inputs : (string * Pt2_tensor.t) list;
    model : I.Model.t;
    program : Pytorch_types.ExportedProgram.t;
  }

  let shared_inputs config index evaluation =
    let* _, metadata_path = asset config index "image_metadata" in
    let* metadata = read metadata_path >>= J.parse in
    let* _, images_path = asset config index "decoded_images" in
    let* images = tensors images_path in
    let* () = F.Images.check ~index ~metadata images in
    let* recipe =
      F.Preprocess.Recipe.of_json evaluation.Evaluation.preprocessing
    in
    let* binding =
      match evaluation.contract.inputs with
      | [ binding ] when binding.sample_layout = "CHW" -> Ok binding
      | _ -> J.bad "image preprocessing requires one CHW image binding"
    in
    let* inputs =
      Err.List.map
        (fun case ->
          let* image =
            Err.of_option
              (`Timm_fixture ("missing decoded image " ^ case.F.Case.image))
              (List.assoc_opt case.image images)
          in
          let* reference =
            Err.of_option (`Timm_fixture "missing image input reference")
              (List.assoc_opt binding.sample case.inputs)
          in
          let* input = F.Preprocess.run recipe image ~geometry:case.geometry in
          let+ () = F.Descriptor.check reference.descriptor input in
          (reference.key, input))
        evaluation.cases
    in
    let+ () = J.unique (List.map fst inputs) in
    inputs

  let open_ ?(shared = false) config index name =
    let* model =
      Err.of_option
        (`Timm_fixture ("unknown current model " ^ name))
        (List.assoc_opt name index.I.models)
    in
    let* eval_asset, path = asset config index model.evaluation in
    let* evaluation = Evaluation.open_ path eval_asset.members in
    let* graph_path =
      ensure config ~layer:Pt2_fixture.Fault.Graph model.graph
    in
    let* graph_bytes = read graph_path in
    let* program =
      Pt2_archive.program_of_json graph_bytes |> wrap Pt2_archive.pp_error
    in
    let* () = F.Contract.check_graph evaluation.contract ~graph_bytes program in
    let* () = J.same "contract model" name evaluation.contract.model in
    let* () =
      J.same "input asset" model.inputs evaluation.contract.input_asset
    in
    let* decoded = I.asset_of index "decoded_images" in
    let* () =
      J.same "decoded image identity" decoded.pin.sha256
        evaluation.contract.decoded_sha256
    in
    let* inputs =
      if shared then shared_inputs config index evaluation
      else
        let* _, input_path = asset config index model.inputs in
        tensors input_path
    in
    let+ () =
      F.check_cases evaluation.contract evaluation.cases ~inputs
        ~outputs:evaluation.outputs
    in
    { evaluation; graph_bytes; inputs; model; program }

  let archive config bundle ~model_dir =
    let* bytes = read (Filename.concat model_dir "models/model.json") in
    let* () =
      J.same "checkout graph identity" (Pt2_sha256.string bytes)
        bundle.model.graph.sha256
    in
    let* map_pin =
      Err.of_option (`Timm_fixture "missing weight map") bundle.model.weight_map
    in
    let* map_bytes =
      read (Filename.concat model_dir "models/safetensors.json")
    in
    let* () =
      J.same "checkout weight map byte count"
        (Int64.of_int (String.length map_bytes))
        map_pin.size
    in
    let* () =
      J.same "checkout weight map identity"
        (Pt2_sha256.string map_bytes)
        map_pin.sha256
    in
    let* _ = ensure config ~layer:Pt2_fixture.Fault.Map map_pin in
    Pt2_safetensors_unix.open_dir model_dir
    |> wrap Pt2_safetensors_unix.pp_error
end
