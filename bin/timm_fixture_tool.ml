open Err.Syntax
module F = Timm_fixture
module J = F.Json
module U = Timm_fixture_unix

let ( >>= ) = Err.bind
let env name default = Option.value (Sys.getenv_opt name) ~default

let discovery () =
  match Sys.getenv_opt "TIMM_MANIFEST" with
  | Some path when path <> "" -> U.Discovery.local path
  | _ ->
      U.Discovery.remote
        ~url:
          (env "TIMM_MANIFEST_URL"
             (Printf.sprintf
                "https://github.com/%s/releases/download/%s/manifest-v2.json"
                (env "TIMM_REPO" "TheCBaH/devcontainer.pytorch-image-models")
                (env "TIMM_RELEASE" "v0.0.6")))

let config () =
  let source_dir =
    match Sys.getenv_opt "TIMM_SOURCE_DIR" with
    | Some "" -> None
    | value -> value
  in
  let transport =
    if Sys.getenv_opt "TIMM_OFFLINE" = Some "1" then None
    else Some Pt2_fixture_unix.Transport.curl
  in
  let on_transfer =
    Option.map
      (fun path url bytes ->
        let oc =
          open_out_gen
            [ Open_append; Open_binary; Open_creat; Open_wronly ]
            0o600 path
        in
        Fun.protect
          ~finally:(fun () -> close_out oc)
          (fun () -> Printf.fprintf oc "%Ld\t%s\n" bytes url))
      (Sys.getenv_opt "TIMM_TRANSFER_LOG")
  in
  U.Config.create ?source_dir ?transport ?on_transfer
    (env "TIMM_CACHE" "data/timm-cache")

let bundle ~shared name =
  let* index = discovery () in
  match index with
  | U.Discovery.Absent -> Ok None
  | V2 index ->
      let* config = config () in
      let+ bundle = U.Bundle.open_ ~shared config index name in
      Some (config, index, bundle)

let archive config bundle =
  U.Bundle.archive config bundle
    ~model_dir:
      (Filename.concat
         (env "MODELS_DIR" "modules/devcontainer.pytorch-image-models/models")
         bundle.U.Bundle.evaluation.contract.model)

let replay config bundle =
  let* archive = archive config bundle in
  let evaluation = bundle.U.Bundle.evaluation in
  Err.List.iter
    (fun case ->
      let* inputs = F.bind_inputs evaluation.contract case bundle.inputs in
      let* actual =
        Native_interp.run_named archive ~inputs
        |> U.wrap J.External.Source.Native Native_interp.pp_error
      in
      let* () =
        Err.List.iter2
          ~unequal_lengths:(fun _ _ ->
            `Timm_fixture J.Failure.Output_arity_differs)
          (fun binding native ->
            let* logical =
              Pt2_fixture_replay.to_logical native
                ~rank:(List.length binding.F.Binding.graph_shape)
              |> Err.import ~pos:__POS__ (fun (`Native_layout why) ->
                  `Native_layout why)
            in
            let reference = List.assoc binding.sample case.F.Case.outputs in
            let tensor =
              List.assoc reference.F.Reference.key evaluation.outputs
            in
            F.compare_output evaluation.contract binding tensor logical)
          evaluation.contract.outputs actual
      in
      Printf.printf "%s/%s: all outputs pass\n%!" evaluation.contract.model
        case.id;
      Ok ())
    evaluation.cases

let verify_patterns dir =
  let* text = U.read (Filename.concat dir "fixtures.json") in
  let* j = J.parse text in
  let* () = J.version 1 j in
  let* images = U.tensors (Filename.concat dir "images.pt") in
  let* fixtures = J.member "fixtures" j >>= J.array in
  Err.List.iter
    (fun j ->
      let* name = J.field "name" j in
      let* preprocessing = J.member "preprocessing" j in
      let* recipe = F.Preprocess.Recipe.of_json preprocessing in
      let* inputs = U.tensors (Filename.concat dir (name ^ "-inputs.pt")) in
      let* cases = J.member "cases" j >>= J.array in
      Err.List.iter
        (fun j ->
          let* image = J.field "image" j in
          let* geometry = J.member "geometry" j in
          let* descriptor = J.member "tensor" j >>= F.Descriptor.of_json in
          let* actual =
            F.Preprocess.run recipe (List.assoc image images) ~geometry
          in
          let* () = F.Descriptor.check descriptor (List.assoc image inputs) in
          let* () = F.Descriptor.check descriptor actual in
          Printf.printf "%s/%s: exact bits and geometry\n%!" name image;
          Ok ())
        cases)
    fixtures

let run () =
  match Array.to_list Sys.argv with
  | [ _; "verify-inputs-all" ] -> (
      let* discovered = discovery () in
      match discovered with
      | U.Discovery.Absent ->
          J.bad J.Failure.V2_manifest_required_for_input_verification
      | V2 index ->
          let* config = config () in
          let+ () =
            Err.List.iter
              (fun (name, _) ->
                let* bundle = U.Bundle.open_ ~shared:true config index name in
                Printf.printf "%s: %d cases, exact bits and geometry\n%!" name
                  (List.length bundle.evaluation.cases);
                Ok ())
              index.models
          in
          0)
  | [ _; "verify-patterns"; dir ] ->
      let+ () = verify_patterns dir in
      0
  | [
   _;
   (("download" | "replay" | "replay-images" | "verify-inputs") as command);
   name;
  ] -> (
      let shared =
        command = "replay-images" || command = "verify-inputs"
        || (command = "download" && Sys.getenv_opt "TIMM_MODE" = Some "images")
      in
      let* result = bundle ~shared name in
      match result with
      | None -> Ok 42
      | Some (config, _, bundle) ->
          let+ () =
            if command = "download" then
              let+ _ = archive config bundle in
              ()
            else if command = "verify-inputs" then (
              print_endline (name ^ ": exact decoded-image input hashes");
              Ok ())
            else replay config bundle
          in
          0)
  | _ -> J.bad J.Failure.Usage

let () =
  let status =
    match Err.export ~pos:__POS__ (run ()) with
    | Ok status -> status
    | Error e ->
        prerr_endline (Fmt.str "%a" F.pp_error e);
        2
  in
  exit status
