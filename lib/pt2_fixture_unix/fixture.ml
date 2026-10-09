open Err.Syntax
module Map = Pt2_checkpoint_map
module Cohort = Pt2_fixture.Cohort

type error = [ Fault.error | Pt2_archive.error | Pt2_checkpoint_map_unix.error ]

let pp_error ppf : error -> unit = function
  | #Fault.error as e -> Fault.pp_error ppf e
  | #Pt2_archive.error as e -> Pt2_archive.pp_error ppf e
  | #Pt2_checkpoint_map_unix.error as e ->
      Pt2_checkpoint_map_unix.pp_error ppf e

type t = {
  archive : Pt2_archive.t;
  bundle : Bundle.t;
  captures : Map.Prepare.t;
  document : Map.Document.t;
}

(* Only the identity; the call contract is read by the replay layer. *)
let artifact_id_jsont =
  Jsont.Object.map ~kind:"contract" Fun.id
  |> Jsont.Object.mem "artifact_id" Jsont.string
  |> Jsont.Object.skip_unknown |> Jsont.Object.finish

let big = 0x2000_0000

(* The map's declared checkpoint files are exactly the ones the cohort pinned,
   byte for byte: the map's own digest is pinned too, so this is a second,
   independent statement of the same facts. *)
let check_sources (entry : Cohort.entry) (doc : Map.Document.t) =
  let* () =
    Err.List.iter
      (fun (s : Map.Document.Source.t) ->
        match
          List.find_opt
            (fun (p : Cohort.Pin.t) -> String.equal p.name s.pin.name)
            entry.sources
        with
        | None -> Err.fail (`Source_pin_missing s.pin.name)
        | Some pinned ->
            Pt2_fixture.Check.pin (Pt2_fixture.Fault.Source s.pin.name) s.pin
              pinned)
      doc.checkpoint_files
  in
  Err.List.iter
    (fun (p : Cohort.Pin.t) ->
      match Map.Document.find_source doc p.name with
      | Some _ -> Err.return ()
      | None -> Err.fail (`Member_missing p.name))
    entry.sources

let of_bundle ?limits (c : Bundle.config) (b : Bundle.t) =
  let read name = Bundle.read_member ~max_bytes:big b name in
  let* contract = read "contract.json" in
  let* artifact_id =
    Jsont_bytesrw.decode_string artifact_id_jsont contract
    |> Err.import ~pos:__POS__ (fun e -> `Contract_decode e)
  in
  let* () =
    Pt2_fixture.Check.field Pt2_fixture.Fault.Contract
      Pt2_fixture.Fault.Artifact_id ~actual:artifact_id
      ~expected:b.entry.artifact_id
  in
  let* model_json = read "models/model.json" in
  let* program = Pt2_archive.program_of_json model_json in
  let* weights_json = read "data/weights/model_weights_config.json" in
  let* weights = Pt2_archive.weights_config_of_json weights_json in
  let* constants_json = read "data/constants/model_constants_config.json" in
  let* constants = Pt2_archive.constants_config_of_json constants_json in
  let* captures_json = read "captures.json" in
  let* inventory = Map.Captures.of_string ?limits captures_json in
  let* map_json = read b.entry.map_member in
  let* document = Map.Document.of_string ?limits map_json in
  let* () = check_sources b.entry document in
  let* () =
    Map.Validate.check document
      {
        Map.Validate.artifact_id = b.entry.artifact_id;
        captures = inventory;
        constants;
        graph_digest = Pt2_sha256.string model_json;
        program;
        weights;
      }
  in
  let declared =
    List.map
      (fun (s : Map.Document.Source.t) -> s.pin)
      document.checkpoint_files
    @ Option.to_list document.graph_owned
  in
  let* sources =
    Err.List.map
      (fun (pin : Map.Document.Pin.t) ->
        let* path =
          Fetch.ensure ?hash:c.hash ?transport:c.transport
            ~layer:(Pt2_fixture.Fault.Source pin.name) c.cache pin
        in
        let+ bytes = Pt2_checkpoint_map_unix.map_file path in
        { Map.Prepare.name = pin.name; bytes })
      declared
  in
  let* verified = Map.Prepare.verify_sources ?limits document sources in
  let* captures = Map.Prepare.capture_set ?limits document verified in
  let archive =
    Pt2_archive.of_parts ~program ~weights ~constants
      ~load:(Map.Prepare.load captures)
  in
  Err.return { archive; bundle = b; captures; document }

let open_ ?limits c cohort entry =
  let* b = Bundle.ensure c cohort entry in
  of_bundle ?limits c b
