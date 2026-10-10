open Err.Syntax
open Transformers_metadata.Json_util
module M = Transformers_metadata
module U = Pt2_fixture_unix
module F = Pt2_fixture

let unwrap result = Err.or_raise ~pp_error:Fault.pp result

let main name work =
  let result =
    try work () with
    | Sys_error e -> invalid e
    | Unix.Unix_error (e, op, path) ->
        invalid (op ^ " " ^ path ^ ": " ^ Unix.error_message e)
    | Err.Exn.E e -> invalid (Fmt.str "%a" Err.Exn.pp_kind e)
  in
  match result with
  | Ok () -> ()
  | Error e ->
      Fmt.epr "%s: %a@." name Fault.pp (Err.Error.kind e);
      exit 2

let entry cohort id =
  Err.of_option
    (`Metadata_missing ("cohort artifact " ^ id))
    (List.find_opt
       (fun (e : F.Cohort.entry) -> e.artifact_id = id)
       cohort.F.Cohort.entries)

let config cohort_path cache_path =
  let* bytes = U.Fetch.read cohort_path in
  let* cohort = F.Cohort.of_string bytes in
  let+ cache = U.Cache.create cache_path in
  (cohort, U.Bundle.config cache)

let fixture config cohort id =
  let* entry = entry cohort id in
  U.Fixture.open_ config cohort entry

let asset config pin_json =
  let* pin = pin_of pin_json in
  let* file =
    U.Fetch.ensure ~layer:(F.Fault.Source pin.name) config.U.Bundle.cache pin
  in
  U.Fetch.read file

let bytes_pin pin_json bytes =
  let* pin = pin_of pin_json in
  F.Check.size (F.Fault.Source pin.name)
    (Int64.of_int (String.length bytes))
    pin.size
  >>= fun () ->
  F.Check.digest (F.Fault.Source pin.name) (Pt2_sha256.string bytes) pin.sha256

let load adapter cohort_path cache_path assets_path =
  let* cohort, config = config cohort_path cache_path in
  let* assets = read assets_path in
  let* artifact_id = field "artifact_id" assets in
  let* producer =
    M.Producer.load ~consumer_root:"." ~root:"modules/devcontainer.transformers"
  in
  let request =
    M.Assets.Request.{ adapter; artifact_id; output = "assets.json" }
  in
  let* cohort_json = read cohort_path in
  let* derived = M.Assets.generate config producer cohort_json [ request ] in
  let* expected =
    Err.of_option (`Metadata_missing "derived adapter")
      (List.assoc_opt "assets.json" derived)
  in
  let* () =
    equal ~identity:assets_path ~field:"verified asset recipe" assets expected
  in
  let* fixture = fixture config cohort artifact_id in
  let+ contract =
    U.Bundle.read_member fixture.bundle "contract.json" >>= F.Contract.of_string
  in
  (assets, config, fixture, contract)
