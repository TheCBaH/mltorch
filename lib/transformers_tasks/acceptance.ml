open Err.Syntax
open Transformers_metadata.Json_util
module F = Pt2_fixture
module J = Jsont.Json

let order expected actual =
  let* () = unique (List.map fst actual) in
  let* () =
    Spec.require
      (List.length expected = List.length actual)
      "complete input coverage"
  in
  Err.List.map
    (fun (name, _) ->
      let+ value =
        Err.of_option (`Missing_tensor name) (List.assoc_opt name actual)
      in
      (name, value))
    expected

let report = Task_report.make

let reference bundle case =
  let* fields = members case in
  let* artifact =
    match List.assoc_opt "artifact_id" fields with
    | Some value -> string value
    | None -> field "artifact_id" bundle.Reference.Bundle.manifest
  in
  let+ reference =
    Err.of_option (`Metadata_missing artifact)
      (List.find_opt
         (fun r ->
           Err.payload (field "artifact_id" r.Reference.Reference.descriptor)
           = Ok artifact)
         bundle.references)
  in
  (artifact, reference)

let case config cohort producer ~models bundle row =
  let* id = field "id" row in
  let* artifact, reference = reference bundle row in
  let* expected = Adapter.role bundle row "inputs" in
  let* actual =
    Adapter.prepare config producer bundle reference row >>= order expected
  in
  let* input_checks = Diagnostic.compare ~atol:0. ~rtol:0. expected actual in
  let* input_report =
    report ~backend:"consumer-adapter" artifact id ~atol:0. ~rtol:0.
      input_checks
  in
  let* output_report, boundaries =
    if not models then Ok (J.null (), [])
    else
      let* () =
        Spec.require
          (List.for_all F.Compare.passed input_checks)
          "adapter comparison failed before model execution"
      in
      let* fixture = Demo.execution_fixture config cohort reference in
      let* contract = text reference.contract >>= F.Contract.of_string in
      let normalizations = ref [] in
      let on_empty_caches r = normalizations := Input.normalizations r in
      let* output_values =
        Input.run ~on_empty_caches fixture.archive contract actual
      in
      let* expected = Adapter.role bundle row "outputs" in
      let* outputs =
        Diagnostic.compare ~atol:contract.atol ~rtol:contract.rtol expected
          output_values
      in
      let* output_report =
        report ~normalizations:!normalizations
          ~pins:(Demo.execution_pins fixture)
          artifact id ~atol:contract.atol ~rtol:contract.rtol outputs
      in
      let+ boundaries =
        Boundary.run config cohort bundle row reference fixture contract actual
          output_values
      in
      (output_report, boundaries)
  in
  let* recipe = field "recipe_id" bundle.manifest in
  Ok
    (obj
       [
         ("id", J.string id);
         ("status", J.string "compared");
         ("adapter", input_report);
         ("model", output_report);
         ( "required_boundaries",
           J.list (List.map J.string (Boundary.required ~models recipe)) );
         ("boundaries", J.list boundaries);
         ( "coverage",
           J.string
             (if models then
                "all model inputs/outputs; supported image host/tower \
                 boundaries; tokenizer offsets/special-token masks and image \
                 intermediates deferred"
              else
                "all model inputs; model/host/towers, tokenizer \
                 offsets/special-token masks and image intermediates deferred")
         );
       ])

let run config cohort producer ~models bundle =
  let* () = Adapter.recipe producer bundle.Reference.Bundle.manifest in
  let* rows = member "cases" bundle.manifest >>= array in
  let* cases =
    Err.List.map
      (fun row ->
        let* id = field "id" row in
        match Err.payload (case config cohort producer ~models bundle row) with
        | Ok result -> Ok result
        | Error e ->
            Ok
              (obj
                 [
                   ("id", J.string id);
                   ("status", J.string "refused");
                   ("error", J.string (Fmt.str "%a" Fault.pp e));
                 ]))
      rows
  in
  let* recipe = field "recipe_id" bundle.manifest in
  let* fixture_id = member "fixture_id" bundle.manifest in
  Ok
    (obj
       [
         ("schema_version", J.int 2);
         ("recipe_id", J.string recipe);
         ("fixture_id", fixture_id);
         ("status", J.string "partial task coverage");
         ("consumer_acceptance", J.string "not promoted");
         ("cases", J.list cases);
       ])

let passed value =
  let report_passed value = Err.payload (field "status" value) = Ok "passed" in
  let boundary_inventory_required =
    Err.payload (member "schema_version" value >>= integer) = Ok 2L
  in
  let boundaries_passed value =
    let checked =
      let* fields = members value in
      match
        ( List.assoc_opt "required_boundaries" fields,
          List.assoc_opt "boundaries" fields )
      with
      | None, None ->
          (* Schema-1 bounded generation has its own host checks. *)
          Ok (not boundary_inventory_required)
      | Some required, Some boundaries ->
          let* required = Spec.names required in
          let* boundaries = array boundaries in
          let* actual = Err.List.map (field "boundary") boundaries in
          let* () = unique required in
          let* () = unique actual in
          let* reports = Err.List.map (member "report") boundaries in
          Ok (required = actual && List.for_all report_passed reports)
      | _ -> Ok false
    in
    Err.payload checked = Ok true
  in
  match Err.payload (member "cases" value >>= array) with
  | Error _ -> false
  | Ok cases ->
      cases <> []
      && List.for_all
           (fun c ->
             Err.payload (field "status" c) = Ok "compared"
             && boundaries_passed c
             &&
             match
               (Err.payload (member "adapter" c), Err.payload (member "model" c))
             with
             | Ok adapter, Ok (Jsont.Null _) -> report_passed adapter
             | Ok adapter, Ok model ->
                 report_passed adapter && report_passed model
             | _ -> false)
           cases
