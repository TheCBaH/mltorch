open Err.Syntax
open Transformers_metadata.Json_util
open Spec
module L = F.Logical
module R = F.Report

let declared_comparison ~name json comparisons =
  let* declared = member name json in
  let* status = field "status" declared in
  let passed = List.for_all F.Compare.passed comparisons in
  let* () =
    require
      (status = if passed then "pass" else "mismatch")
      "producer diagnostic status"
  in
  let* rows = member "outputs" declared >>= array in
  let* () =
    require
      (List.length rows = List.length comparisons)
      "producer comparison coverage"
  in
  Err.List.iter
    (fun (row, (out : F.Compare.t)) ->
      let* name = field "name" row in
      let* elements = member "elements" row >>= integer in
      let* mismatches = member "over_tolerance" row >>= integer in
      let* bitwise = member "bitwise" row >>= bool in
      let* max_error = member "max_absolute_error" row in
      let* () =
        match max_error with
        | Jsont.Number (n, _) when Float.is_finite n && n >= 0. ->
            require
              (n = Int32.float_of_bits (Int32.bits_of_float out.max_abs_error))
              "producer binary32 maximum absolute error"
        | _ -> invalid "producer maximum absolute error"
      in
      (* Bitwise identity is checked separately from tolerance, including zeros. *)
      require
        (name = out.name && elements = out.elements
        && mismatches = out.mismatches
        && ((not bitwise) || F.Compare.passed out))
        "producer comparison counts")
    (List.combine rows comparisons)

let compare ~atol ~rtol expected actual =
  let* () =
    require
      (List.map fst expected = List.map fst actual)
      "diagnostic output set/order"
  in
  Ok
    (List.map
       (fun ((name, e), (_, a)) ->
         F.Compare.tensor ~atol ~rtol ~name ~expected:e ~actual:a)
       (List.combine expected actual))

let equal_values (expected : L.t) (actual : L.t) =
  let* () =
    require
      (expected.dtype = actual.dtype && expected.shape = actual.shape)
      "producer equality dtype/shape"
  in
  let equal = ref true in
  let* () =
    Err.List.iter
      (fun tensor ->
        if tensor.L.dtype = Pt2_checkpoint_map.Dtype.F32 then (
          let finite = ref true in
          for i = 0 to Int64.to_int (L.numel tensor) - 1 do
            if not (Float.is_finite (L.get_float tensor i)) then finite := false
          done;
          require !finite "nonfinite diagnostic output")
        else Ok ())
      [ expected; actual ]
  in
  for i = 0 to Int64.to_int (L.numel expected) - 1 do
    let same =
      if expected.dtype = Pt2_checkpoint_map.Dtype.F32 then
        L.get_float expected i = L.get_float actual i
      else L.get_int64 expected i = L.get_int64 actual i
    in
    if not same then equal := false
  done;
  Ok !equal

let floats manifest =
  let checked value =
    match value with
    | Jsont.Number (n, _) when Float.is_finite n && n >= 0. -> Ok n
    | _ -> invalid "task tolerance"
  in
  let* atol = path [ "tolerances"; "atol" ] manifest >>= checked in
  let+ rtol = path [ "tolerances"; "rtol" ] manifest >>= checked in
  (atol, rtol)

let case bundle (row : Jsont.json) =
  let* id = field "id" row in
  let* artifact = field "artifact_id" row in
  let* reference =
    Err.of_option (`Metadata_missing artifact)
      (List.find_opt
         (fun r ->
           Err.payload (field "artifact_id" r.Reference.Reference.descriptor)
           = Ok artifact)
         bundle.Reference.Bundle.references)
  in
  let* files = Tensors.case_files bundle row in
  let roles = [ "eager"; "exported"; "inputs"; "published" ] in
  let* () =
    require
      (List.sort String.compare (List.map fst files)
      = List.sort String.compare
          (List.map (fun role -> "cases/" ^ id ^ "/" ^ role ^ ".pt") roles))
      "diagnostic tensor file coverage"
  in
  let role role =
    let name = "cases/" ^ id ^ "/" ^ role ^ ".pt" in
    let* descriptors =
      Err.of_option (`Metadata_missing name) (List.assoc_opt name files)
    in
    let+ values = Tensors.load (Filename.concat bundle.dir name) descriptors in
    (descriptors, values)
  in
  let* input_specs, inputs = role "inputs" in
  let* output_specs, published = role "published" in
  let* _, eager = role "eager" in
  let* _, exported = role "exported" in
  let* raw = Reference.read_member bundle ("raw/" ^ id ^ ".json") >>= parse in
  let* original_id = field "case_id" raw in
  let* raw_artifact = field "artifact_id" raw in
  let* () = same ~identity:id ~field:"raw artifact" raw_artifact artifact in
  let* original_cases_bytes =
    U.Bundle.read_member reference.bundle "cases.json"
  in
  let* original_cases = F.Cases.of_string original_cases_bytes in
  let* () =
    require
      (List.exists
         (fun c -> c.F.Cases.Case.id = original_id)
         original_cases.cases)
      "diagnostic original case"
  in
  let* () =
    Err.List.iter
      (fun (role, specs, actual) ->
        let name = "cases/" ^ original_id ^ "/" ^ role ^ ".pt" in
        let* expected = Tensors.original reference.bundle name specs in
        let* pin = path [ "published_members"; role ^ ".pt" ] raw in
        let* expected_pin =
          Err.of_option (`Member_missing name)
            (Smap.find_opt name reference.bundle.manifest.members)
        in
        let* sha = field "sha256" pin in
        let* size = member "size" pin >>= integer in
        let* () =
          require
            (sha = Pt2_sha256.Digest.to_hex expected_pin.sha256
            && size = expected_pin.size)
            "raw published-member pin"
        in
        Tensors.same_tensor_sets ~identity:name expected actual)
      [ ("inputs", input_specs, inputs); ("outputs", output_specs, published) ]
  in
  let* atol, rtol = floats bundle.manifest in
  let* expected_tol = member "tolerances" reference.contract in
  let* actual_tol = member "tolerances" bundle.manifest in
  let* () =
    equal ~identity:id ~field:"original tolerances" actual_tol expected_tol
  in
  let pairs =
    [
      ("eager_vs_published", published, eager);
      ("reexport_vs_eager", eager, exported);
      ("reexport_vs_published", published, exported);
    ]
  in
  let* comparison_claims = member "comparisons" row in
  let* rows =
    Err.List.map
      (fun (name, expected, actual) ->
        let* outputs = compare ~atol ~rtol expected actual in
        let* () = declared_comparison ~name comparison_claims outputs in
        let consumer_case =
          R.
            {
              error = None;
              id;
              inputs_digest_ok = true;
              outputs;
              outputs_digest_ok = true;
            }
        in
        let report =
          R.
            {
              artifact_id = artifact;
              atol;
              backend = "producer-diagnostic:" ^ name;
              cases = [ consumer_case ];
              consumer = "OCaml fixture comparison; no model execution";
              normalizations = [];
              pins = [];
              refusal = None;
              rtol;
              scope =
                Some
                  "Diagnostic comparison does not change consumer component \
                   acceptance";
              status = R.status_of_cases [ consumer_case ];
            }
        in
        let* payload = parse (R.to_string report) in
        let* bitwise =
          Err.List.map
            (fun ((tensor, e), (_, a)) ->
              let same =
                e.L.dtype = a.L.dtype && e.shape = a.shape
                && Pt2_sha256.Digest.equal
                     (Pt2_sha256.bigstring e.data)
                     (Pt2_sha256.bigstring a.data)
              in
              let* declared_rows =
                path [ name; "outputs" ] comparison_claims >>= array
              in
              let* declared = row_by "name" tensor declared_rows in
              let* claim = member "bitwise" declared >>= bool in
              (* The producer labels torch.equal as bitwise; it considers signed
             zeros equal. Validate that claim and report raw byte equality too. *)
              let* producer_equal = equal_values e a in
              let+ () =
                require (claim = producer_equal)
                  "producer exact-value comparison claim"
              in
              (tensor, J.bool same))
            (List.combine expected actual)
        in
        Ok
          (obj
             [
               ("comparison", J.string name);
               ("report", payload);
               ("bitwise", obj bitwise);
             ]))
      pairs
  in
  Ok (original_id, J.list rows)

let run bundle =
  let* kind = field "kind" bundle.Reference.Bundle.manifest in
  let* () = require (kind = "diagnostic") "recipe is not a diagnostic" in
  let* loading = member "model_loading" bundle.manifest in
  let* () =
    Err.List.iter
      (fun key ->
        let* rows = member key loading >>= array in
        require (rows = []) ("diagnostic model loading " ^ key))
      [ "error_msgs"; "mismatched_keys"; "missing_keys"; "unexpected_keys" ]
  in
  let* cases = member "cases" bundle.manifest >>= array in
  let* results = Err.List.map (case bundle) cases in
  let* () =
    Err.List.iter
      (fun reference ->
        let* bytes =
          U.Bundle.read_member reference.Reference.Reference.bundle "cases.json"
        in
        let* expected = F.Cases.of_string bytes in
        let id = reference.bundle.entry.artifact_id in
        let actual =
          List.filter_map
            (fun (row, (case_id, _)) ->
              if Err.payload (field "artifact_id" row) = Ok id then Some case_id
              else None)
            (List.combine cases results)
        in
        require
          (actual = List.map (fun c -> c.F.Cases.Case.id) expected.cases)
          "complete original diagnostic case coverage")
      bundle.references
  in
  let* id = member "fixture_id" bundle.manifest in
  Ok
    (obj
       [
         ("schema_version", J.int 1);
         ("fixture_id", id);
         ("status", J.string "diagnostic_compared");
         ("consumer_acceptance", J.string "unchanged; component replay decides");
         ("manifest", bundle.manifest);
         ("cases", J.list (List.map snd results));
       ])
