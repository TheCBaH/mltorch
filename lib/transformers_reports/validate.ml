open Err.Syntax
open Transformers_metadata.Json_util
module J = Jsont.Json

let require yes why = if yes then Ok () else invalid why

let nullable_string = function
  | Jsont.Null _ -> Ok None
  | json ->
      let+ s = string json in
      Some s

let names json = array json >>= Err.List.map string

let count json =
  let* s = string json in
  match Int64.of_string_opt s with
  | Some n when n >= 0L && Int64.to_string n = s -> Ok n
  | _ -> invalid "nonnegative decimal int64 count"

let number json =
  match json with
  | Jsont.Number (n, _) when Float.is_finite n && n >= 0. -> Ok ()
  | Jsont.String (("nan" | "inf" | "-inf"), _) -> Ok ()
  | _ -> invalid "comparison error number"

let numel spec =
  let* shape = member "shape" spec >>= array in
  Err.List.fold_left
    (fun n j ->
      let* d = integer j in
      let* () =
        require (d = 0L || n <= Int64.div Int64.max_int d) "shape size overflow"
      in
      Ok (Int64.mul n d))
    1L shape

let output spec report =
  let* name = field "name" report in
  let* expected_name = field "name" spec in
  let* () = same ~identity:name ~field:"output name" name expected_name in
  let* verdict = field "verdict" report in
  let* () =
    require
      (List.mem verdict
         [
           "pass";
           "values_differ";
           "dtype_differs";
           "shape_differs";
           "unsupported_dtype";
         ])
      "output verdict"
  in
  let* elements = member "elements" report >>= count in
  let* mismatches = member "mismatches" report >>= count in
  let* () = require (mismatches <= elements) "mismatches exceed elements" in
  let* expected = numel spec in
  let* () =
    if List.mem verdict [ "pass"; "values_differ" ] then
      require
        (elements = expected && mismatches = 0L = (verdict = "pass"))
        "output coverage/verdict mismatch"
    else Ok ()
  in
  let* () = member "max_abs_error" report >>= number in
  let* () = member "max_rel_error" report >>= number in
  let* first = member "first_mismatches" report >>= array in
  let* () =
    require
      (List.length first <= Pt2_fixture.Compare.max_reported
      && Int64.of_int (List.length first) <= mismatches)
      "mismatch sample count"
  in
  let* () =
    Err.List.iter
      (fun row ->
        let* index = member "index" row >>= array in
        let* shape = member "shape" spec >>= array in
        let* () =
          require
            (List.length index = List.length shape)
            "mismatch coordinate rank"
        in
        let* () =
          Err.List.iter
            (fun (i, d) ->
              let* i = count i in
              let* d = integer d in
              require (i < d) "mismatch coordinate bound")
            (List.combine index shape)
        in
        let* _ = field "actual" row in
        let+ _ = field "expected" row in
        ())
      first
  in
  let* () =
    match verdict with
    | "dtype_differs" ->
        let* _ = field "actual_dtype" report in
        let* expected_dtype = field "expected_dtype" report in
        let* dtype = field "dtype" spec in
        (* Contract spellings differ from compact report dtype codes. *)
        let* dtype =
          Err.of_option (`Metadata_invalid "contract dtype")
            (Pt2_checkpoint_map.Dtype.of_torch_name dtype)
        in
        same ~identity:name ~field:"expected dtype" expected_dtype
          (Pt2_checkpoint_map.Dtype.to_code dtype)
    | "shape_differs" ->
        let* actual = member "actual_shape" report >>= array in
        let* _ = Err.List.map count actual in
        let* expected =
          member "expected_shape" report >>= array >>= Err.List.map count
        in
        let* shape = member "shape" spec >>= array >>= Err.List.map integer in
        require (shape = expected) "expected shape mismatch"
    | "unsupported_dtype" ->
        let+ _ = field "dtype" report in
        ()
    | _ -> Ok ()
  in
  Ok (verdict = "pass")

let case specs json =
  let* id = field "id" json in
  let* error = member "error" json >>= nullable_string in
  let* input_ok = member "inputs_digest_verified" json >>= bool in
  let* output_ok = member "outputs_digest_verified" json >>= bool in
  let* outputs = member "outputs" json >>= array in
  let* declared = member "passed" json >>= bool in
  let* () =
    require
      (outputs = [] || List.length outputs = List.length specs)
      ("partial output set: " ^ id)
  in
  let* passes =
    if outputs = [] then Ok []
    else Err.List.map (fun (s, o) -> output s o) (List.combine specs outputs)
  in
  let passed =
    input_ok && output_ok && error = None && passes <> []
    && List.for_all Fun.id passes
  in
  let* () = require (passed = declared) ("case passed flag: " ^ id) in
  let details =
    (match error with None -> [] | Some s -> [ id ^ ": " ^ s ])
    @ (if input_ok then [] else [ id ^ ": input digest failed" ])
    @ (if output_ok then [] else [ id ^ ": output digest failed" ])
    @ if outputs = [] then [ id ^ ": no compared outputs" ] else []
  in
  let* failures =
    Err.List.map
      (fun out ->
        let* v = field "verdict" out in
        let* name = field "name" out in
        let* n = field "mismatches" out in
        let* total = field "elements" out in
        Ok
          (if v = "pass" then []
           else [ id ^ " " ^ name ^ ": " ^ v ^ " (" ^ n ^ "/" ^ total ^ ")" ]))
      outputs
  in
  Ok (id, passed, details @ List.concat failures)

let report ~manifest ~manifest_sha256 ~expected json =
  let* () = schema 3 json in
  let* id = field "artifact_id" json in
  let* expected_id = field "artifact_id" expected in
  let* () = same ~identity:id ~field:"artifact" id expected_id in
  let* run_id = member "run_id" manifest in
  let* actual_run = member "run_id" json in
  let* () = equal ~identity:id ~field:"run identity" actual_run run_id in
  let* sha = field "run_manifest_sha256" json in
  let* () = same ~identity:id ~field:"manifest digest" sha manifest_sha256 in
  let* execution = member "execution" json in
  let* policy = member "policy" manifest in
  let* () = equal ~identity:id ~field:"execution policy" execution policy in
  let* backend = field "backend" json in
  let* p = Run.Policy.of_json policy in
  let* () =
    same ~identity:id ~field:"backend route" backend (Run.Policy.backend p)
  in
  let* consumer = field "consumer" json in
  let* expected_consumer =
    path [ "identity"; "consumer"; "commit" ] manifest >>= string
  in
  let* () =
    same ~identity:id ~field:"consumer commit" consumer expected_consumer
  in
  let* pins = member "pins" json in
  let* expected_pins = Expectation.report_pins expected in
  let* () = equal ~identity:id ~field:"report pins" pins expected_pins in
  let* _ = member "normalizations" json >>= names in
  let* _ = member "scope" json >>= nullable_string in
  let* refusal = member "refusal" json >>= nullable_string in
  let* status = field "status" json in
  let* fields = members json in
  match List.assoc_opt "acquisition_error" fields with
  | Some error ->
      let* error = string error in
      let* cases = member "cases" json >>= array in
      let+ () =
        require
          (status = "failed" && cases = [] && refusal = None)
          "acquisition failure cannot pass"
      in
      (status, [ "acquisition: " ^ error ])
  | None ->
      let* tolerances = member "tolerances" json in
      let* expected_tolerances = member "tolerances" expected in
      let* () =
        equal ~identity:id ~field:"contract tolerances" tolerances
          expected_tolerances
      in
      let* specs = member "outputs" expected >>= array in
      let* cases =
        member "cases" json >>= array >>= Err.List.map (case specs)
      in
      let case_ids = List.map (fun (id, _, _) -> id) cases in
      let* () = unique case_ids in
      let* expected_cases = member "cases" expected >>= names in
      let* () =
        require
          (case_ids = expected_cases || (status = "refused" && cases = []))
          "requested case coverage mismatch"
      in
      let passed =
        cases <> [] && List.for_all (fun (_, pass, _) -> pass) cases
      in
      let* () =
        require
          (match status with
          | "passed" -> passed && refusal = None
          | "failed" -> (not passed) && refusal = None
          | "refused" -> refusal <> None
          | _ -> false)
          "artifact status inconsistent with cases/refusal"
      in
      Ok
        ( status,
          (match refusal with None -> [] | Some s -> [ "refusal: " ^ s ])
          @ List.concat_map (fun (_, _, details) -> details) cases )
