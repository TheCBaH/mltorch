module CE = Coral_estimate.Coral_census
module CM = Coral_estimate.Coral_model

let%test_unit
    "the estimator prices the recorded engine rather than selecting again" =
  let case =
    List.find
      (fun (c : Coral_estimate_cases.t) -> c.name = "linear 9x32 k64")
      Coral_estimate_cases.all
  in
  let census =
    match Fit_test.censuses case with [ c ] -> c | _ -> assert false
  in
  assert (
    List.for_all
      (fun ((c : CE.Contract_call.t), _) ->
        c.engine = Some Contraction.Select.Engine.Zvt)
      census.contracts);
  assert (census.contracts <> []);
  let vector_core =
    { CM.Core.matrix = false; vector = true; packed = (fun _ -> false) }
  in
  let fs = CM.features vector_core census in
  assert (
    List.exists (fun (name, _) -> String.starts_with ~prefix:"zvt.f32." name) fs);
  assert (not (List.mem_assoc "rvv.macs" fs))
