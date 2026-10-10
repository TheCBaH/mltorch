open Err.Syntax
open Transformers_metadata.Json_util
module F = Pt2_fixture

module Entry = struct
  type t = {
    artifact_id : string;
    gate : string;
    policy : Run.Policy.t;
    reason : string;
  }
end

let of_json ~cohort_bytes json =
  let* () = schema 1 json in
  let* sha = field "cohort_sha256" json in
  let* () =
    same ~identity:"numerical policies" ~field:"cohort bytes" sha
      (Transformers_metadata.Source.hash cohort_bytes)
  in
  let* cohort = F.Cohort.of_string cohort_bytes in
  let* rows = member "requirements" json >>= array in
  let* rows =
    Err.List.map
      (fun row ->
        let* artifact_id = field "artifact_id" row in
        let* gate = field "gate" row in
        let* () =
          if List.mem gate [ "core"; "deferred" ] then Ok ()
          else invalid "numerical gate class"
        in
        let* reason = field "reason" row in
        let* () =
          if reason <> "" then Ok () else invalid "numerical policy reason"
        in
        let+ policy = member "policy" row >>= Run.Policy.of_json in
        Entry.{ artifact_id; gate; policy; reason })
      rows
  in
  let ids = List.map (fun row -> row.Entry.artifact_id) rows in
  let* () = unique ids in
  let expected =
    List.map (fun row -> row.F.Cohort.artifact_id) cohort.entries
  in
  let+ () =
    if List.sort String.compare ids = List.sort String.compare expected then
      Ok ()
    else invalid "numerical requirements must cover the complete cohort"
  in
  rows

let core entries = List.filter (fun entry -> entry.Entry.gate = "core") entries
