open Transformers_metadata.Json_util
module R = Pt2_fixture.Report

let make ?(backend = "native-direct") ?(normalizations = []) ?(pins = [])
    artifact id ~atol ~rtol outputs =
  let case =
    R.
      {
        error = None;
        id;
        inputs_digest_ok = true;
        outputs_digest_ok = true;
        outputs;
      }
  in
  let report =
    R.
      {
        artifact_id = artifact;
        atol;
        rtol;
        cases = [ case ];
        consumer = "see immutable task run";
        backend;
        normalizations;
        pins;
        refusal = None;
        scope =
          Some
            "bounded producer task fixture; unimplemented boundaries remain \
             explicit";
        status = R.status_of_cases [ case ];
      }
  in
  parse (R.to_string report)
