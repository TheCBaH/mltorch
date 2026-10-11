module W = Wire

type evidence = Independent_baseline | Shared_frontier | Two_pass_validation
type execution = Complete | Failed | Interrupted | Refused
type numerical = Agreement | Disagreement | Not_established

type t = {
  execution : execution;
  coverage : Coverage.t;
  numerical : numerical;
  evidence : evidence;
}

let decode_evidence, encode_evidence =
  W.enum ~kind:"evidence_kind"
    [
      (Independent_baseline, "independent_baseline");
      (Shared_frontier, "shared_frontier");
      (Two_pass_validation, "two_pass_validation");
    ]

let decode_execution, encode_execution =
  W.enum ~kind:"execution"
    [
      (Complete, "complete");
      (Failed, "failed");
      (Interrupted, "interrupted");
      (Refused, "refused");
    ]

let decode_numerical, encode_numerical =
  W.enum ~kind:"numerical"
    [
      (Agreement, "agreement");
      (Disagreement, "disagreement");
      (Not_established, "not_established");
    ]

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* execution = W.Obj.required o "execution" decode_execution in
      let* coverage = W.Obj.required o "coverage" Coverage.decode in
      let* numerical = W.Obj.required o "numerical" decode_numerical in
      let* evidence = W.Obj.required o "evidence" decode_evidence in
      Ok { execution; coverage; numerical; evidence })

let encode c =
  W.obj
    [
      W.field "execution" encode_execution c.execution;
      W.field "coverage" Coverage.encode c.coverage;
      W.field "numerical" encode_numerical c.numerical;
      W.field "evidence" encode_evidence c.evidence;
    ]
