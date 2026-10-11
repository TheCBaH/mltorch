module W = Wire
module Case = W.Text (Id.Case)

type qualification = Exact | Within_declared_contract

type t = {
  case_id : Id.Case.t;
  ordered_output_payload_digest : Digest.Payload.t list;
  output_specs_digest : Digest.Output_specs.t;
  tested_graph : Graph_key.t;
  tested_engine : Engine_identity.t;
  reference_engine : Engine_identity.t;
  first_pass_run_digest : Digest.Run_evidence.t;
  second_pass_run_digest : Digest.Run_evidence.t;
  full_graph_confirmation_digest : Digest.Run_evidence.t;
  contract_digest : Digest.Contract.t;
  qualification : qualification;
}

let decode_qualification, encode_qualification =
  W.enum ~kind:"qualification"
    [ (Exact, "exact"); (Within_declared_contract, "within_declared_contract") ]

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* case_id = W.Obj.required o "case_id" Case.decode in
      let* ordered_output_payload_digest =
        W.Obj.required o "ordered_output_payload_digest"
          (W.list Wire_digest.payload)
      in
      let* output_specs_digest =
        W.Obj.required o "output_specs_digest" Wire_digest.output_specs
      in
      let* tested_graph = W.Obj.required o "tested_graph" Graph_key.decode in
      let* tested_engine =
        W.Obj.required o "tested_engine" Engine_identity.decode
      in
      let* reference_engine =
        W.Obj.required o "reference_engine" Engine_identity.decode
      in
      let* first_pass_run_digest =
        W.Obj.required o "first_pass_run_digest" Wire_digest.run_evidence
      in
      let* second_pass_run_digest =
        W.Obj.required o "second_pass_run_digest" Wire_digest.run_evidence
      in
      let* full_graph_confirmation_digest =
        W.Obj.required o "full_graph_confirmation_digest"
          Wire_digest.run_evidence
      in
      let* contract_digest =
        W.Obj.required o "contract_digest" Wire_digest.contract
      in
      let* qualification =
        W.Obj.required o "qualification" decode_qualification
      in
      Ok
        {
          case_id;
          ordered_output_payload_digest;
          output_specs_digest;
          tested_graph;
          tested_engine;
          reference_engine;
          first_pass_run_digest;
          second_pass_run_digest;
          full_graph_confirmation_digest;
          contract_digest;
          qualification;
        })

let encode g =
  W.obj
    [
      W.field "case_id" Case.encode g.case_id;
      W.field "ordered_output_payload_digest"
        (W.enc_list Wire_digest.encode_payload)
        g.ordered_output_payload_digest;
      W.field "output_specs_digest" Wire_digest.encode_output_specs
        g.output_specs_digest;
      W.field "tested_graph" Graph_key.encode g.tested_graph;
      W.field "tested_engine" Engine_identity.encode g.tested_engine;
      W.field "reference_engine" Engine_identity.encode g.reference_engine;
      W.field "first_pass_run_digest" Wire_digest.encode_run_evidence
        g.first_pass_run_digest;
      W.field "second_pass_run_digest" Wire_digest.encode_run_evidence
        g.second_pass_run_digest;
      W.field "full_graph_confirmation_digest" Wire_digest.encode_run_evidence
        g.full_graph_confirmation_digest;
      W.field "contract_digest" Wire_digest.encode_contract g.contract_digest;
      W.field "qualification" encode_qualification g.qualification;
    ]
