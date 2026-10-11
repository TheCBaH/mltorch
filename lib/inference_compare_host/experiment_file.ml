open Inference_compare_schema
module W = Wire

type case_payload = {
  key : Id.Canonical_value.t;
  payload_id : Id.Payload.t;
  file : string;
}

type endpoint = { graph : string; engine : Engine_spec.t }

type t = {
  experiment_id : Id.Experiment.t;
  case_id : Id.Case.t;
  digest_algorithm : Id.Digest_algorithm.t;
  left : endpoint;
  right : endpoint;
  mapping : string;
  selection : string;
  case : case_payload list;
  default_donor : Side.t;
  overrides : Donor_override.t list;
  limits : Resource_limits.t;
  retain_units : Id.Unit.t list;
}

module Exp_id = W.Text (Id.Experiment)
module Case_id = W.Text (Id.Case)
module Algorithm = W.Text (Id.Digest_algorithm)
module Key = W.Text (Id.Canonical_value)
module Payload = W.Text (Id.Payload)
module Unit_id = W.Text (Id.Unit)

let decode_case : case_payload W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* key = W.Obj.required o "key" Key.decode in
      let* payload_id = W.Obj.required o "payload_id" Payload.decode in
      let* file = W.Obj.required o "file" W.string in
      Ok { key; payload_id; file })

let decode_endpoint : endpoint W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* graph = W.Obj.required o "graph" W.string in
      let* engine = W.Obj.required o "engine" Engine_spec.decode in
      Ok { graph; engine })

let encode_endpoint (e : endpoint) =
  W.obj
    [
      W.field "graph" W.enc_string e.graph;
      W.field "engine" Engine_spec.encode e.engine;
    ]

let encode_case (c : case_payload) =
  W.obj
    [
      W.field "key" Key.encode c.key;
      W.field "payload_id" Payload.encode c.payload_id;
      W.field "file" W.enc_string c.file;
    ]

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* () = W.check_version o ~expected:"1" in
      let* experiment_id = W.Obj.required o "experiment_id" Exp_id.decode in
      let* case_id = W.Obj.required o "case_id" Case_id.decode in
      let* digest_algorithm =
        W.Obj.required o "digest_algorithm" Algorithm.decode
      in
      let* left = W.Obj.required o "left" decode_endpoint in
      let* right = W.Obj.required o "right" decode_endpoint in
      let* mapping = W.Obj.required o "mapping" W.string in
      let* selection = W.Obj.required o "selection" W.string in
      let* case = W.Obj.required o "case" (W.list decode_case) in
      let* default_donor = W.Obj.required o "default_donor" Side.decode in
      let* overrides =
        W.Obj.required o "overrides" (W.list Donor_override.decode)
      in
      let* limits = W.Obj.required o "limits" Resource_limits.decode in
      let* retain_units =
        W.Obj.required o "retain_units" (W.list Unit_id.decode)
      in
      Ok
        {
          experiment_id;
          case_id;
          digest_algorithm;
          left;
          right;
          mapping;
          selection;
          case;
          default_donor;
          overrides;
          limits;
          retain_units;
        })

let encode (t : t) =
  W.obj
    [
      W.field "schema_version" W.enc_string "1";
      W.field "experiment_id" Exp_id.encode t.experiment_id;
      W.field "case_id" Case_id.encode t.case_id;
      W.field "digest_algorithm" Algorithm.encode t.digest_algorithm;
      W.field "left" encode_endpoint t.left;
      W.field "right" encode_endpoint t.right;
      W.field "mapping" W.enc_string t.mapping;
      W.field "selection" W.enc_string t.selection;
      W.field "case" (W.enc_list encode_case) t.case;
      W.field "default_donor" Side.encode t.default_donor;
      W.field "overrides" (W.enc_list Donor_override.encode) t.overrides;
      W.field "limits" Resource_limits.encode t.limits;
      W.field "retain_units" (W.enc_list Unit_id.encode) t.retain_units;
    ]
