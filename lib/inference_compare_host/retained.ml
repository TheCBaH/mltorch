(* The index of payloads kept in a run directory, and the reasons others were
   not kept. A payload is content-addressed under payloads/<algorithm>/<hex>;
   the index says which role it played, so replay can find a unit's common
   inputs and a reader can see exactly what is missing and why. *)

open Inference_compare_schema
module W = Wire
module Instance = W.Text (Id.Value_instance)
module Unit_id = W.Text (Id.Unit)

type role = Candidate | Common_input | Raw_output

type entry = {
  instance : Id.Value_instance.t;
  unit : Id.Unit.t;
  role : role;
  subject : string;  (** The canonical value or the tensor local id. *)
  side : Side.t option;
  spec : Tensor_spec.t;
  payload_digest : Digest.Payload.t;
}

type reason = Budget_exhausted | Write_failed

type omission = {
  unit : Id.Unit.t;
  role : role;
  subject : string;
  reason : reason;
}

let decode_role, encode_role =
  W.enum ~kind:"retained_role"
    [
      (Candidate, "candidate");
      (Common_input, "common_input");
      (Raw_output, "raw_output");
    ]

let decode_reason, encode_reason =
  W.enum ~kind:"omission_reason"
    [ (Budget_exhausted, "budget_exhausted"); (Write_failed, "write_failed") ]

let decode_entry : entry W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* instance = W.Obj.required o "instance" Instance.decode in
      let* unit = W.Obj.required o "unit" Unit_id.decode in
      let* role = W.Obj.required o "role" decode_role in
      let* subject = W.Obj.required o "subject" W.string in
      let* side = W.Obj.optional o "side" Side.decode in
      let* spec = W.Obj.required o "spec" Tensor_spec.decode in
      let* payload_digest =
        W.Obj.required o "payload_digest" Wire_digest.payload
      in
      Ok { instance; unit; role; subject; side; spec; payload_digest })

let encode_entry (e : entry) =
  W.obj
    [
      W.field "instance" Instance.encode e.instance;
      W.field "unit" Unit_id.encode e.unit;
      W.field "role" encode_role e.role;
      W.field "subject" W.enc_string e.subject;
      W.field "side" (W.enc_option Side.encode) e.side;
      W.field "spec" Tensor_spec.encode e.spec;
      W.field "payload_digest" Wire_digest.encode_payload e.payload_digest;
    ]

let decode_omission : omission W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* unit = W.Obj.required o "unit" Unit_id.decode in
      let* role = W.Obj.required o "role" decode_role in
      let* subject = W.Obj.required o "subject" W.string in
      let* reason = W.Obj.required o "reason" decode_reason in
      Ok { unit; role; subject; reason })

let encode_omission (m : omission) =
  W.obj
    [
      W.field "unit" Unit_id.encode m.unit;
      W.field "role" encode_role m.role;
      W.field "subject" W.enc_string m.subject;
      W.field "reason" encode_reason m.reason;
    ]

type index = { entries : entry list; omissions : omission list }

let decode_index : index W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* entries = W.Obj.required o "entries" (W.list decode_entry) in
      let* omissions = W.Obj.required o "omissions" (W.list decode_omission) in
      Ok { entries; omissions })

let encode_index (i : index) =
  W.obj
    [
      W.field "entries" (W.enc_list encode_entry) i.entries;
      W.field "omissions" (W.enc_list encode_omission) i.omissions;
    ]
