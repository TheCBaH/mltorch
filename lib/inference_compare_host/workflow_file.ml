(* The file a certification names: the experiment it certifies, the
   computation contract, the validation workflow itself and, when the contract
   requires an end-to-end acceptance, the frozen reference outputs. Paths
   resolve relative to the file. *)

open Inference_compare_schema
module W = Wire
module Occurrence = W.Text (Id.Output_occurrence)

type frozen = { occurrence : Id.Output_occurrence.t; file : string }

type t = {
  experiment : string;
  contract : string;
  workflow : Validation_workflow.t;
  frozen_outputs : frozen list;
}

let decode_frozen : frozen W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* occurrence = W.Obj.required o "occurrence" Occurrence.decode in
      let* file = W.Obj.required o "file" W.string in
      Ok { occurrence; file })

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* () = W.check_version o ~expected:"1" in
      let* experiment = W.Obj.required o "experiment" W.string in
      let* contract = W.Obj.required o "contract" W.string in
      let* workflow = W.Obj.required o "workflow" Validation_workflow.decode in
      let* frozen_outputs =
        W.Obj.required o "frozen_outputs" (W.list decode_frozen)
      in
      Ok { experiment; contract; workflow; frozen_outputs })

let encode (t : t) =
  W.obj
    [
      W.field "schema_version" W.enc_string "1";
      W.field "experiment" W.enc_string t.experiment;
      W.field "contract" W.enc_string t.contract;
      W.field "workflow" Validation_workflow.encode t.workflow;
      W.field "frozen_outputs"
        (W.enc_list (fun (f : frozen) ->
             W.obj
               [
                 W.field "occurrence" Occurrence.encode f.occurrence;
                 W.field "file" W.enc_string f.file;
               ]))
        t.frozen_outputs;
    ]
