module W = Wire
module Run_id = W.Text (Id.Run)
module Case = W.Text (Id.Case)
module Experiment_id = W.Text (Id.Experiment)
module Unit_id = W.Text (Id.Unit)

type payload =
  | Failed of Failure.t
  | Frontier_committed of Committed_value.t list
  | Graph_completed of { side : Side.t; outputs : Computed_value.t list }
  | Inputs_verified of Verified_input.t list
  | Outputs_compared of Comparison_result.t list

type t = {
  sequence : int64;
  run_id : Id.Run.t;
  case_id : Id.Case.t;
  experiment_id : Id.Experiment.t;
  unit : Id.Unit.t option;
  payload : payload;
}

let decode_payload : payload W.decoder =
  W.variant (fun tag o ->
      let ( let* ) = W.( let* ) in
      match tag with
      | "failed" ->
          let* f = W.Obj.required o "failure" Failure.decode in
          Ok (Failed f)
      | "frontier_committed" ->
          let* vs = W.Obj.required o "values" (W.list Committed_value.decode) in
          Ok (Frontier_committed vs)
      | "graph_completed" ->
          let* side = W.Obj.required o "side" Side.decode in
          let* outputs =
            W.Obj.required o "outputs" (W.list Computed_value.decode)
          in
          Ok (Graph_completed { side; outputs })
      | "inputs_verified" ->
          let* vs = W.Obj.required o "inputs" (W.list Verified_input.decode) in
          Ok (Inputs_verified vs)
      | "outputs_compared" ->
          let* rs =
            W.Obj.required o "results" (W.list Comparison_result.decode)
          in
          Ok (Outputs_compared rs)
      | tag -> W.unknown_tag ~kind:"event_payload" tag)

let encode_payload = function
  | Failed f -> W.tagged "failed" [ W.field "failure" Failure.encode f ]
  | Frontier_committed vs ->
      W.tagged "frontier_committed"
        [ W.field "values" (W.enc_list Committed_value.encode) vs ]
  | Graph_completed { side; outputs } ->
      W.tagged "graph_completed"
        [
          W.field "side" Side.encode side;
          W.field "outputs" (W.enc_list Computed_value.encode) outputs;
        ]
  | Inputs_verified vs ->
      W.tagged "inputs_verified"
        [ W.field "inputs" (W.enc_list Verified_input.encode) vs ]
  | Outputs_compared rs ->
      W.tagged "outputs_compared"
        [ W.field "results" (W.enc_list Comparison_result.encode) rs ]

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* sequence = W.Obj.required o "sequence" W.nonneg_int64 in
      let* run_id = W.Obj.required o "run_id" Run_id.decode in
      let* case_id = W.Obj.required o "case_id" Case.decode in
      let* experiment_id =
        W.Obj.required o "experiment_id" Experiment_id.decode
      in
      let* unit = W.Obj.optional o "unit" Unit_id.decode in
      let* payload = W.Obj.required o "payload" decode_payload in
      Ok { sequence; run_id; case_id; experiment_id; unit; payload })

let encode e =
  W.obj
    [
      W.field "sequence" W.enc_int64 e.sequence;
      W.field "run_id" Run_id.encode e.run_id;
      W.field "case_id" Case.encode e.case_id;
      W.field "experiment_id" Experiment_id.encode e.experiment_id;
      W.field "unit" (W.enc_option Unit_id.encode) e.unit;
      W.field "payload" encode_payload e.payload;
    ]
