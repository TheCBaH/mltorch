(* Per-run mutable bookkeeping: monotone event sequence numbers, value
   instance ids, and the deferred cleanups of the unit in flight. *)

open Inference_compare_schema

type t = {
  run_id : Id.Run.t;
  case_id : Id.Case.t;
  experiment_id : Id.Experiment.t;
  emit : Event.t -> (unit, string) result;
  mutable sequence : int64;
  mutable instances : int;
  mutable events : Event.t list;  (** Newest first, as emitted successfully. *)
}

let create ~run_id ~case_id ~experiment_id ~emit =
  {
    run_id;
    case_id;
    experiment_id;
    emit;
    sequence = 0L;
    instances = 0;
    events = [];
  }

let instance t =
  t.instances <- t.instances + 1;
  match
    Id.Value_instance.of_string
      (Printf.sprintf "%s.%s.v%d"
         (Id.Run.to_string t.run_id)
         (Id.Experiment.to_string t.experiment_id)
         t.instances)
  with
  | Ok i -> i
  | Error _ -> invalid_arg "Run_state.instance"

let failure ?side ?unit ?node ?tensor ?rule ?comparison ?text code phase :
    Failure.t =
  let diagnostic =
    Option.bind text (fun s ->
        match Prose.Diagnostic_text.of_string s with
        | Ok d -> Some d
        | Error _ -> None)
  in
  {
    code;
    phase;
    side;
    unit;
    node;
    tensor;
    rule;
    comparison;
    expected = None;
    actual = diagnostic;
  }

(* Emit one event. A sink failure is itself a failure: the event is not
   recorded, and the caller must not publish what the event describes. *)
let emit t ?unit payload : (unit, Failure.t) result =
  let ev : Event.t =
    {
      sequence = t.sequence;
      run_id = t.run_id;
      case_id = t.case_id;
      experiment_id = t.experiment_id;
      unit;
      payload;
    }
  in
  match t.emit ev with
  | Ok () ->
      t.sequence <- Int64.succ t.sequence;
      t.events <- ev :: t.events;
      Ok ()
  | Error msg -> Error (failure ?unit ~text:msg Evidence_failed Commit)

(* Deferred cleanups, run newest first. *)
module Defer = struct
  type t = (unit -> unit) list ref

  let create () : t = ref []
  let add (t : t) f = t := f :: !t

  let run (t : t) =
    let fs = !t in
    t := [];
    List.iter (fun f -> f ()) fs
end
