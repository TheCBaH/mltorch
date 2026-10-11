(** The immutable record of one run.

    A run directory [<root>/run-<run id>] is created exclusively and holds
    [manifest.json] (every identity the run depends on), [plan.json],
    [policy_bindings.json], [events.jsonl] (typed events, monotone sequence),
    [inputs/] (the experiment and every file it names, byte for byte, so the
    directory replays on its own), [retained.json] with
    [payloads/<algorithm>/<digest>] (what was kept, and why anything was not),
    [summary.json] and, written last and atomically, [completion.json] binding
    the digest of every other file. A directory without [completion.json] is
    incomplete, whatever else it contains, and is never read as a result. *)

open Inference_compare_schema

type error =
  | Duplicate_run of string
  | Io of File_io.error
  | Digest of string
  | Unportable_path of string

val pp_error : Format.formatter -> error -> unit

type t

val create :
  root:string ->
  hashers:Inference_compare_codecs.Payload_hasher.Registry.t ->
  Session.prepared ->
  run_id:Id.Run.t ->
  (t, error) result
(** Creates the directory, writes the manifest, plan and policy bindings, and
    opens the event stream. Refuses an existing directory. *)

val directory : t -> string

val emit : t -> Event.t -> (unit, string) result
(** Appends one line and flushes. A failure is reported, not hidden: the
    coordinator then publishes nothing for the unit. *)

val retain :
  t -> Inference_compare_core.Coordinator.snapshot -> (unit, string) result
(** Applies the experiment's retention mode and budget to one unit's payloads.
    Under [Required] retention a payload that does not fit is an error;
    otherwise it is recorded as an omission. *)

val finalize :
  t -> Inference_compare_core.Coordinator.result -> (unit, error) result
(** Writes the retention index, the summary and finally the completion record.
    Calling it twice, or after a failed write, leaves the directory incomplete
    rather than corrupt. *)

val abandon : t -> unit
(** Closes the event stream without finalizing (the interrupted-run path). *)

(** A test seam: make the next event write fail, as a full disk would. *)
module Fault : sig
  val fail_events_after : t -> int -> unit
end
