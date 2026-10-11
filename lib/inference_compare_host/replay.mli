(** Re-executing a recorded run.

    A replay first proves it is looking at the same experiment: the run must be
    complete, every recorded file must match its pinned digest, the inputs kept
    under [inputs/] must hash to the manifest's pins, the sealed plan recomputed
    from them must equal the stored one, and the engines must have the recorded
    identities. Only then does it execute, and its result is the comparison of
    the fresh events against the recorded ones -- never a mere comparison of two
    summaries. *)

open Inference_compare_schema

type error =
  | Evidence of Evidence_read.error
  | Load of Session.error
  | Identity of string  (** Incompatible pinned identities. *)
  | Missing_payload of { unit : Id.Unit.t; keys : string list; omitted : bool }
      (** The retained common inputs of a unit are absent. [omitted] says the
          run recorded an omission (budget or write failure) rather than never
          having retained the unit. *)
  | Unknown_unit of Id.Unit.t

val pp_error : Format.formatter -> error -> unit

type report = {
  scope : [ `Unit of Id.Unit.t | `Full ];
  reexecuted : bool;
  identical : bool;
  differences : string list;
  notes : string list;
}

val unit : run_dir:string -> Id.Unit.t -> (report, error) result
(** Re-runs one unit from its retained common inputs. *)

val full : run_dir:string -> (report, error) result
(** Re-runs the whole experiment from its pinned initial payloads and recorded
    donors, flagging any difference as nondeterminism. *)

val differences : recorded:Event.t list -> replayed:Event.t list -> string list
(** What differs between two event streams once instance ids and sequence
    numbers are set aside: input digests, computed-output digests, comparison
    results (including their policy digests) and committed donors, event by
    event. Empty when the streams agree. *)
