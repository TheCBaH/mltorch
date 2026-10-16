(** One guest run on a Coral simulator: the image written to a fresh directory,
    the simulator spawned in its own session, the input stream fed to stdin
    while stdout and stderr drain, under a wall-clock deadline that kills the
    whole process group.

    The frame decides the outcome, not the exit status: the status carries only
    the low eight bits of the guest's exit code, and the simulators' own
    statuses (124 at the RTL cycle limit, 125 on a fault halt) can equal a guest
    code, so a result frame is success whatever the status, and its absence is
    an error even with 0. *)

module Trap : sig
  type t = { mcause : int32; mepc : int32; mtval : int32 }
end

module Run_error : sig
  type t =
    | Cycle_limit
        (** The RTL testbench reported its loop limit (and exited, 124). *)
    | Guest_trap of Trap.t
    | Input_refused of Coral_runtime.Refusal.t
    | Malformed_output of { malformed : Coral_wire.Malformed.t; tail : string }
    | Output_limit of int64
        (** More output than the frame and 4 MiB of text: the process was
            killed. *)
    | Simulator_crashed of { status : string; tail : string }
        (** No frame, and the process failed; [tail] is its last output. *)
    | Simulator_missing of string
    | Timeout

  val pp : t Fmt.t
end

type outcome = {
  argv : string list;  (** As run, the image path included. *)
  mcycle : int64;
  minstret : int64;
  outputs : (int * string) list;
      (** Each written slot's bytes ([In_out], [Out]), by index. *)
  status : int32;  (** The entry's return value. *)
  text : string;  (** The simulator's own output around the frame. *)
  wall : float;  (** Seconds. *)
}

val run :
  ?deadline:float ->
  ?flags:string list ->
  dir:string ->
  Coral_image.t ->
  bound:(int -> string option) ->
  (outcome, Run_error.t) Err.t
(** [dir] is the archive root. [bound k] is read slot [k]'s bytes, cut or
    zero-padded to its size. [deadline] is in seconds (default 120); [flags] go
    after the simulator's path ([--cycles=N] on RTL). *)
