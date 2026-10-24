(** One guest run on the subsystem simulator ([Vchip_*], a [Subsystem] profile).

    The image is written to a fresh directory with every slot's bytes in it
    (read slots from [bound], the others zero), since no HTIF call can carry
    them. The simulator backdoor-loads it, waits on its SPI-DPI port
    ([SPI_DPI_PORT], a free port), and the host starts the core over the SPI
    bridge ({!Coral_spi.start}). The host waits for the guest's byte on UART1,
    which the simulator logs, then reads the mailbox's done word over the bridge
    once and the record ({!Coral_mailbox}), and kills the whole process group:
    the simulator never exits on its own. *)

(** Bytes a run would read back over the bridge, against the caller's limit. *)
module Over_budget : sig
  type t = { bytes : int64; budget : int64 }
end

module Run_error : sig
  type t =
    | Digest_mismatch of int32  (** The guest's layout digest. *)
    | Guest_trap of Coral_run.Trap.t
    | Mailbox_malformed of Coral_mailbox.Malformed.t
    | No_core_csr  (** The profile has no core CSR region to start through. *)
    | Read_back_mismatch of int
        (** A slot's bytes from the bridge do not match the guest's hash. *)
    | Read_back_over_budget of Over_budget.t
        (** The slots to read back (after [skip_unchanged]) exceed [budget]:
            none is read. *)
    | Simulator_crashed of { status : string; tail : string }
        (** The process ended before the mailbox was done. *)
    | Simulator_missing of string
    | Spi_failed of string
    | Timeout

  val pp : t Fmt.t
end

type outcome = {
  argv : string list;
  hashes : (int * int32) list;
      (** Each written slot's {!Coral_mailbox.hash}, by slot index. *)
  mcycle : int64;
  minstret : int64;
  read_back : (int * string) list;
      (** The bytes of each slot asked for by [read_back], by slot index. *)
  status : int32;  (** The entry's return value. *)
  text : string;  (** The simulator's own output. *)
  wall : float;  (** Seconds from the core's start to the done word. *)
}

val run :
  ?deadline:float ->
  ?read_back:int list ->
  ?skip_unchanged:bool ->
  ?budget:int64 ->
  dir:string ->
  Coral_image.t ->
  bound:(int -> string option) ->
  (outcome, Run_error.t) Err.t
(** [dir] is the subsystem archive root. [bound k] is read slot [k]'s bytes, cut
    or zero-padded to its size. [deadline] is in seconds (default 600), from the
    launch. [read_back] (default none) are written slots read over the bridge
    after the guest is done, at about a kilobyte a second, for results a hash
    cannot carry, such as timings; each must still match its hash. With
    [skip_unchanged] (default [false]) a [read_back] slot whose hash says it
    still holds its starting bytes (its [bound] bytes, or zeros) is not read:
    those bytes are reported. [budget] (default none) bounds the bytes read
    back; above it the run fails with [Read_back_over_budget]. *)

val matches : outcome -> int -> string -> bool
(** [matches o k bytes]: written slot [k] hashed as [bytes] do. *)
