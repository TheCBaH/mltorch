(** The bytes between the host and a Coral guest: one input stream on the
    guest's stdin, one frame on its stdout. Integers are little-endian; [u32]
    fields are [int32], [u64] fields [int64].

    {v
    input:  "CNPUIN01" version digest length payload "CNPUEND1"
    result: "CNPUOUT1" version digest status mcycle minstret length payload
            "CNPUEND1"
    trap:   "CNPUTRP1" version mcause mepc mtval "CNPUEND1"
    v}

    [digest] is the layout's ({!Coral_layout.t}); [mcycle] and [minstret] are
    the counters' deltas across the entry call ([u64]); the rest are [u32]. The
    payloads are the [In] slots' bytes, and the [Out] slots', in slot order.
    Simulators print their own text around the guest's, so the host looks for
    the first frame magic and keeps what surrounds it. *)

val version : int32

val input : digest:int32 -> string -> string
(** The stream for one run, around the concatenated [In] slots. *)

module Frame : sig
  type t =
    | Result of {
        mcycle : int64;
        minstret : int64;
        payload : string;
        status : int32;
      }
    | Trap of { mcause : int32; mepc : int32; mtval : int32 }
end

val result :
  digest:int32 ->
  status:int32 ->
  mcycle:int64 ->
  minstret:int64 ->
  string ->
  string
(** The result frame a guest writes, for tests of {!scan}. *)

val trap : mcause:int32 -> mepc:int32 -> mtval:int32 -> string
(** The trap frame a guest writes; it carries no digest, since a trap may
    precede the input. *)

module Malformed : sig
  type t =
    | Bad_trailer  (** The bytes after the frame are not ["CNPUEND1"]. *)
    | Digest_mismatch of { actual : int32; expected : int32 }
    | No_frame
    | Payload_size of { actual : int64; expected : int64 }
    | Truncated  (** The output ends inside a frame. *)
    | Version of int32

  val pp : t Fmt.t
end

module Scan : sig
  type t = { after : string; before : string; frame : Frame.t }
  (** [before] and [after] are the simulator's text around the frame. *)
end

val scan :
  digest:int32 -> payload_size:int64 -> string -> (Scan.t, Malformed.t) result
(** The first frame in a simulator's stdout. A result frame must carry [digest]
    and exactly [payload_size] payload bytes. *)
