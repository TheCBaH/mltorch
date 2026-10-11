(** Immutable payload bytes with explicit ownership.

    A blob is created through a {!Ledger}, which counts live bytes against a
    limit before any allocation is retained, and is held by one owner plus any
    number of leases. The bytes are freed (and the ledger credited) when the
    owner has dropped it and every lease has been released. Reading a blob after
    that raises [Invalid_argument]: a use after release is a defect, not a
    recoverable condition. *)

module Ledger : sig
  type t

  val create : ?limit:int64 -> unit -> t
  (** [limit] bounds live bytes; absent means unbounded. *)

  val live_blobs : t -> int
  val live_bytes : t -> int64
  val peak_bytes : t -> int64
  val outstanding_leases : t -> int

  val balanced : t -> bool
  (** No live blob and no outstanding lease: every handle was returned. *)
end

type t

val create : Ledger.t -> string -> (t, Codec_error.t) Err.t
(** Takes ownership of the string. [Resource_exceeded] when live bytes would
    pass the ledger's limit; nothing is retained then. *)

val contents : t -> string
val length : t -> int64

val drop : t -> unit
(** The owner's release. Raises [Invalid_argument] on a second call. *)

module Lease : sig
  type blob := t
  type t

  val release : t -> unit
  (** Raises [Invalid_argument] on a second call. *)

  val contents : t -> string
  val blob : t -> blob
end

val lease : t -> Lease.t
(** Raises [Invalid_argument] on a blob with no remaining holder. *)
