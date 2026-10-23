(** A client of the subsystem simulator's SPI-DPI server: the TCP port through
    which a host reaches the SoC's SPI-to-TileLink bridge, as the release's
    [tools/loader.py] does, in OCaml.

    A command is 17 bytes, little-endian: the command (u8), an address (u32),
    data (u64) and a count (u32), then its payload. The reply is the data (u64)
    and a success byte, then any payload. Bus transfers are whole 16-byte beats.
    The bridge moves about a kilobyte a second, so it starts the core and reads
    small records; it carries no slot bytes. *)

type t

exception Failed of string
(** A refused command or a broken connection, with the reason. *)

val connect : port:int -> t
(** Raises {!Failed} when the server does not answer. *)

val close : t -> unit

val idle : t -> int -> unit
(** Toggles the SPI clock for that many cycles, as the loader does to flush the
    reset synchronizers before the first transfer. *)

val read : t -> int64 -> beats:int -> string
(** [beats] 16-byte beats from a 16-byte-aligned address. *)

val write : t -> int64 -> string -> unit
(** Whole beats to a 16-byte-aligned address. *)

val write_word : t -> int64 -> int32 -> unit
(** One 32-bit word, by reading its beat and writing it back. *)

val start : t -> csr:int64 -> pc:int64 -> unit
(** Starts the core at [pc]: the PC to [csr + 4], then the clock gate released
    ([csr] = 1) and reset released ([csr] = 0). *)
