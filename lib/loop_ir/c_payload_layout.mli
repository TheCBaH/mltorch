(** The byte layout of one payload file (weights, inputs or outputs) of the
    whole-model C backend. Pure: sizes and offsets are [int64], so the layout is
    js_of_ocaml-safe; reading and writing the files is the native host's.

    A file is a 64-byte header, then the tensors, each at a multiple of 64 from
    the start of the file, padding zero. See the C backend design record for the
    header's bytes. *)

module Role : sig
  type t = Inputs | Outputs | Weights

  val code : t -> int
  (** The header's role field: 1 weights, 2 inputs, 3 outputs. *)

  val to_string : t -> string
end

module Entry : sig
  type t = {
    id : Tensor_id.t;
    sg : Tensor_sig.t;
    offset : int64;  (** from the start of the file *)
    bytes : int64;  (** the dense storage: numel times the cell size *)
  }
end

type t = { role : Role.t; entries : Entry.t list; length : int64 }
(** [entries] are in the order given to {!create}; an id may repeat (a graph
    that lists an output twice), each occurrence having its own range. *)

type error = [ `Payload_overflow of Role.t * Tensor_id.t ]

val pp_error : Format.formatter -> [< error ] -> unit
val header_bytes : int64
val alignment : int64

val create :
  Role.t -> (Tensor_id.t * Tensor_sig.t) list -> (t, [> error ]) Err.t

val header : t -> identity:string -> string
(** The 64 header bytes. [identity] is 16 bytes. *)
