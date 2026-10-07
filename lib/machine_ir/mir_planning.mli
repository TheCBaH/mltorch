(** The resolved planning summary a lowering consumes: the numerical decisions
    structured SSA planning made, as immutable, serializable data bound to the
    exact program it planned. The SSA plan does not retain the target it was
    planned against and a bare SSA or CFG program carries no FMA provenance, so
    this summary is the only source of those permissions: a lowering refuses a
    program whose summary is missing or does not match, and never infers a
    permission from the opcodes it finds. *)

module Precision : sig
  type t = F32 | F64

  val name : t -> string
end

(** What a [Float_fma] in the planned program may mean. *)
module Fma : sig
  type t =
    | Exact
        (** contraction permitted and planned against a guaranteed fused
            operation: one rounding, compared bitwise *)
    | Forbidden  (** no contraction: the program holds no [Float_fma] *)
    | Relaxed_madd
        (** an external engine may fuse or not at its choice; Machine IR still
            executes one rounding, and only the external comparison boundary
            accepts either result *)

  val name : t -> string
end

(** A capability the plan assumed of its executor. Closed and alphabetical. *)
module Capability : sig
  type t =
    | Fused_multiply_add
    | Helper of string  (** a named math helper, e.g. ["exp"] *)
    | Vector_bits of int64  (** the physical register width it planned for *)

  val compare : t -> t -> int
end

type t = private {
  subject : string;
      (** digest of the planned program's canonical text: what this summary is
          about *)
  policy : string;  (** the numerical policy's stable identity *)
  schedule : string;  (** the schedule's identity: target and blocking *)
  precision : Precision.t;
  lanes : Mir_type.Lanes.t;  (** the logical vector width planned *)
  fma : Fma.t;
  capabilities : Capability.t list;  (** sorted, without duplicates *)
}

val make :
  subject:string ->
  policy:string ->
  schedule:string ->
  precision:Precision.t ->
  lanes:Mir_type.Lanes.t ->
  fma:Fma.t ->
  capabilities:Capability.t list ->
  t

val equal : t -> t -> bool

val to_string : t -> string
(** Canonical text, one [key=value] line per field: the serialized form. *)

val of_string : string -> (t, [> `Malformed_summary of string ]) result

type mismatch =
  [ `Missing_planning_summary
  | `Planning_subject_mismatch of string * string  (** expected, found *)
  | `Unauthorized_contraction of Fma.t ]

val pp_mismatch : Format.formatter -> [< mismatch ] -> unit

val admit :
  t option -> subject:string -> contracts:bool -> (t, [> mismatch ]) result
(** [admit summary ~subject ~contracts] accepts a summary bound to [subject];
    [contracts] says whether the program holds a fused multiply-add, which
    [Forbidden] refuses. *)
