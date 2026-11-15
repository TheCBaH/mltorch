(** The scalar domains. [Index] is the language's valid 32-bit position and
    [I64] an exact two's-complement integer. [F32] and [F64] are different types
    even where an interpreter holds both in an OCaml [float]. No scalar is a
    byte offset: byte addressing is formed below this IR, by Machine IR
    lowering. *)

type scalar = F32 | F64 | I64 | Index | Pred

(** The lane count of a vector or mask. A domain of its own: no extent, position
    or id can be passed for it. *)
module Lanes : sig
  include Core.Tagged_int.S

  val decimal : t -> string
  (** The count as the decimal text a generated source embeds. *)

  val split : t -> 'a list -> 'a list * 'a list
  (** [split per l] is the first [per] items of [l] and the rest: one register's
      lanes. *)
end

module Bits : Core.Tagged_int.S
(** A width in bits: a scalar's, or a vector register's. *)

(** One register of a vector held in several: its position among them. A lane's
    register and a register's first lane are the two directions of one division
    by the lanes a register holds. *)
module Chunk : sig
  include Core.Tagged_int.S

  val all : Lanes.t -> per:Lanes.t -> t list
  (** The registers [lanes] lanes take at [per] lanes each, in order. A count
      that is not a multiple leaves the last register partial and unnamed: the
      caller refuses it first. *)

  val map : Lanes.t -> per:Lanes.t -> (t -> 'a) -> 'a list
end

(** One lane's position within a vector: a different domain from the count of
    lanes, so an extract's lane cannot be passed for a width. *)
module Lane : sig
  include Core.Tagged_int.S

  val zero : t

  val all : Lanes.t -> t list
  (** Every position of a vector of that many lanes, in order. *)

  val add : t -> by:t -> t
  (** The position [by] lanes past a lane. *)

  val chunk : t -> per:Lanes.t -> Chunk.t
  (** The register holding a lane, at [per] lanes a register. *)

  val slot : t -> per:Lanes.t -> t
  (** A lane's position inside its register. *)

  val first : Chunk.t -> per:Lanes.t -> t
  (** The first lane of a register. *)

  val offset : t -> stride:int64 -> int64
  (** The element offset of a lane, [stride] elements between lanes. *)
end

val max_lanes : Lanes.t
(** The widest vector the IR names. A logical width, not a register: a target
    maps it onto however many registers it takes. *)

(** [Effect] is a compile-time sequencing token with no machine representation.
*)
type t =
  | Effect
  | Local
  | Mask of Lanes.t
  | Scalar of scalar
  | Vec of scalar * Lanes.t

val bits : scalar -> Bits.t
(** The width of one lane in bits, which decides how many lanes one register
    holds. A predicate is a single truth value: as a vector it is a mask, whose
    lanes take the width of the values it compares. *)

val scalar_name : scalar -> string
val equal : t -> t -> bool
val pp : Format.formatter -> t -> unit

(** The phantom markers of the builder's typed handles. *)

type chain = Chain_marker
type f32 = F32_marker
type f64 = F64_marker
type i64 = I64_marker
type index = Index_marker
type local = Local_marker
type pred = Pred_marker
