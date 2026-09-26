(** The one table of the failure records emitted JavaScript returns: a failure
    is a value, [{ kind: "<name>", ... }], never a host exception. The emitter
    builds each record from it and the in-process executor decodes with it, so a
    field written is a field read, and a record's shape cannot drift from its
    reader. *)

(** The [kind:] strings, closed and alphabetical. *)
module Kind : sig
  type t =
    | Coord_out_of_range
    | Defect
    | Gather_index_out_of_range
    | I64_division_by_zero
    | I64_division_overflow
    | I64_from_float_infinite
    | I64_from_float_nan
    | I64_from_float_out_of_range
    | Index_overflow
    | Scan_meter
    | Scan_projection
    | Unbound_local

  val all : t list
  val to_string : t -> string
  val of_string : string -> t option
end

(** The record keys, closed and alphabetical. *)
module Field : sig
  type t =
    | Axis
    | Buffer
    | Cached
    | Coord
    | Extent
    | Index
    | Lane
    | Lhs
    | Limit
    | Op
    | Raw
    | Rhs
    | Row
    | Site
    | Value
    | Which

  val to_string : t -> string
end

val kind_key : string
(** The key every record carries its {!Kind} under. *)

val fields : Kind.t -> Field.t list
(** The keys a record of this kind carries after [kind], in the order written.
    [Site] is the ordinal of the [Fail_if] statement in program order: it names
    the static part of a failure (which local) that has no printed form. *)

val record : Kind.t -> (Field.t * Js_ast.expr) list -> Js_ast.expr
(** The record literal. The fields must be exactly [fields kind], in order:
    [Invalid_argument] otherwise, so a field cannot be forgotten or invented at
    one call site. *)

(** Values of the closed string-valued fields. *)

module Overflow_op : sig
  type t = Add | Mul

  val to_string : t -> string
  val of_string : string -> t option
end

module Projection : sig
  type t = Lane | Row

  val to_string : t -> string
  val of_string : string -> t option
end

module Meter : sig
  type t = State_over_limit | Updates_exhausted

  val to_string : t -> string
  val of_string : string -> t option
end

val sites : Loop_program.t -> Loop_failure.t array
(** The program's [Fail_if] failures in emission order (statements in program
    order, a [For] body, then an [If]'s branches): the [Site] of a record
    indexes this array. The emitter numbers the sites it writes in the same walk
    and checks each against this array by physical equality, so the two cannot
    drift. *)
