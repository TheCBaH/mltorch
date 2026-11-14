(** The matched-policy bitwise golden of the acceptance harness: outputs are
    compared bit for bit, and only under the numerics policy the golden was made
    with. A comparison under another policy is refused, not weakened. *)

type t = { policy : string; values : float array }
(** Expected outputs, binary32 values held as floats, under a policy. *)

(** Alphabetical. *)
type mismatch =
  | Length_differs of { expected : Count.t; actual : Count.t }
  | Policy_differs of { expected : string; actual : string }
  | Value_differs of { index : Count.t; expected : int64; actual : int64 }
      (** the first element whose bits differ, as 64-bit patterns *)

val compare : t -> policy:string -> float array -> (unit, mismatch) result
(** [Ok ()] when [actual] has the golden's length and every element has exactly
    the golden's bits (a NaN must match its payload too). *)

val with_defect : float array -> float array
(** The same outputs with one deliberate defect: the lowest mantissa bit of the
    first element flipped. The suite must show {!compare} refuses it. *)

val pp_mismatch : Format.formatter -> mismatch -> unit
