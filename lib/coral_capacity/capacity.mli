(** Image against configuration: the code budget, then the layout.

    The code budget is the ITCM region's size. An image over it is refused with
    the sizes of both, before the layout is asked, so a code overflow is not
    reported as a data one. Everything else, data capacity included, is the
    authority of {!Coral_layout.place}: the plan keeps no sum of its own. *)

module Code_over_budget : sig
  type t = { budget : Code_bytes.t; needed : Code_bytes.t }
end

type error = [ Coral_layout.error | `Code_over_budget of Code_over_budget.t ]

val pp_error : [< error ] Fmt.t

val budget : Coral_profile.t -> (Code_bytes.t, [> error ]) Err.t
(** The ITCM size; [`Missing_region Itcm] when the profile has none. *)

val check_code : Coral_profile.t -> Code_bytes.t -> (unit, [> error ]) Err.t
(** Refuses [`Code_over_budget] when the size exceeds {!budget}; a size equal to
    it fits. *)

module Placed : sig
  type t = {
    code : Code_bytes.t;
    layout : Coral_layout.t;
    slots : (string * Coral_layout.Placed_slot.t) list;
        (** Each allocation's name and where the layout put it. *)
  }
end

val place : Coral_profile.t -> Image.t -> (Placed.t, [> error ]) Err.t
(** {!check_code}, then {!Coral_layout.place} with one executable segment and
    one slot per allocation. *)
