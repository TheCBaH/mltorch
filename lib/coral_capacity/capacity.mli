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

(** The candidates of one segment, each an image carrying the code size of its
    assembled form. One over the budget is rejected, with the sizes, and the
    rest are placed; none placed is a refusal of its own, with every reason. *)
module Admission : sig
  type rejection = { name : string; reason : error }
  type t = { placed : (string * Placed.t) list; rejected : rejection list }
end

val admit :
  Coral_profile.t ->
  Image.t list ->
  (Admission.t, [> `None_fit of Admission.rejection list ]) Err.t
(** Each image through {!place}, in order, keyed by its name. [`None_fit] when
    the list is empty or none was placed. *)

(** A candidate for selection: an image and the census of the program it runs.
*)
module Candidate : sig
  type t = { census : Coral_estimate.Coral_census.t; image : Image.t }
end

module Selection : sig
  type t = {
    cycles : float;  (** the estimate of the chosen candidate *)
    name : string;
    placed : Placed.t;
    ranked : (string * float) list;
        (** every placed candidate and its estimate, cheapest first *)
    rejected : Admission.rejection list;
  }
end

val select :
  Coral_profile.t ->
  core:Coral_estimate.Coral_model.Core.t ->
  rates:Coral_estimate.Coral_model.Rates.t ->
  Candidate.t list ->
  (Selection.t, [> `None_fit of Admission.rejection list ]) Err.t
(** {!admit} the images, price each placed candidate's census with
    {!Coral_estimate.Coral_model.features} and
    {!Coral_estimate.Coral_model.Rates.predict}, and choose the cheapest. Equal
    estimates go to the name that sorts first, then to the earlier candidate.
    [`None_fit] as {!admit}. *)
