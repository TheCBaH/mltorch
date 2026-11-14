(** The paired timing rule. A *sample* is the time per call of one batch; a
    *pair* is one old and one new sample taken back to back, the order
    alternating from pair to pair by the caller. The first 3 pairs are discarded
    as warm-up and 15 are kept. The gain of a pair is [100 * ln (t_old / t_new)]
    in percent. The statistic is the median gain with the interval from the 4th
    to the 12th of the 15 sorted gains, a distribution-free interval for the
    median with 96.5% coverage. *)

type pair = {
  old_time : float;  (** seconds per call, the accepted form *)
  new_time : float;  (** seconds per call, the candidate *)
  output_matches : bool;  (** the output was checked bitwise after the sample *)
}

(** Alphabetical. *)
type decision =
  | Accept  (** the interval's lower end is above 0 *)
  | Inconclusive
      (** neither end settles it: needs another reason or more pairs *)
  | Output_differs
      (** a sample's output differed: refused whatever the timing *)
  | Regression  (** the interval's upper end is below 0 *)

type report = {
  gains : float list;  (** the 15 kept gains, in measured order *)
  median : float;
  low : float;  (** the 4th sorted gain *)
  high : float;  (** the 12th sorted gain *)
  decision : decision;
}

(** Alphabetical. *)
type error =
  | Non_positive_time  (** a time of zero or less, or not a number *)
  | Too_few_pairs  (** fewer than the 3 warm-up and 15 kept *)
  | Too_many_pairs

val gain : old_time:float -> new_time:float -> float
(** [100 * ln (old_time / new_time)]. *)

val decide : pair list -> (report, error) result
(** The decision for the 18 pairs of a run, warm-up first. *)

val pp_decision : Format.formatter -> decision -> unit
val pp_report : Format.formatter -> report -> unit
