(** Non-negative least squares on relative error: the rates that minimize
    [sum_i ((rates . features_i - cycles_i) / cycles_i)^2] with every rate at
    least zero (Lawson and Hanson's active set), so each sample counts by its
    relative miss whatever its size. *)

val fit :
  ?priors:(string * float) list ->
  ?weight:float ->
  ((string * float) list * float) list ->
  Coral_model.Rates.t
(** Each sample: its features and its measured cycles (positive). Features no
    sample has are absent from the result. A prior [(feature, cycles)] adds a
    row pulling that feature's rate toward [cycles] with [weight] (default 1,
    one sample's relative miss): how a rate observed on the hardware keeps
    collinear classes apart. *)
