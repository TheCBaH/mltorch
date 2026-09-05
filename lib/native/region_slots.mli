(* The one flat-array slot layout shared by [Region_execution] (production)
   and [Region_eval] (reference): each local gets a contiguous
   [(offset, count)] range -- one slot for a scalar, [extent] for a vector --
   within a single per-key [float array]. Sharing this layout and its reader
   does not merge the two evaluators' own [evaluate_locals]/[emit] loops,
   which stay independently implemented so each remains its own oracle. *)

type t

val of_locals : Region_local.t list -> t
(** The running per-local [(offset, count)] fold. The offset arithmetic is plain
    [int], not [Int64]-checked: [Region_program.check] already bounds the SUM of
    every local's slot count against [max_size] on [Int64] before a
    [Region_program.t] can exist, so by the time a program reaches here the
    total is already proven to fit. *)

val total : t -> int
val offset : t -> Expr.Local_var.t -> (int * int) option

val reader :
  t ->
  float array ->
  (Expr.Local_var.t -> float option) * (Expr.Local_var.t -> int -> float option)
(** Reads an already-filled slot range: the first function answers a plain
    [Value.Local] (only meaningful for a scalar's single slot), the second a
    [Value.Local_at] at a computed position within a vector's range. *)
