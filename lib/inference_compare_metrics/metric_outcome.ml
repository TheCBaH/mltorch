(* What one group's score computation yields. Anything other than [Score] makes
   the metric unavailable for that group; it is never replaced by a number. *)
type t =
  | Nonfinite  (** An operand was NaN or infinite. *)
  | Numerical_failure  (** Overflow, or a result outside the metric's range. *)
  | Resource_failure
  | Score of float
  | Undefined  (** Empty group, zero norm, or a violated mass condition. *)
  | Unsupported

(* Worst first: the status a metric reports when groups disagree. *)
let severity = function
  | Resource_failure -> 5
  | Numerical_failure -> 4
  | Nonfinite -> 3
  | Unsupported -> 2
  | Undefined -> 1
  | Score _ -> 0
