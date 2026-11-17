val backend : Ssa_fuzz.Fuzz.backend
(** {!Ssa_c.kernel} compiled with the host C compiler and run; a typed refusal
    is [Refused]. *)

val timed : (unit -> 'a) -> 'a
(** Runs the function, failing the run when it passes a five-minute cap. *)

val fail_on : (_, Ssa_fuzz.Fuzz.disagreement) result -> unit
(** Fails with the disagreement printed. *)
