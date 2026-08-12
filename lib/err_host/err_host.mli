(* Host-side trace policy for mltorch's executables.

   [Err] deliberately reads no environment: a library that reconfigures itself
   from the ambient environment cannot be reasoned about by the program linking
   it. Choosing a policy is the HOST's job, and this module is where mltorch's
   executables do it. Nothing under lib/ other than this module may call
   [Err.Config.set], and nothing at all may call it as a module-load side
   effect. *)

(* The five variables, in the order [Err.Config.of_strings] takes them:

     MLTORCH_ERROR_TRACE              off | boundaries | all | map,filter,...
     MLTORCH_ERROR_BACKTRACE          off (or never) | origin | events
     MLTORCH_ERROR_MAX_EVENTS         <int>
     MLTORCH_ERROR_MAX_FRAMES         <int>
     MLTORCH_ERROR_MAX_EXTERNAL_BYTES <int>

   Surrounding whitespace is ignored, and the spellings above are the ones
   [Err.Config.pp] emits, so a logged policy can be fed back verbatim.

   Unset variables take [Err]'s own defaults, and if NONE of the five is set
   the policy is left alone entirely rather than reset to those defaults — so a
   host that has already chosen one keeps it.

   [MLTORCH_ERROR_TRACE=boundaries] with [MLTORCH_ERROR_BACKTRACE=off] is
   [Err.Config.deterministic]: identical diagnostics on every backend wherever
   a call site supplied [~pos]. Note that [off] on the TRACE axis is not the
   same thing — it discards the whole event trail, which is what
   [Err.Config.fast] does despite its name. *)
val vars : string list

(* Parse a policy from [lookup], which stands in for [Sys.getenv_opt] so this
   is testable without mutating the process environment. [None] means no
   variable was set. *)
val parse :
  (string -> string option) ->
  (Err.Config.t option, Err.Config.of_strings_error) Err.t

(* [parse Sys.getenv_opt], then [Err.Config.set] when a policy was given.
   Returns the policy actually installed, if any, so a caller can report it. *)
val install_from_env :
  unit -> (Err.Config.t option, Err.Config.of_strings_error) Err.t

(* Render a failure from either of the above. The payload alone: an invalid
   setting is the operator's typo, not a developer diagnostic. *)
val pp_error : Format.formatter -> Err.Config.of_strings_error -> unit
