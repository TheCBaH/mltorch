(* Project-wide result plumbing and shared monad building blocks.

   [Monad] contains generic monad implementations (currently only [State]).
   All other values/types in this module concern the [result]-based error
   framework. *)

module Monad = Monad

(* Project-wide Result plumbing. The error side of every [result] is an
   [Error.t] — a polymorphic-variant payload plus a detection backtrace. This
   module owns only the generic machinery: smart constructors that capture the
   callsite, binding operators, and result-aware list folds. The error *sets*
   and their printers live in each component (a local [type error] + [pp_error]
   that composes via row union), never here. *)

module Error : sig
  (* The payload ['e] (the owning component's polymorphic-variant error row)
     plus the call stack captured the moment the error was built. It stays
     visible in the signature (never hidden behind an existential) so a
     function's error set is named at compile time. *)
  type +'e t = { kind : 'e; backtrace : Printexc.raw_backtrace }

  (* Build a bare [t] (not wrapped in a [result]), capturing the callstack the
     same way [fail] does. For call sites that need an [Error.t] directly
     (e.g. collecting a [t list] rather than short-circuiting on a [result]) —
     prefer [fail] when a [result] is what's actually needed. *)
  val make : 'e -> 'e t

  (* Print the payload via [pp_payload], then the detection backtrace. Core
     holds no knowledge of any payload's shape — the printer is supplied by the
     component. The backtrace block degrades to a notice when the binary carries
     no debug info (e.g. release without [-g]); the payload line always prints. *)
  val pp : (Format.formatter -> 'e -> unit) -> Format.formatter -> 'e t -> unit
end

type ('a, 'e) result = ('a, 'e Error.t) Stdlib.result

(* Build a failure from a payload, capturing the detection callstack
   automatically (no [~here] argument). *)
val fail : 'e -> ('a, 'e) result

(* Build a success ( = [Ok]); provided for symmetry with [fail] at success
   leaves. Pattern-match the transparent [Ok]/[Error] constructors when
   destructuring. *)
val return : 'a -> ('a, 'e) result

(* Rewrite the payload (e.g. to lift a sub-component's error into a wider row),
   preserving the original detection backtrace. *)
val map_error : ('e -> 'f) -> ('a, 'e) result -> ('a, 'f) result

module Syntax : sig
  val ( let* ) : ('a, 'e) result -> ('a -> ('b, 'e) result) -> ('b, 'e) result
  val ( let+ ) : ('a, 'e) result -> ('a -> 'b) -> ('b, 'e) result
  val ( >>= ) : ('a, 'e) result -> ('a -> ('b, 'e) result) -> ('b, 'e) result
  val ( >>| ) : ('a, 'e) result -> ('a -> 'b) -> ('b, 'e) result
end

(* Shared pretty-printing glue. Handwritten printers should use Fmt directly
   for their structure; this module only factors out repo-wide conventions that
   would otherwise repeat verbatim (stringifying printers, [none], and
   unwrapping [Core.Error.kind] in results). *)
module Pretty : sig
  val to_string : 'a Fmt.t -> 'a -> string
  val option_or : none:string -> 'a Fmt.t -> 'a option Fmt.t
  val result : ok:'a Fmt.t -> error:'e Fmt.t -> ('a, 'e) Stdlib.result Fmt.t
  val error_kind : 'e Fmt.t -> 'e Error.t Fmt.t
  val core_result : ok:'a Fmt.t -> error:'e Fmt.t -> ('a, 'e) result Fmt.t

  val capture_to_string :
    ?like:Format.formatter -> (Format.formatter -> unit) -> string
end

(* Result-aware list combinators: short-circuit on the first [Error] and thread
   the error row automatically, so migrated callers stay concise. *)
module List : sig
  val map : ('a -> ('b, 'e) result) -> 'a list -> ('b list, 'e) result

  val fold_left :
    ('acc -> 'a -> ('acc, 'e) result) -> 'acc -> 'a list -> ('acc, 'e) result
end
