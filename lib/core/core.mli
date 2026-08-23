(* Repo-local printer glue.

   The error framework used to live here. It is [Err] (vendored/err_trace) now,
   and every call site names [Err] directly — see .ai/error_handling_design.md.
   What is left is the part that never belonged to a generic error library:
   conventions built on Fmt, which [Err] does not and must not depend on.

   The library keeps the name [core] because renaming it would touch every
   consumer to no effect. Read it as "the repo's shared printer glue".

   [Float_bits] is the other shared primitive in this library. Its explicit
   exact and portable policies keep structural float identities from silently
   choosing incompatible NaN semantics. *)

module Float_bits : module type of Float_bits

(* Handwritten printers should use Fmt directly for their structure; this
   module only factors out repo-wide conventions that would otherwise repeat
   verbatim (stringifying printers, [none], and unwrapping [Err.Error.kind] in
   results). See .ai/printer_conventions.md. *)
module Pretty : sig
  val to_string : 'a Fmt.t -> 'a -> string
  val option_or : none:string -> 'a Fmt.t -> 'a option Fmt.t
  val result : ok:'a Fmt.t -> error:'e Fmt.t -> ('a, 'e) Stdlib.result Fmt.t

  (* Print an [Err.Error.t] as its payload alone. The provenance is dropped, so
     this is for output a person reads, not a diagnostic — [Err.Error.pp]
     prints the whole wrapper. *)
  val error_kind : 'e Fmt.t -> 'e Err.Error.t Fmt.t

  (* The same, lifted over a whole [Err.t]. *)
  val err_result : ok:'a Fmt.t -> error:'e Fmt.t -> ('a, 'e) Err.t Fmt.t

  val capture_to_string :
    ?like:Format.formatter -> (Format.formatter -> unit) -> string
end
