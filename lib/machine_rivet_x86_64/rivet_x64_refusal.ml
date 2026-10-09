(* Why a published x86-64 artifact does not become a Rivet module. *)

type t =
  | Form of string  (** a selected form Rivet rejected, by its rendering *)
  | Helper of string  (** a helper this route cannot bind *)
  | Location of string
  | Register of string
  | Rivet of string  (** the encoder's own message for a surface instruction *)

let pp fmt = function
  | Form f -> Fmt.pf fmt "no Rivet instruction for %s" f
  | Helper h -> Fmt.pf fmt "helper %s has no implementation in this image" h
  | Location l -> Fmt.pf fmt "%s where the form needs a register" l
  | Register r -> Fmt.pf fmt "%s is not a register of the form's file" r
  | Rivet m -> Fmt.string fmt m
