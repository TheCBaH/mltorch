(* Why a published x86-64 artifact does not become a Rivet module. *)

type t =
  | Cpu_feature of string  (** a feature this CPU does not report *)
  | Cpu_unknown  (** a CPU whose features cannot be read *)
  | Form of string  (** a selected form Rivet rejected, by its rendering *)
  | Helper of string  (** a helper this route cannot bind *)
  | Location of string
  | Register of string
  | Rivet of string  (** the encoder's own message for a surface instruction *)

let pp fmt = function
  | Cpu_feature f -> Fmt.pf fmt "this CPU does not report feature %s" f
  | Cpu_unknown -> Fmt.string fmt "this CPU's features cannot be read"
  | Form f -> Fmt.pf fmt "no Rivet instruction for %s" f
  | Helper h -> Fmt.pf fmt "helper %s has no implementation in this image" h
  | Location l -> Fmt.pf fmt "%s where the form needs a register" l
  | Register r -> Fmt.pf fmt "%s is not a register of the form's file" r
  | Rivet m -> Fmt.string fmt m
