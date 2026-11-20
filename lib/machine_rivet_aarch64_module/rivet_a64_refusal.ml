(* Why a published AArch64 artifact does not become a Rivet module. Payloads
   name the form or location; the printer is the only prose. *)

open Machine_ir

type t =
  | Address_offset of { symbol : string; addend : int64 }
      (** a symbol offset the form cannot carry *)
  | Contraction_format of Contraction.Element.t
      (** a contraction over operands this route does not realize *)
  | Contraction_numerics of Contraction.Numerics.t
      (** a contraction under a rounding order this route does not realize *)
  | Contraction_stride of int64
      (** a contraction stride, in elements, past what a byte step carries *)
  | Cpu_feature of string  (** a feature this CPU does not report *)
  | Cpu_unknown  (** a CPU whose features cannot be read *)
  | Form of string  (** a selected form Rivet has no instruction for *)
  | Frame_access of { bytes : int64; bank : Mir_target.Bank.t }
      (** a save or reload of a width or register file with no scalar form *)
  | Helper of string
      (** a helper bound to a C library symbol, under the dependency-free mode
      *)
  | Host_symbol of string  (** a helper the process does not define *)
  | Location of string
      (** a location of this shape where the form needs a register *)
  | Register of string  (** a view that is no AArch64 register of its file *)
  | Unresolved_reference of Mir_id.Func.t
      (** an instruction naming a symbol the artifact's relocations omit *)

let pp fmt = function
  | Address_offset { symbol; addend } ->
      Fmt.pf fmt "symbol %s+%Ld: an offset no form carries" symbol addend
  | Contraction_format e ->
      Fmt.pf fmt "no AArch64 contraction over %s operands"
        (Contraction.Element.name e)
  | Contraction_numerics n ->
      Fmt.pf fmt "no AArch64 contraction under %s numerics"
        (Contraction.Numerics.name n)
  | Contraction_stride s ->
      Fmt.pf fmt "contraction stride %Ld is past what a byte step carries" s
  | Cpu_feature f -> Fmt.pf fmt "this CPU does not report feature %s" f
  | Cpu_unknown -> Fmt.string fmt "this CPU's features cannot be read"
  | Form f -> Fmt.pf fmt "no Rivet instruction for %s" f
  | Frame_access { bytes; bank } ->
      Fmt.pf fmt "no scalar form moves %Ld bytes of the %s bank" bytes
        (Mir_target.Bank.name bank)
  | Helper h ->
      Fmt.pf fmt
        "helper %s needs the system math library, which the mode forbids" h
  | Host_symbol n ->
      Fmt.pf fmt "host symbol %s does not resolve in this process" n
  | Location l -> Fmt.pf fmt "%s where the form needs a register" l
  | Register r -> Fmt.pf fmt "%s is not a register of the form's file" r
  | Unresolved_reference f ->
      Fmt.pf fmt "%a names a symbol the relocations omit" Mir_id.Func.pp f
