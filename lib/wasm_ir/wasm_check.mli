(** A structural validator for {!Wasm.Module.t}: the spec's operand-stack
    algorithm over the scalar subset, plus the module-level index, export,
    memory and data checks. [Wasm_encode] runs it first, so an invalid program
    is an error before any byte is written. It is independent of the encoder and
    of any host engine: a defect in either is caught by the other, and by
    [WebAssembly.validate] in the tests. *)

module Reason : sig
  type t =
    | Alignment_too_large of { align : int; natural : int }
    | Bad_label of int
    | Bad_limits of { min_pages : int; max_pages : int option }
    | Const_expected
    | Data_out_of_memory of { offset : int; length : int }
    | Duplicate_export of string
    | Global_immutable of int
    | Memory_required
    | Offset_out_of_range of int
    | Operand_mismatch of {
        expected : Wasm_type.t;
        actual : Wasm_type.t option;  (** [None]: unreachable code, any type *)
      }
    | Select_operands_differ of Wasm_type.t * Wasm_type.t
    | Stack_height_mismatch of { expected : int; actual : int }
    | Stack_underflow
    | Unknown_export_target of int
    | Unknown_func of int
    | Unknown_global of int
    | Unknown_local of int
end

module Invalid : sig
  type t = {
    func : int option;
        (** index in the function index space (imports first), when the fault is
            inside a function body *)
    reason : Reason.t;
  }
end

type error = [ `Wasm_invalid of Invalid.t ]

val pp_error : Format.formatter -> [< error ] -> unit
val module_ : Wasm.Module.t -> (unit, error) Err.t
