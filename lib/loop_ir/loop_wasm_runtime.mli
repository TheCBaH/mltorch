(** The functions generated Wasm calls besides itself: the four [Math] imports
    and the helpers defined in the module. Each helper is a transcription of its
    C or JavaScript counterpart ([Loop_c_runtime], [Loop_js_runtime]); they are
    emitted only when a kernel reaches them. *)

module Callee : sig
  (** Closed and alphabetical: the order helpers are emitted in. *)
  type t =
    | Ceil_div
    | Cos
    | Erf
    | Exp
    | F16_to_float
    | Fail_set
    | Fill_f32
    | Floor_div
    | I64_div
    | Log
    | Sin

  val all : t list

  val index : t -> int
  (** The position in [all]: the symbolic function index a kernel body calls
      before linking renumbers it. *)

  val of_index : int -> t

  val import : t -> string option
  (** The [Math] function bound for an import, [None] for a defined helper. *)

  val deps : t -> t list
  (** The callees a helper's own body calls. *)
end

val import_module : string
(** The import module name of the four [Math] functions. *)

val call : Callee.t -> Wasm.Instr.t
val body : Callee.t -> Wasm.Func.t option
val signature : Callee.t -> Wasm.Func_type.t
