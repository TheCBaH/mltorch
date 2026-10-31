(** The functions generated Wasm calls besides itself: the four [Math] imports
    and the helpers defined in the module. Each helper is a transcription of its
    C or JavaScript counterpart ([Loop_c_runtime], [Loop_js_runtime]); they are
    emitted only when a kernel reaches them. *)

module Callee : sig
  (** Closed and alphabetical: the order helpers are emitted in. *)
  type t =
    | Ceil_div
    | Cos
    | Cos_f32  (** {!F32_math.Value.cos}'s sequence, one [f32] operation each *)
    | Cos_f32x4
        (** the same sequence on four lanes, one [f32x4] operation each *)
    | Erf
    | Erf_f32  (** {!F32_math.Value.erf}'s sequence, one [f32] operation each *)
    | Erf_f32x4
        (** the same sequence on four lanes, one [f32x4] operation each *)
    | Exp
    | Exp_f32  (** {!F32_math.Value.exp}'s sequence, one [f32] operation each *)
    | Exp_f32x4
        (** the same sequence on four lanes, one [f32x4] operation each *)
    | F16_to_f32  (** an [f16] cell's binary32 value, no binary64 step *)
    | F16_to_float
    | Fail_set
    | Fill_f32
    | Floor_div
    | Fma_f32  (** [F32_math.fma32]: [a * b + c] rounded once, as [fmaf] does *)
    | I64_div
    | Log
    | Log_f32  (** {!F32_math.Value.log}'s sequence, one [f32] operation each *)
    | Log_f32x4
        (** the same sequence on four lanes, one [f32x4] operation each *)
    | Sin
    | Sin_f32  (** {!F32_math.Value.sin}'s sequence, one [f32] operation each *)
    | Sin_f32x4
        (** the same sequence on four lanes, one [f32x4] operation each *)

  val all : t list

  val index : t -> int
  (** The position in [all]: the symbolic function index a kernel body calls
      before linking renumbers it. *)

  val of_index : int -> t

  val name : t -> string
  (** The helper's or import's name, as a manifest prints it. *)

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
