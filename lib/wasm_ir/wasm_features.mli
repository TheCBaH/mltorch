(** The post-MVP features a module uses, found by scanning its instructions, so
    a host can check them before compiling and a deployment can state what it
    requires. A scalar module from this repository needs no SIMD and no relaxed
    instruction; {!Simd128} exists so a later vector emitter reports itself here
    too. *)

(** Closed and alphabetical. *)
type t =
  | Bulk_memory  (** [memory.copy], [memory.fill] *)
  | Non_trapping_float_to_int  (** the [trunc_sat] conversions *)
  | Relaxed_simd
      (** [f32x4.relaxed_madd]: off by default in node 20 (behind
          [--experimental-wasm-relaxed-simd]), so a host must ask the engine *)
  | Sign_extension  (** [i32.extend8_s], [i32.extend16_s] *)
  | Simd128  (** any [v128] instruction (none is representable yet) *)

val all : t list
val name : t -> string

val of_op : Wasm_op.t -> t option
(** The feature an operation needs, [None] for core ones. *)

val of_module : Wasm.Module.t -> t list
(** The features the module uses, in {!all} order, without duplicates. *)

val probe : t -> string
(** A tiny valid module (binary) that uses exactly the feature, for a host to
    validate: a runtime that rejects it cannot run a module needing it. Feature
    detection validates the actual extension rather than inferring support from
    a version number. *)
