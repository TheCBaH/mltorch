(** The quantities of a Wasm kernel's linear memory that are all [int] on the
    wire and must not be passed for each other: a cell width, an absolute
    address, an offset into the kernel's local region. Shared by the Loop and
    SSA Wasm emitters.

    An address or offset is bounded below 2{^ 31} at construction, so {!const}
    never wraps and no value here needs a 63-bit [int] under js_of_ocaml. *)

(** A power-of-two byte width: of a cell, or the natural alignment of an access.
*)
module Width : sig
  type t = W1 | W2 | W4 | W8

  val arg : t -> Wasm.Mem_arg.t
  (** An access aligned to the width, at offset zero. *)

  val bytes : t -> int64

  val of_load : Wasm.Load.t -> Wasm.Mem_arg.t
  (** An access at the load's natural alignment, at offset zero. *)

  val scale : t -> Wasm.Instr.t list
  (** Turns an [i32] cell index on the stack into a byte offset. *)
end

module type S = sig
  type t

  val zero : t
  val of_int64 : int64 -> t

  val to_int64 : t -> int64
  (** Raises [Invalid_argument] outside [\[0, 2{^31})]. *)

  val const : t -> Wasm.Instr.t
  (** The [i32] immediate. *)
end

(** An absolute address in the module's linear memory. *)
module Address : sig
  include S

  val data : t -> string -> Wasm.Data.t
  (** [bytes] placed at the address. *)
end

module Local_offset : S
(** A byte offset from the [local] base pointer, into the region every kernel's
    local arrays overlay. *)
