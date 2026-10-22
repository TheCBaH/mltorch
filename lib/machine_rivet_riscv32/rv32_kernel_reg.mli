(** The registers the kernel AST names, one abstract type per class so an
    integer register cannot be passed where a floating-point or vector one is
    wanted. There is no unchecked constructor: a computed number goes through
    [of_index]. Each [all] and [index] keep architectural number order, which is
    the encoding. *)

module Class : sig
  (** Alphabetical. *)
  type t = Fpr | Gpr | Mtreg | Vreg

  val pp : t Fmt.t
end

module Register_index : sig
  type t = { class_ : Class.t; index : int }
  (** A register number that names no register of its class. *)

  val pp : t Fmt.t
end

module type Bank = sig
  type t

  val all : t list
  (** In number order. *)

  val compare : t -> t -> int
  val equal : t -> t -> bool

  val index : t -> int
  (** The architectural number, 0 to 31. *)

  val of_index : int -> (t, [> `Register_index of Register_index.t ]) Err.t

  val pp : t Fmt.t
  (** The ABI name ([a0], [fa0]), or [v0]. *)

  val to_rivet : t -> Riscv32_encode.Reg.t
end

module Gpr : sig
  include Bank

  val zero : t
  val ra : t
  val sp : t
  val gp : t
  val tp : t
  val t0 : t
  val t1 : t
  val t2 : t
  val s0 : t
  val s1 : t
  val a0 : t
  val a1 : t
  val a2 : t
  val a3 : t
  val a4 : t
  val a5 : t
  val a6 : t
  val a7 : t
  val s2 : t
  val s3 : t
  val s4 : t
  val s5 : t
  val s6 : t
  val s7 : t
  val s8 : t
  val s9 : t
  val s10 : t
  val s11 : t
  val t3 : t
  val t4 : t
  val t5 : t
  val t6 : t
end

module Fpr : sig
  include Bank

  val ft0 : t
  val ft1 : t
  val ft2 : t
  val ft3 : t
  val ft4 : t
  val ft5 : t
  val ft6 : t
  val ft7 : t
  val fs0 : t
  val fs1 : t
  val fa0 : t
  val fa1 : t
  val fa2 : t
  val fa3 : t
  val fa4 : t
  val fa5 : t
  val fa6 : t
  val fa7 : t
  val fs2 : t
  val fs3 : t
  val fs4 : t
  val fs5 : t
  val fs6 : t
  val fs7 : t
  val fs8 : t
  val fs9 : t
  val fs10 : t
  val fs11 : t
  val ft8 : t
  val ft9 : t
  val ft10 : t
  val ft11 : t
end

module Vreg : sig
  include Bank

  val v0 : t
  val v1 : t
  val v2 : t
  val v3 : t
  val v4 : t
  val v5 : t
  val v6 : t
  val v7 : t
  val v8 : t
  val v9 : t
  val v10 : t
  val v11 : t
  val v12 : t
  val v13 : t
  val v14 : t
  val v15 : t
  val v16 : t
  val v17 : t
  val v18 : t
  val v19 : t
  val v20 : t
  val v21 : t
  val v22 : t
  val v23 : t
  val v24 : t
  val v25 : t
  val v26 : t
  val v27 : t
  val v28 : t
  val v29 : t
  val v30 : t
  val v31 : t
end

(** Coral NPU's Zvt matrix tile registers, [mt0] to [mt15], each 16 rows of 16
    bytes. A 32-bit accumulator tile of 16 x 16 elements spans four of them, so
    [mt0], [mt4], [mt8] and [mt12] name the four independent ones. *)
module Mtreg : sig
  type t

  val all : t list
  (** In number order. *)

  val compare : t -> t -> int
  val equal : t -> t -> bool

  val index : t -> int
  (** The tile number, 0 to 15. *)

  val of_index : int -> (t, [> `Register_index of Register_index.t ]) Err.t

  val pp : t Fmt.t
  (** [mt0] to [mt15]: Rivet's spelling of the operand. *)

  val mt0 : t
  val mt4 : t
  val mt8 : t
  val mt12 : t
end
