(** What a backend can do with a vector, as data: legality and cost kept apart.

    A target says, per operation and element shape, whether it can express the
    operation natively, only by expanding it into scalar lane operations, or not
    at all (a refusal, never a silent degradation), and gives a relative cost.
    The vectorizer asks {e whether} an operation is legal from this description
    and only then {e whether it pays}; an expanded or costly operation is still
    legal, and the cost model alone decides to keep a loop scalar.

    Costs are in units of one scalar binary64 arithmetic operation. They are
    estimates for planning, recorded with the measurements that checked them,
    never part of what a program means. *)

(** The operations a vector body is made of. Closed and alphabetical. *)
module Op : sig
  type t =
    | Add
    | Bool_store
    | Broadcast_load
    | Const
    | Contiguous_load
    | Contiguous_store
    | Convert_i32_load
    | Div
    | Float_max
    | Fma
    | Index_value
    | Logic  (** [Not], [Or] on masks *)
    | Neg_abs
    | Round_f32
    | Select
    | Splat
    | Sqrt_trunc
    | Strided_load
    | Strided_store
    | Sub
    | Mul
    | Transcendental  (** [exp], [log], [sin], [cos], [erf] *)
    | Value_compare  (** [Value_eq], [Value_lt], [Pool_better] *)

  val all : t list
  val name : t -> string
end

type support =
  | Native  (** one or a few vector instructions *)
  | Expanded
      (** lane by lane through scalar operations: legal, usually a loss *)

type t = {
  name : string;
  vector_bits : int;  (** the physical register width *)
  precision : Loop_numerics.Precision.t;
      (** the working precision of the lanes this description prices *)
  lanes : int;
      (** the logical lane count the vectorizer plans for: how many lanes of
          [precision] one vector iteration covers (a multiple of the lanes of
          one register) *)
  inner_loops : bool;
      (** whether a vector loop may hold inner loops (reductions run per lane in
          lockstep): measured to pay on Wasm and to lose on native C *)
  fma : bool;
      (** whether the target has a guaranteed fused multiply-add the plan may
          use under a policy that permits contraction: native NEON does (through
          [fmaf]), standard Wasm SIMD does not *)
  relaxed_madd : bool;
      (** whether vector code may use a multiply-add the engine fuses or not at
          its choice ([f32x4.relaxed_madd]): a plan built for it states that its
          results are either one, never that they are bitwise reproducible *)
  row_block : int;
      (** how many consecutive iterations of a scalar loop around a vector loop
          one blocked iteration covers (see {!Loop_block}); 1 is no blocking.
          Registers, not lanes, bound it: each row holds its own accumulators *)
  support : Op.t -> support;
  cost : Op.t -> float;
      (** relative cost of one vector operation across [lanes], against [lanes]
          scalar operations at [1.0] each *)
  at : Loop_numerics.Precision.t -> t;
      (** the same machine at another working precision: lane count and prices
          change (a binary32 load converts nothing, [Round_f32] is free), the
          legality does not *)
}

val wasm128 : t
(** Portable WebAssembly SIMD: 128-bit registers, two binary64 lanes each, four
    logical lanes (two registers) by default. *)

val wasm128_relaxed : t
(** {!wasm128} with relaxed SIMD's [f32x4.relaxed_madd] for binary32 vector
    multiply-adds. A module planned for it needs the [relaxed-simd] feature: a
    host that cannot validate that probe plans for {!wasm128} instead. *)

val neon128 : t
(** AArch64 NEON, the installed native target: 128-bit registers, two binary64
    lanes each, four logical lanes. *)

val scalar : t
(** A target with no vector unit: every operation [Expanded] at the scalar cost,
    so nothing is ever profitable. The control for the cost model. *)

val forced : t -> t
(** The same legality with every cost zero, so the vectorizer takes every legal
    loop: for tests of the paths the cost model would decline (strided accesses,
    expanded transcendentals, bool stores). *)

val with_inner_loops : bool -> t -> t
(** The same target with vector loops that hold inner loops allowed or not, at
    every precision. *)

val with_row_block : int -> t -> t
(** The same target with another row-block factor, at every precision. *)

val f32 : t -> t
(** [f32 t] is [t.at F32]: the target a binary32 kernel is planned against. *)

val all : t list

val profitable : t -> (Op.t * int) list -> bool
(** [profitable target body] is true when the vector cost of the multiset of
    operations [body] (each with how many times it occurs per iteration) is
    below [target.lanes] scalar iterations of them. *)
