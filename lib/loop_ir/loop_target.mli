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
  lanes : int;
      (** the logical lane count the vectorizer plans for: how many binary64
          lanes one vector iteration covers (a multiple of the lanes of one
          register) *)
  support : Op.t -> support;
  cost : Op.t -> float;
      (** relative cost of one vector operation across [lanes], against [lanes]
          scalar operations at [1.0] each *)
}

val wasm128 : t
(** Portable WebAssembly SIMD: 128-bit registers, two binary64 lanes each, four
    logical lanes (two registers) by default. *)

val neon128 : t
(** AArch64 NEON, the installed native target: 128-bit registers, two binary64
    lanes each, four logical lanes. *)

val scalar : t
(** A target with no vector unit: every operation [Expanded] at the scalar cost,
    so nothing is ever profitable. The control for the cost model. *)

val all : t list

val profitable : t -> (Op.t * int) list -> bool
(** [profitable target body] is true when the vector cost of the multiset of
    operations [body] (each with how many times it occurs per iteration) is
    below [target.lanes] scalar iterations of them. *)
