(* Small guarded/shaped types shared by op [params] records (kernel size,
   stride, padding, …) — too trivial individually to earn their own files, so
   grouped here as submodules. See .ai/native_op_config.md. *)

module Nonneg : sig
  (* A non-negative int (>= 0), distinct from a bare int so a stray negative
     value (e.g. a typo'd pad) is caught at construction. Zero is a normal
     value here (e.g. "no padding") — contrast [Pos], where zero is
     rejected. *)
  type t = private int

  val of_int : int -> t (* raises [Invalid_argument] if < 0 *)
  val to_int : t -> int
  (* also available as the free coercion [(x :> int)] *)

  val pp : Format.formatter -> t -> unit
  val jsont : t Jsont.t
end

module Pos : sig
  (* A strictly positive int (>= 1) — for quantities where zero is
     meaningless (a stride of 0 would mean the output coordinate never
     advances), not merely non-negative. Contrast [Nonneg]. *)
  type t = private int

  val of_int : int -> t (* raises [Invalid_argument] if < 1 *)
  val to_int : t -> int
  (* also available as the free coercion [(x :> int)] *)

  val pp : Format.formatter -> t -> unit
  val jsont : t Jsont.t
end

module Bad : sig
  (* What can be wrong with an op-configuration value, and the only approved
     route from an untrusted one to a guarded type.

     HERE rather than in either importer. [Native_interp] reads serialized
     metadata and [Op_bridge] reads live tensors, but they must accept and
     reject the same node, and two vocabularies for "this stride is zero" is one
     drift away from two contracts. The asserting [Pos.of_int]/[Nonneg.of_int]
     above state a TRUSTED precondition; these state the same rule for a value
     that came from a model. Same split as [Dim.extent]/[Dim.extent_checked]. *)

  type param =
    [ `Dilation
    | `Groups
    | `Kernel_size
    | `Output_padding
    | `Output_size
    | `Padding
    | `Split_size
    | `Stride ]

  type fault = [ `Negative of int | `Not_positive of int ]
  (** Distinct because the constructors are: [Pos] forbids zero and [Nonneg]
      permits it, so collapsing them would report "not positive" for a padding
      of 0, which is ordinary. *)

  type t = { op : string; param : param; fault : fault }

  val pp_param : Format.formatter -> param -> unit
  val pp : Format.formatter -> t -> unit
  val pos : op:string -> param:param -> int -> (Pos.t, t) result

  val nonneg : op:string -> param:param -> int -> (Nonneg.t, t) result
  (** A plain [result], not an [Err.t]: the payload IS the error, and both
      callers immediately re-wrap it in their own domain's row — a detection
      origin minted here would be discarded at every call site. *)
end

module Hw : sig
  (* The H/W pair shape shared by conv/pool op config (kernel size, stride,
     padding). Scoped to H and W only — not a general N-axis vector (that's
     [Vec6], a different role: tensor shapes/coords over all six axes, not a
     pair of op hyperparameters). *)
  type 'a t = { h : 'a; w : 'a }

  val jsont : 'a Jsont.t -> 'a t Jsont.t
  val pp : (Format.formatter -> 'a -> unit) -> Format.formatter -> 'a t -> unit
end
