(* Small guarded/shaped types shared by op [params] records (kernel size,
   stride, padding, …) — too trivial individually to earn their own files, so
   grouped here as submodules. See .ai/native_op_config.md. The scalars and the
   H/W pair are [Core.Geometry]'s (so [Expr] can take them); [Bad] and the wire
   codecs are added here. *)

module Nonneg : sig
  include module type of struct
    include Core.Geometry.Nonneg
  end

  val jsont : t Jsont.t
end

module Pos : sig
  include module type of struct
    include Core.Geometry.Pos
  end

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
    | `Repeats
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
  include module type of struct
    include Core.Geometry.Hw
  end

  val jsont : 'a Jsont.t -> 'a t Jsont.t
end
