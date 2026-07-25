(* Tensor value clusters: which edge in the source graph is which edge in the
   destination, and what a verifier may assert about their values. The claim is
   supplied by the recipe and never inferred — see .ai/native_transform_design.md
   §3 and §6. *)

(* A lossy representation a value passed through. Compared by format name and
   quantization parameters, so two independent passes through BF16 collapse to
   one entry. *)
module Precision : sig
  type t = { fmt : Payload.packed_fmt; quant : Quant.t option }

  val compare : t -> t -> int
  val pp : Format.formatter -> t -> unit

  module Set : Set.S with type elt = t
end

(* Strictest to weakest: Identical <= Equivalent <= Approximate <= Unverifiable.

   [Approximate] records WHICH lossy representations the value passed through
   rather than ordering them: F16 and BF16 are incomparable (range versus
   mantissa) and quantization error depends on scale and saturation, so a
   "coarser of the two" would be a fiction. Composition is set union and choosing
   a tolerance is the checker's policy — which is also why F32 -> BF16 -> F32
   composes to [Approximate {bf16}] instead of reading as lossless.

   [Unverifiable] is the bottom: the two edges correspond structurally but
   nothing may be asserted about their values. *)
type relation =
  | Approximate of Precision.Set.t
  | Equivalent
  | Identical
  | Unverifiable

val join : relation -> relation -> relation
val pp_relation : Format.formatter -> relation -> unit

module Id :
  Cluster_relation.ID with type t = Tensor_id.t and module Set = Tensor_id.Set

module Label : Cluster_relation.LABEL with type t = relation
include module type of Cluster_relation.Make (Id) (Label)

(* Convenience constructors for the common shapes; [of_clusters] normalises. *)
val pair : Tensor_id.t -> Tensor_id.t -> relation -> Cluster.t
val create : Tensor_id.t -> Cluster.t
val delete : Tensor_id.t -> Cluster.t
