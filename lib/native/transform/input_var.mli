(* A symbolic variable standing for one CORRESPONDING pair of graph inputs: the
   σ hypothesis of .ai/native_transform_verify.md §7, "the two graphs are fed the
   same data".

   Its own namespace, deliberately. The representative used to be a
   [Tensor_id.t] allocated above both graphs' highest id, and the ceiling
   arithmetic existed only because "the two graphs share one numeric namespace" —
   reusing an id from the cluster (the minimum, say) collapsed the crossed pair
   {src t0 ↔ dst t1} and {src t1 ↔ dst t0} onto one variable and proved
   sub(a,b) identical to sub(b,a). A distinct type removes the premise rather
   than working around it: an input variable cannot collide with an edge because
   it is not an edge, and there is no ceiling to compute.

   Numbered by corresponding-input cluster, in cluster order, so a printed
   counterexample is reproducible. *)

type t

val compare : t -> t -> int
val equal : t -> t -> bool
val of_int : int -> t
val pp : Format.formatter -> t -> unit
val to_int : t -> int

module Map : Map.S with type key = t
