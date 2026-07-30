(* A symbolic variable standing for one CORRESPONDENCE CLUSTER: the one value
   both of its sides are claimed to hold, named once so the two graphs' different
   ids for it compare equal.

   Its own namespace, deliberately. The representative used to be a
   [Tensor_id.t] allocated above both graphs' highest id, and the ceiling
   arithmetic existed only because "the two graphs share one numeric namespace" —
   reusing an id from the cluster (the minimum, say) collapsed the crossed pair
   {src t0 ↔ dst t1} and {src t1 ↔ dst t0} onto one variable and proved
   sub(a,b) identical to sub(b,a). A distinct type removes the premise rather
   than working around it: a cluster variable cannot collide with an edge because
   it is not an edge, and there is no ceiling to compute.

   What a variable is ENTITLED to assume is NOT this module's business, nor
   [Boundary_index]'s: both only say which cluster an edge belongs to. The driver
   grants one to every dependency cluster EXCEPT the cluster under test, which
   would otherwise name both its sides the same thing before either definition
   was looked at and discharge itself. The exception to that exception is a
   user-data graph input, which is σ — .ai/native_transform_verify.md §7. A
   variable one side alone names is crossed rather than stopped at (§13 of
   .ai/native_transform_local_verify_plan.md).

   Numbered by non-vacuous cluster, in [Graph_map.clusters_over] order, so a
   printed counterexample is reproducible. Printed [v0], never [t0]: a reader
   must not be able to mistake one for an edge. *)

type t

val compare : t -> t -> int
val equal : t -> t -> bool
val of_int : int -> t
val pp : Format.formatter -> t -> unit
val to_int : t -> int

module Map : Map.S with type key = t
