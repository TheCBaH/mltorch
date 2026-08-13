type t = private int

val of_int : int -> t (* builder-internal allocation *)
val to_int : t -> int
val equal : t -> t -> bool
val compare : t -> t -> int
val pp : Format.formatter -> t -> unit
val jsont : t Jsont.t

(* Raises [Invalid_argument] if allocating [count] consecutive ids starting at
   [next] would leave the id space. Every multi-output builder calls this before
   advancing its counter, so the two dialects' overflow behaviour cannot drift.

   [count > max_int - next], NOT [next + count > max_int]: js_of_ocaml's
   [Sys.int_size] is 32, so a wrapped sum sails straight past the naive
   comparison. A check on a wrapped result is not a bound — see
   .ai/js_backends_design.md.

   This is an INVARIANT, not a reachable resource ceiling, which is why it
   raises rather than returning a row the way [Shape_error.Output_count] does.
   Per-node output count is already bounded at [Kernel.Limits.Hard.outputs], so
   reaching the id space needs on the order of 2^31 live edges, each holding a
   [Tensor_sig.t] in a map: memory is exhausted first by orders of magnitude on
   every backend. The two failures are also different contracts — an
   over-limit node is a graph we decline to build, this is a builder that
   cannot keep its own promise. *)
val check_room : next:int -> count:int -> unit

module Map : Map.S with type key = t
module Set : Set.S with type elt = t
