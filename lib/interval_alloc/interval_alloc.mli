(** A standalone interval allocator. A client describes one pool as a script of
    allocations and frees; the allocator places every block at an offset so that
    blocks live at the same time never share cells, and [check] turns a
    placement into a {!Witness} that only a verified placement can produce.

    Units are the client's. Sizes, offsets and pool totals are [int64], since
    the library is reachable from js_of_ocaml, whose [int] is 32 bits. Time is a
    position in the script, so no tick type appears here. A block allocated at
    position [p] and freed at [q] is live from [p] up to, not including, [q]: an
    operand freed after its consumer's output was allocated conflicts with that
    output. *)

module Event : sig
  type 'k t = Alloc of { key : 'k; size : int64 } | Free of 'k
end

(** A placed block, as errors report it. *)
module Block : sig
  type 'k t = { key : 'k; offset : int64; size : int64 }
end

module Negative_size : sig
  type 'k t = { key : 'k; size : int64 }
end

module Out_of_pool : sig
  type 'k t = { block : 'k Block.t; pool : int64 }
end

module Overlap : sig
  type 'k t = { first : 'k Block.t; second : 'k Block.t }
end

module Script : sig
  type 'k t
  (** Validated: every key is allocated once and freed at most once, after its
      allocation, and no size is negative. *)

  val validate :
    equal:('k -> 'k -> bool) ->
    'k Event.t list ->
    ( 'k t,
      [> `Double_alloc of 'k
      | `Double_free of 'k
      | `Free_unknown of 'k
      | `Negative_size of 'k Negative_size.t ] )
    Err.t
  (** Keys are compared only here and in {!check}, so validation is quadratic in
      the number of blocks at worst. *)

  val events : 'k t -> 'k Event.t list
end

module Strategy : sig
  type t =
    | Greedy_by_area
    | Greedy_by_lifetime
    | Greedy_by_size
    | Greedy_by_size_best_fit

  val all : t list
  val equal : t -> t -> bool
  val pp : Format.formatter -> t -> unit
end

(** A search budget: how many iterations (one move and one decode each) and the
    seed of its moves. No clock: the same budget and seed give the same result
    on every backend. A negative count is treated as zero. *)
module Budget : sig
  type t

  val create : iterations:int64 -> seed:int64 -> t
end

module Stop : sig
  type t =
    | Budget_exhausted
    | Lower_bound
        (** The pool equals the lower bound, so it is optimal for this script's
            order, whatever the budget. *)
end

(** What a search did. *)
module Effort : sig
  type t = { iterations : int64; stop : Stop.t }
end

(** Offsets in allocation order plus the pool size. Anyone can build one; it is
    a claim until {!check} accepts it. *)
module Solution : sig
  type 'k t

  val placements : 'k t -> ('k * int64) list
  val pool : 'k t -> int64

  module Unsafe : sig
    val make : pool:int64 -> ('k * int64) list -> 'k t
    (** For tests that need a bad solution. *)
  end
end

(** A placement {!check} accepted. Abstract: only [check] makes one. *)
module Witness : sig
  type 'k t
end

val lower_bound : 'k Script.t -> (int64, [> `Live_overflow of 'k ]) Err.t
(** The largest sum of live sizes at once: no placement can use less. *)

module Stats : sig
  type t = {
    lower_bound : int64;
    constructive : Strategy.t * int64;
        (** The winning strategy of the portfolio and its pool. *)
    pool : int64;
    iterations : int64;
    stop : Stop.t;
  }

  val pp : Format.formatter -> t -> unit
end

val solve :
  Strategy.t -> 'k Script.t -> ('k Solution.t, [> `Pool_overflow of 'k ]) Err.t

val check :
  'k Script.t ->
  'k Solution.t ->
  ( 'k Witness.t,
    [> `Duplicate_placement of 'k
    | `Negative_offset of 'k
    | `Offset_overflow of 'k
    | `Out_of_pool of 'k Out_of_pool.t
    | `Overlap of 'k Overlap.t
    | `Unknown_key of 'k
    | `Unplaced of 'k ] )
  Err.t
(** Every block placed exactly once, inside the pool, and no two blocks live at
    the same time overlapping. *)

val placements : 'k Witness.t -> ('k * int64) list
(** In allocation order. *)

val pool : 'k Witness.t -> int64

val improve :
  Budget.t ->
  'k Script.t ->
  'k Solution.t ->
  ( 'k Solution.t * Effort.t,
    [> `Duplicate_placement of 'k
    | `Live_overflow of 'k
    | `Negative_offset of 'k
    | `Offset_overflow of 'k
    | `Out_of_pool of 'k Out_of_pool.t
    | `Overlap of 'k Overlap.t
    | `Pool_overflow of 'k
    | `Unknown_key of 'k
    | `Unplaced of 'k ] )
  Err.t
(** Search over block orders starting from a checked solution; never returns a
    larger pool than it was given. Stops at the lower bound, with
    [stop = Lower_bound], whatever the budget. *)

val solve_best :
  ?budget:Budget.t ->
  'k Script.t ->
  ( 'k Solution.t * Stats.t,
    [> `Live_overflow of 'k | `Pool_overflow of 'k ] )
  Err.t
(** The best of the four strategies, then {!improve}'s search from the winner.
    [Stats.stop = Lower_bound] iff the pool equals the bound, that is, is
    optimal for this script's order. *)
