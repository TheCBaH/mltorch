(** A standalone interval allocator. A client describes one pool as a script of
    allocations and frees; the allocator places every block at an offset so that
    blocks live at the same time never share cells, and [check] turns a
    placement into a {!Witness} that only a verified placement can produce.

    Sizes, offsets and pools are bytes ({!Core.Storage_units}), and every block
    has an alignment its offset must be a multiple of. Offsets are relative to
    the pool: that its base is aligned is the client's to provide. Time is a
    position in the script, so no tick type appears here. A block allocated at
    position [p] and freed at [q] is live from [p] up to, not including, [q]: an
    operand freed after its consumer's output was allocated conflicts with that
    output. *)

open Core.Storage_units

module Event : sig
  type 'k t =
    | Alloc of { key : 'k; size : Byte_size.t; alignment : Byte_alignment.t }
    | Free of 'k
end

(** A placed block, as errors report it. *)
module Block : sig
  type 'k t = { key : 'k; offset : Byte_offset.t; size : Byte_size.t }
end

module Misaligned : sig
  type 'k t = { block : 'k Block.t; alignment : Byte_alignment.t }
end

module Out_of_pool : sig
  type 'k t = { block : 'k Block.t; pool : Byte_size.t }
end

module Overlap : sig
  type 'k t = { first : 'k Block.t; second : 'k Block.t }
end

module Script : sig
  type 'k t
  (** Validated: every key is allocated once and freed at most once, after its
      allocation. *)

  val validate :
    equal:('k -> 'k -> bool) ->
    'k Event.t list ->
    ( 'k t,
      [> `Double_alloc of 'k | `Double_free of 'k | `Free_unknown of 'k ] )
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

  val placements : 'k t -> ('k * Byte_offset.t) list
  val pool : 'k t -> Byte_size.t

  module Unsafe : sig
    val make : pool:Byte_size.t -> ('k * Byte_offset.t) list -> 'k t
    (** For tests that need a bad solution. *)
  end
end

(** A placement {!check} accepted. Abstract: only [check] makes one. *)
module Witness : sig
  type 'k t
end

val lower_bound : 'k Script.t -> (Byte_size.t, [> `Live_overflow of 'k ]) Err.t
(** The largest sum of live sizes at once: no placement can use less. It ignores
    alignment, so it stays a bound, not always a tight one. *)

module Stats : sig
  type t = {
    lower_bound : Byte_size.t;
    constructive : Strategy.t * Byte_size.t;
        (** The winning strategy of the portfolio and its pool. *)
    pool : Byte_size.t;
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
    | `Misaligned of 'k Misaligned.t
    | `Offset_overflow of 'k
    | `Out_of_pool of 'k Out_of_pool.t
    | `Overlap of 'k Overlap.t
    | `Unknown_key of 'k
    | `Unplaced of 'k ] )
  Err.t
(** Every block placed exactly once, at a multiple of its alignment, inside the
    pool, and no two blocks live at the same time overlapping. *)

val placements : 'k Witness.t -> ('k * Byte_offset.t) list
(** In allocation order. *)

val pool : 'k Witness.t -> Byte_size.t

val fit : 'k Script.t -> 'k Witness.t -> 'k Witness.t
(** The same placement at its actual length: the pool cut to the highest
    occupied end, zero-size blocks at zero. [script] must be the one the witness
    was checked against. A placement checked against a script of larger blocks
    at the same alignments is one of this script too, and [fit] then gives back
    what the larger blocks' padding held. *)

val improve :
  Budget.t ->
  'k Script.t ->
  'k Solution.t ->
  ( 'k Solution.t * Effort.t,
    [> `Duplicate_placement of 'k
    | `Live_overflow of 'k
    | `Misaligned of 'k Misaligned.t
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

val improve_at :
  ?at:(int64 -> unit) ->
  iterations:int64 list ->
  seed:int64 ->
  'k Script.t ->
  'k Solution.t ->
  ( (int64 * 'k Solution.t * Effort.t) list,
    [> `Duplicate_placement of 'k
    | `Live_overflow of 'k
    | `Misaligned of 'k Misaligned.t
    | `Offset_overflow of 'k
    | `Out_of_pool of 'k Out_of_pool.t
    | `Overlap of 'k Overlap.t
    | `Pool_overflow of 'k
    | `Unknown_key of 'k
    | `Unplaced of 'k ] )
  Err.t
(** {!improve} at every budget in [iterations] (negative ones count as zero;
    sorted and deduplicated in the result), from ONE walk to the largest: each
    entry is what [improve] with that budget and [seed] returns, since a larger
    budget only extends the same walk, and a walk that reaches the lower bound
    stops, reporting that result at every later budget. [at b] is called as the
    result for budget [b] is taken, in increasing order: a caller's clock hook,
    this library has none. *)

val solve_best_at :
  ?at:(int64 -> unit) ->
  iterations:int64 list ->
  seed:int64 ->
  'k Script.t ->
  ( (int64 * 'k Solution.t * Stats.t) list,
    [> `Live_overflow of 'k | `Pool_overflow of 'k ] )
  Err.t
(** {!solve_best} at every budget in [iterations], the portfolio run once and
    its search walked once, as {!improve_at}. *)

(** A reference minimum-pool search, independent of the strategies above so its
    answers can grade them. [feasible] is a bounded, exhaustive test of one
    ceiling; [minimum] bisects between the live bound and a checked incumbent,
    so every result is a checked placement plus proven bounds
    [lower <= optimum <= upper]. A search cut by a limit is never reported as
    infeasible. No clock: work is counted in expanded search states. *)
module Reference : sig
  module Limits : sig
    type t = {
      max_states : int64;  (** Expanded states, over all queries of a search. *)
      max_depth : int64;  (** Branching decisions on one path. *)
    }
  end

  (** Work done so far, shared by the queries of one search. *)
  module Work : sig
    type t

    val create : unit -> t
    val states : t -> int64

    val max_depth : t -> int64
    (** The deepest path reached. *)
  end

  (** Which limit cut a search short. *)
  module Cut : sig
    type t = Depth | States
  end

  (** The answer to one ceiling; ['w] is a feasible answer's witness. *)
  module Answer : sig
    type 'w t =
      | Feasible of 'w
      | Infeasible  (** Proven: no placement fits. *)
      | Unknown of Cut.t
  end

  val feasible :
    Limits.t ->
    Work.t ->
    'k Script.t ->
    ceiling:Byte_size.t ->
    ( 'k Witness.t Answer.t,
      [> `Duplicate_placement of 'k
      | `Misaligned of 'k Misaligned.t
      | `Offset_overflow of 'k
      | `Out_of_pool of 'k Out_of_pool.t
      | `Overlap of 'k Overlap.t
      | `Unknown_key of 'k
      | `Unplaced of 'k ] )
    Err.t
  (** Does any placement fit in [ceiling]? A feasible answer's witness has its
      actual length as its pool. The error row is the checker's: reaching it
      means the search built a placement that fails its own check. *)

  (** A feasible answer outside the proven interval, or above its ceiling. *)
  module Invalid_candidate : sig
    type t = { pool : Byte_size.t; lower : Byte_size.t; ceiling : Byte_size.t }
  end

  module Status : sig
    type t =
      | Incomplete  (** [lower < upper]. *)
      | Optimal_above_live_bound
          (** [lower = upper] above the live bound: rests on exhaustive
              [Infeasible] answers. *)
      | Optimal_live_bound  (** [lower = upper = ] the live bound. *)
  end

  module Stop : sig
    type t = Closed | Depth_limit | State_limit
  end

  module Bounds : sig
    type 'w t = {
      status : Status.t;
      stop : Stop.t;
      live_bound : Byte_size.t;
      initial_upper : Byte_size.t;  (** The incumbent's actual length. *)
      lower : Byte_size.t;
      upper : Byte_size.t;
      incumbent : 'w;  (** A placement of length [upper]. *)
      queries : int64;
    }
  end

  val bisect :
    live_bound:Byte_size.t ->
    upper:Byte_size.t ->
    incumbent:'w ->
    pool:('w -> Byte_size.t) ->
    query:
      (Byte_size.t ->
      ( 'w Answer.t,
        ([> `Invalid_candidate of Invalid_candidate.t ] as 'e) )
      Err.t) ->
    ('w Bounds.t, 'e) Err.t
  (** The bisection behind [minimum], over any oracle: the first query is
      [live_bound], each later one the midpoint of what is open below the
      incumbent. A feasible answer tightens [upper] to its [pool], an infeasible
      one raises [lower] past its ceiling; an unknown one stops.
      [upper = live_bound] asks nothing. *)

  val minimum :
    Limits.t ->
    'k Script.t ->
    incumbent:'k Solution.t ->
    ( 'k Witness.t Bounds.t * Work.t,
      [> `Duplicate_placement of 'k
      | `Invalid_candidate of Invalid_candidate.t
      | `Live_overflow of 'k
      | `Misaligned of 'k Misaligned.t
      | `Offset_overflow of 'k
      | `Out_of_pool of 'k Out_of_pool.t
      | `Overlap of 'k Overlap.t
      | `Unknown_key of 'k
      | `Unplaced of 'k ] )
    Err.t
  (** Checks [incumbent], normalizes it to its highest occupied end, then
      bisects with [feasible] under one shared [Work]. *)
end
