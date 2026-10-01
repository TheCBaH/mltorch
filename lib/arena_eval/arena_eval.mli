(* Allocator evaluation over one kind's arena problem: every existing strategy,
   the generic order search from each, the production portfolio, and the
   reference minimum, all on exactly the same script. Placement quality and
   planning cost only: nothing here allocates a tensor.

   Every placement is checked before it is reported. No clock of its own: the
   caller passes [~now], in seconds, so a run's timings are the host's. *)

module Config : sig
  type t = {
    iterations : int64 list;  (** The order-search budgets, in iterations. *)
    seeds : int64 list;
    repeats : int;
        (** Timing samples per measurement; the fastest is kept. Placements are
            deterministic, so they are taken from the first. *)
    reference : Interval_alloc.Reference.Limits.t;
  }
end

(** How a row's placement was made. *)
module Method : sig
  type t =
    | Constructive of Interval_alloc.Strategy.t  (** One placement pass. *)
    | Improved of Interval_alloc.Strategy.t
        (** That strategy's placement, then [Interval_alloc.improve]: a generic
            first-fit order search, not the strategy's own decoder. *)
    | Portfolio  (** [Interval_alloc.solve_best], the production planner. *)

  val pp : Format.formatter -> t -> unit
end

(** Seconds. A portfolio's [construct] is its zero-budget run, which covers all
    four constructive passes; its [search] is what a budget adds to that. An
    improved row's [construct] is its strategy's own pass. *)
module Timing : sig
  type t = { construct : float; search : float; check : float }
end

module Row : sig
  type t = {
    method_ : Method.t;
    budget : (int64 * int64) option;
        (** Iterations and seed; [None] for a constructive pass, which has no
            search. *)
    constructive_pool : Core.Storage_units.Byte_size.t;
        (** The pool before any search. *)
    pool : Core.Storage_units.Byte_size.t;
    effort : Interval_alloc.Effort.t option;
    placements : (Tensor_id.t * Core.Storage_units.Byte_offset.t) list;
    digest : string;
    timing : Timing.t;
  }
end

module Reference_row : sig
  type t = {
    bounds :
      (Tensor_id.t * Core.Storage_units.Byte_offset.t) list
      Interval_alloc.Reference.Bounds.t;
        (** The incumbent is the placement of length [bounds.upper]. *)
    states : int64;
    max_depth : int64;
    seconds : float;
    digest : string;
  }
end

type error =
  [ `Arena_placement of Arena_plan.Placement_error.t
  | `Invalid_candidate of Interval_alloc.Reference.Invalid_candidate.t ]

val digest : (Tensor_id.t * Core.Storage_units.Byte_offset.t) list -> string
(** Of a placement, in the order given. *)

val strategies :
  now:(unit -> float) ->
  Config.t ->
  Tensor_id.t Interval_alloc.Script.t ->
  (Row.t list, [> error ]) Err.t
(** Per strategy in [Strategy.all] order: its constructive row, then one
    improved row per iteration budget and seed, each from that strategy's own
    placement; then one portfolio row per budget and seed. *)

val reference :
  now:(unit -> float) ->
  Config.t ->
  Tensor_id.t Interval_alloc.Script.t ->
  (Reference_row.t, [> error ]) Err.t
(** [Interval_alloc.Reference.minimum] from the zero-budget portfolio's
    placement. Timed once: a bounded search is not repeated. *)

val placed :
  Config.t ->
  Arena_problem.Kind_problem.t ->
  (Core.Storage_units.Byte_size.t, [> error ]) Err.t
(** The pool [Arena_plan.place] holds at the largest budget and the first seed:
    exact sizes, what a plan actually places, where every other figure here is
    of the script it is given. *)
