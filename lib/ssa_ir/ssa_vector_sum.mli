(** Sums along their own axis. Under a policy that permits reordering a sum, a
    left fold that no enclosing loop's lanes took is scheduled across the vector
    unit: [parts] accumulator vectors, rounds of [parts] vectors, leftover
    vectors, a fixed adjacent-pair tree over the accumulators and then over the
    lanes, a sequential tail, and the seed.

    The schedule is a definition, not a heuristic: for [n] terms, [lanes] lanes
    and [parts] parts, with [full = n / lanes], [rounds = full / parts],
    [extra = full mod parts] and [tail = n - full * lanes], accumulator [j] lane
    [k] adds, in order, the terms at [lo + (r * parts + j) * lanes + k] for
    every round [r] and, for [j < extra], the term of the leftover vector; the
    accumulators are combined lane by lane by the adjacent-pair tree, the lanes
    by the same tree, the [tail] terms are added one by one to a fresh zero, and
    the result is [seed + (lanes' tree + tail)]. The logical work (one mark per
    term) is the sum's own, and a mutation that drops a tail, repeats a leftover
    vector or changes a tree turns the suite red.

    Only a sum whose body is straight-line, whose bounds are constant, whose
    term varies with the iteration through a contiguous load and which holds at
    least four vectors is scheduled; any other stays the sequential sum, with
    the reason recorded. *)

module Reason : sig
  type t =
    | Body_statement  (** a loop or branch in the body *)
    | Contiguous_load_missing
        (** nothing in the term reads consecutive cells, so a vector gains
            nothing *)
    | Lanes_declined  (** the target has no vector width to schedule across *)
    | Non_constant_bounds
    | Term_uniform  (** the term does not vary with the iteration *)
    | Too_short of { trips : int64; lanes : Ssa_type.Lanes.t }
    | Unsupported of Ssa_vector_body.Reason.t
        (** the term holds something a vector body cannot *)

  val name : t -> string
end

module Decision : sig
  type outcome =
    | Kept_sequential of Reason.t
    | Scheduled of { parts : int }
        (** how many accumulator vectors the schedule uses *)

  type t = { loop : Ssa_id.Region.t; trips : int64; outcome : outcome }
end

type report = Decision.t list

val program : target:Ssa_target.t -> Ssa_program.t -> Ssa_program.t * report
(** Schedules every eligible sum that is not inside a vector loop. The result
    verifies. The caller decides whether the numerical policy permits it:
    {!Ssa_numerics.reassociation_permitted}. *)
