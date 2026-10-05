(** Independent-output vectorization. A loop whose iterations are separate
    output cells, together with the loops inside it, becomes a loop over vectors
    of consecutive iterations: a cell's lane computes exactly what its scalar
    iteration did, in the same order, so vectorizing never reassociates a sum,
    fuses an operation or changes a rounding. A remainder runs as the original
    scalar loop.

    What is vectorized is derived from the program, never assumed:

    - the loop has constant bounds, carries only the effect, and its body holds
      no operation that can fail, no branch, no scratch object, no meter and no
      integer value that varies with the iteration;
    - every index that varies is affine in the induction value with a literal
      stride, so a memory access is a base coordinate and a step per axis;
    - the loop reads and writes the same cells in each iteration: a buffer it
      writes is accessed everywhere at the one coordinate, and, unless the
      caller states that distinct buffers are distinct memory, it touches no
      other buffer at all;
    - inner loops have bounds that do not vary with the iteration, and a value
      they carry becomes a vector when anything feeding it is one.

    A loop that fails any condition stays as lowered, and the decision names the
    condition. The numerical precision is whatever the program already is
    ({!Ssa_precision}); the target only decides the width and whether the vector
    body pays. *)

module Reason : sig
  type t =
    | Branch  (** the body holds an [if] *)
    | Loop_carried of Ssa_id.Buffer.t
        (** an iteration reads or writes a cell of this buffer that another
            iteration writes, or a store's cell is not the one it reads *)
    | No_vector_form of Ssa_op.t
        (** an operation a vector body cannot hold: one that can fail, a scratch
            or meter operation, an int64 that varies *)
    | Aliasing of Ssa_id.Buffer.t
        (** the loop writes memory and also touches this other buffer, which the
            alias policy does not rule out overlapping *)
    | Carries_values  (** the loop carries more than the effect *)
    | Inner_loops_declined  (** the target keeps loops out of vector bodies *)
    | Non_affine_access
        (** an index varies with the iteration other than by a literal stride *)
    | Non_constant_bounds
    | Store_through_broadcast  (** every lane would write the same cell *)
    | Strided_loop  (** the loop already steps by more than one *)
    | Too_short of { trips : int64; lanes : Ssa_type.Lanes.t }
    | Unprofitable  (** the target's cost model keeps it scalar *)
    | Varying_bounds  (** an inner loop's bounds depend on the iteration *)

  val name : t -> string
  (** A short stable name for tallies. *)

  val pp : Format.formatter -> t -> unit
end

module Decision : sig
  type outcome = Vectorized | Kept_scalar of Reason.t

  type t = {
    loop : Ssa_id.Region.t;  (** the loop's body region *)
    trips : int64;  (** iterations of the loop itself *)
    work : int64;
        (** the innermost-loop iterations one execution covers: [trips] for a
            leaf loop, the product with the inner loops' for a nest *)
    executions : int64;
        (** how many times the loop runs: the product of the constant trip
            counts of its enclosing loops *)
    ops : (Ssa_target.Op.t * int) list;
        (** the vector operations of its body, each weighted by the inner loops
            around it: the multiset the cost model prices *)
    outcome : outcome;
  }
end

type report = Decision.t list
(** One decision per loop tried, in program order; a loop absorbed by an
    enclosing vector loop has none of its own. *)

val program :
  ?alias:Ssa_effects.policy ->
  target:Ssa_target.t ->
  Ssa_program.t ->
  Ssa_program.t * report
(** Loops are tried innermost first: a loop whose body already holds a vector
    loop stays scalar around it, and a loop whose own iterations are independent
    takes its inner loops with it. [alias] defaults to the conservative policy:
    a caller that allocates every buffer on its own says
    {!Ssa_effects.Distinct_buffers}. The result verifies. The target's [lanes]
    is the logical width, and the program's own floating-point precision is
    kept. *)

val tally : report -> (string * (int * int64)) list
(** Per outcome name ([vectorized] or a reason's name): how many loops, and how
    many innermost-loop iterations ([work * executions]) they cover, in a stable
    order. *)
