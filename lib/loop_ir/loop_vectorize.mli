(** The strict vectorizer: finds the loops of a {!Loop_program.t} whose
    iterations are independent output cells and rewrites them as vector loops,
    leaving everything else scalar. Strict means a lane computes exactly its
    scalar iteration (see the vectorization design record in [.ai/]): binary64
    working arithmetic, explicit [Round_f32], no reassociation, no fused
    multiply-add, and no change to what any program observes.

    The analysis reads the Loop IR, never printed backend source. A refused loop
    is a value with a structured reason and stays exactly as lowered: scalar
    retention around a vector loop is not an unsupported-graph fallback. *)

module Reason : sig
  (** Why a loop stays scalar. *)
  type t =
    | Body_statement of string
        (** a statement a vector body cannot hold: failure site, mark, local
            array, nested control, an index or int64 assignment *)
    | Expression of string
        (** a scalar expression with no vector form: a local array read, int64
            arithmetic, a format a vector load does not decode *)
    | Live_out_temp  (** a temporary assigned in the loop is read after it *)
    | Loop_carried
        (** an iteration reads what an earlier iteration wrote: a temporary read
            before it is assigned, or a stored buffer read at another cell *)
    | Non_affine_access
    | Non_constant_bounds
    | Store_through_broadcast
    | Too_short of { trips : int; lanes : int }
    | Unprofitable

  val name : t -> string
  (** A short stable name for tallies. *)

  val equal : t -> t -> bool
  val pp : Format.formatter -> t -> unit
end

module Decision : sig
  type outcome = Vectorized | Kept_scalar of Reason.t

  type t = {
    trips : int;  (** iterations of the loop itself *)
    work : int64;
        (** the innermost-loop iterations one execution covers: [trips] for a
            leaf loop, the product with the inner loops' for a nest *)
    executions : int64;
        (** how many times the loop runs: the product of the constant trip
            counts of its enclosing loops, [1] outside any *)
    ops : int;  (** vector operations in its body, when it has one *)
    outcome : outcome;
  }
end

type report = Decision.t list
(** One decision per loop tried, in program order; a loop absorbed by an
    enclosing vector loop has none of its own. *)

val program :
  ?target:Loop_target.t -> Loop_program.t -> Loop_vector.program * report
(** [target] defaults to {!Loop_target.wasm128}. Its [lanes] sets the logical
    width and its costs decide profitability. The scalar program is returned
    inside the result unchanged. *)

val tally : report -> (string * (int * int64)) list
(** Per outcome name ([vectorized] or a reason's name): how many loops, and how
    many innermost-loop iterations ([work * executions]) they cover, in a stable
    order. *)
