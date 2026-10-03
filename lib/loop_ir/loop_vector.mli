(** The typed vector program (see the vectorization design record in [.ai/]).

    A vector program is a {!Loop_program.t} in which some [For] loops are
    replaced by vector loops, with the scalar program kept beside it: it is
    every backend's fallback and the reference every check compares with. Lanes
    are independent output cells of one loop; there is no register, intrinsic or
    byte width anywhere in these types, and a reduction is never split across
    lanes (a loop that carries one is not a vector loop). *)

(** A vector temporary: one value per lane, assigned and read within one vector
    iteration. *)
module Temp : sig
  include Core.Tagged_int.S
end

(** The cells one memory operation touches: lane [k] is the element at
    [offset + k * stride], where [offset] is a flat row-major offset in
    elements, a function of the loop variable giving lane 0 of the vector
    iteration. Stride 0 is a broadcast, 1 contiguous, anything else strided. *)
module Access : sig
  type t = { buffer : Loop_buffer.t; offset : Loop_index.t; stride : int }
end

(** A float vector, in the working precision: binary64. A narrowing to binary32
    is the explicit {!Round_f32}, never implied by a load or store format. *)
type t =
  | Binary of Expr.Value.binary_op * t * t
  | Const of float
  | Float_max of t * t
  | Fma of t * t * t
      (** [a * b + c] per lane with one rounding: a guaranteed fused operation,
          built only for a target that has one *)
  | Index_value of { base : Loop_index.t; step : int }
      (** lane [k] is [float (base + k * step)] *)
  | Load of Access.t
  | Round_f32 of t
  | Select of mask * t * t
  | Splat of float Loop_expr.t
      (** a scalar expression independent of the loop variable, evaluated once
          per vector iteration and replicated across lanes *)
  | Temp of Temp.t
  | Unary of Expr.Value.unary_op * t

and mask =
  | Not of mask
  | Or of mask * mask
  | Pool_better of t * t
  | Value_eq of t * t
  | Value_lt of t * t

(** What a store converts, as {!Loop_stored} does: [F32] narrows to binary32 on
    write, [Bool] writes [v <> 0.] as a canonical 0/1 cell. *)
type stored = Bool of t | F32 of t

type stmt =
  | Assign of Temp.t * t
      (** assigns the vector temporary, which an inner loop may update again (an
          accumulator): per lane, the scalar assignment sequence *)
  | Index_assign of Loop_temp.t * Loop_index.t
      (** an index temporary the same for every lane: its value does not depend
          on the loop variable *)
  | Inner of {
      var : Loop_var.t;
      lo : Loop_index.t;
      hi : Loop_index.t;
      body : stmt list;
    }
      (** a scalar loop around vector statements, the same trip count for every
          lane (its bounds do not depend on the loop variable): each iteration
          runs the body for all lanes in lockstep *)
  | Mark of Loop_mark.t
      (** an execution mark, once per lane: [lanes] marks per vector iteration
      *)
  | Store of { access : Access.t; value : stored }

type loop = {
  var : Loop_var.t;
  lo : Loop_index.t;
  hi : Loop_index.t;
  lanes : int;
      (** how many consecutive iterations one vector iteration covers: a logical
          width, which a target maps onto however many registers it takes *)
  body : stmt list;
  scalar : Loop_stmt.t;
      (** the original [For], whose body is also the scalar remainder *)
}
(** [\[lo, hi)] is strip-mined into full vectors of [lanes] iterations followed
    by the scalar remainder: [q = (hi - lo) / lanes] vector iterations at
    [var = lo + j * lanes], then [scalar] restricted to [\[lo + q * lanes, hi)].
*)

type vec = t

val shift : Loop_var.t -> int -> t -> t
(** [shift var delta e] is [e] evaluated [delta] iterations of [var] later: the
    same lanes, every access offset and index value base with [var] replaced by
    [var + delta]. A splat does not mention [var] and is unchanged. *)

(** A sum scheduled along its own axis, under a policy that permits reordering
    it (see the fp32 design record in [ai/]): the terms of [\[lo, hi)] are
    summed in vectors of [lanes] consecutive terms across [parts] independent
    accumulators, combined in a fixed tree.

    Precisely, with [n = hi - lo], [full = n / lanes] vectors,
    [main = full / parts] rounds and [extra = full mod parts] left over, every
    accumulator lane [a(j, k)] starts at [+0.]; round [i] adds term
    [lo + (i * parts + j) * lanes + k] into [a(j, k)]; the [extra] vectors go to
    accumulators [0 .. extra - 1] in order; each lane [k] then becomes the
    adjacent-pair tree of [a(0, k) .. a(parts - 1, k)], and the lanes the
    adjacent-pair tree of those; the [n - full * lanes] trailing terms are
    summed sequentially from [+0.] into [t]; and the result is
    [seed + (tree + t)]. One [Reduction] mark runs per term. That is a
    definition of the answer, not an approximation of the sequential sum, and
    {!Loop_vector_expand} spells it out as scalar statements for the
    interpreter. *)
module Reduction : sig
  type t = {
    acc : Loop_temp.t;  (** the float temporary the result is assigned to *)
    seed : float;
    var : Loop_var.t;  (** the sum's variable, free in [term] *)
    lo : int;
    hi : int;
    lanes : int;  (** terms in one accumulator vector *)
    parts : int;  (** independent accumulator vectors *)
    term : vec;
        (** lane [k] of a vector at [var = b] is the scalar term at [b + k]: its
            loads are affine in [var] with the stride their access records *)
    fused : bool;
        (** the accumulate is a fused multiply-add: [term] is a product [a * b]
            and every step, the tail's included, is [fma(a, b, accumulator)]
            with one rounding instead of [accumulator + a * b] with two *)
    scalar : Loop_stmt.t;
        (** the {!Loop_stmt.Reduce_sum} this stands for: every backend's
            sequential fallback *)
  }
end

type node =
  | If of Loop_expr.pred * node list * node list
  | Loop of {
      var : Loop_var.t;
      lo : Loop_index.t;
      hi : Loop_index.t;
      body : node list;
    }  (** a scalar loop that contains a vector loop *)
  | Reduction of Reduction.t
  | Scalar of Loop_stmt.t
  | Vector of loop

type program = { scalar : Loop_program.t; body : node list }

val count_vector_loops : node list -> int
