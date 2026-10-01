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

type node =
  | If of Loop_expr.pred * node list * node list
  | Loop of {
      var : Loop_var.t;
      lo : Loop_index.t;
      hi : Loop_index.t;
      body : node list;
    }  (** a scalar loop that contains a vector loop *)
  | Scalar of Loop_stmt.t
  | Vector of loop

type program = { scalar : Loop_program.t; body : node list }

val count_vector_loops : node list -> int
