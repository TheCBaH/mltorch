(* The typed vector program: a [Loop_program.t] in which some [For] loops are
   replaced by vector loops. It names lanes, element loads, lane-wise
   arithmetic, masks and stores; it names no register, intrinsic or byte
   width. The scalar program it came from is kept beside it: it is every
   backend's fallback and the reference every check compares with. *)

module Temp =
  Core.Tagged_int.Make
    (struct
      let prefix = "v"
    end)
    ()

(* One lane's element index is [offset] (a flat row-major offset in elements, a
   function of the loop variable) plus [lane * stride]: stride 0 is a broadcast,
   1 is contiguous, anything else is strided. [offset] is the offset of the first
   lane of a vector iteration, that is, the scalar offset at the iteration's
   loop-variable value. *)
module Access = struct
  type t = { buffer : Loop_buffer.t; offset : Loop_index.t; stride : int }
end

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
      (** a scalar expression that does not depend on the loop variable,
          evaluated once per vector iteration and replicated *)
  | Temp of Temp.t
  | Unary of Expr.Value.unary_op * t

and mask =
  | Not of mask
  | Or of mask * mask
  | Pool_better of t * t
  | Value_eq of t * t
  | Value_lt of t * t

(* The value a store converts, mirroring [Loop_stored]: [F32] narrows on write,
   [Bool] writes [v <> 0.] as a canonical 0/1 cell. *)
type stored = Bool of t | F32 of t

type stmt =
  | Assign of Temp.t * t
  | Index_assign of Loop_temp.t * Loop_index.t
  | Inner of {
      var : Loop_var.t;
      lo : Loop_index.t;
      hi : Loop_index.t;
      body : stmt list;
    }
  | Mark of Loop_mark.t
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

let rec count_vector_loops nodes =
  List.fold_left
    (fun n -> function
      | Scalar _ -> n
      | Vector _ -> n + 1
      | If (_, a, b) -> n + count_vector_loops a + count_vector_loops b
      | Loop { body; _ } -> n + count_vector_loops body)
    0 nodes
