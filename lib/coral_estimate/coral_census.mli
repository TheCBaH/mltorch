(** What an invocation's planned SSA program executes, counted by what costs
    differently on a Coral core: every operation of a body multiplied by its
    loops' trip counts, which are evaluated from the program's own index
    constants and induction values (a loop whose bounds depend on data is
    counted once and reported). The hand kernels a program calls ([Contract],
    [Attend]) are kept as their descriptors, priced by their own models. *)

(** Where a buffer's cells live. Alphabetical. *)
module Region : sig
  type t = Dtcm | Ram | Sram

  val name : t -> string
end

(** A cost class. Vector classes count 128-bit register groups, so a vector of
    eight binary32 lanes counts two. Alphabetical. *)
module Class : sig
  type t =
    | Branch  (** a structured [if] taken or not *)
    | F32_div
    | F32_fma
    | F32_math of Expr.Value.unary_op
        (** exp, log, sin, cos, erf, trunc: a call *)
    | F32_op  (** add, sub, mul, max, compare, select on binary32 *)
    | F32_sqrt
    | F64_op  (** any binary64 arithmetic *)
    | Int_op  (** index and int64 arithmetic, conversions, checks *)
    | Iteration  (** one trip of a loop *)
    | Load of Region.t
    | Local  (** a scratch cell read or written *)
    | Store of Region.t
    | Vec_f32_div
    | Vec_f32_fma
    | Vec_f32_op
    | Vec_load of Region.t  (** unit stride *)
    | Vec_load_strided of Region.t
    | Vec_math of Expr.Value.unary_op
    | Vec_other  (** splat, extract, insert, iota, masks *)
    | Vec_store of Region.t
    | Vec_store_strided of Region.t

  val compare : t -> t -> int
  val name : t -> string
end

(** A contraction call and where its operands live. *)
module Contract_call : sig
  type t = { desc : Contraction.t; a : Region.t; b : Region.t; c : Region.t }
end

(** An attention call and where its operands live ([kv]: K's, V's alike in every
    route so far). *)
module Attend_call : sig
  type t = {
    desc : Attention_block.t;
    q : Region.t;
    kv : Region.t;
    o : Region.t;
  }
end

type t = {
  counts : (Class.t * float) list;  (** by {!Class.compare}, none zero *)
  contracts : (Contract_call.t * float) list;  (** each call, how many times *)
  attends : (Attend_call.t * float) list;
  bodies : (Class.t * float) list list;
      (** the counts of each innermost loop body (a loop of instructions alone),
          trips included and one entry per distinct body: what runs in one
          pipeline, so what can overlap *)
  unknown : int;
      (** loops with a data-dependent bound counted once, and ifs with a
          data-dependent condition counted at half each branch *)
}

val program :
  ?region:(Ssa_ir.Ssa_id.Buffer.t -> Region.t) -> Ssa_ir.Ssa_program.t -> t
(** [region] defaults to [Dtcm] for every buffer. *)

val count : t -> Class.t -> float
val add : t -> t -> t
val pp : t Fmt.t
