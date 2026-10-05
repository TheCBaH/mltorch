(** The typed construction interface. Handles carry a phantom type, so an
    operation applied to the wrong domain is a compile error; hand-built or
    transformed programs bypass this and are checked by {!Ssa_verify} alone.

    The builder threads the effect chain itself: an effectful call consumes the
    current effect and replaces it, and a region receives, carries and yields
    one without the caller naming it. A call that breaks a precondition the
    types cannot express (an out-of-domain literal, an undeclared buffer, a
    decode of the wrong format) raises [Invalid_argument]: it is a defect in the
    caller, not a verdict about a program. *)

type t
(** A builder for one region. *)

type 'a value = private Ssa_value.t

(** A typed tuple, for the values a loop or a branch carries. *)
type _ pack =
  | Nil : unit pack
  | Cons : 'a value * 'rest pack -> ('a * 'rest) pack

type access =
  | Coord of Ssa_type.index value Expr.Coord.t
  | Flat of Ssa_type.index value

val program :
  buffers:Ssa_buffer.t list ->
  (t -> unit) ->
  (Ssa_program.t, Ssa_verify.error) Err.t
(** Builds the entry region, then verifies the finished revision. *)

val set_origin : t -> Ssa_origin.t -> unit
(** The origin of every operation built after this call, in this builder and its
    regions. *)

(** {1 Constants and conversions} *)

val f32 : t -> float -> Ssa_type.f32 value
(** The value must already be representable in binary32. *)

val f64 : t -> float -> Ssa_type.f64 value
val i64 : t -> int64 -> Ssa_type.i64 value

val index : t -> int64 -> Ssa_type.index value
(** Inside the index domain. *)

val pred : t -> bool -> Ssa_type.pred value
val f32_to_f64 : t -> Ssa_type.f32 value -> Ssa_type.f64 value
val f64_to_f32 : t -> Ssa_type.f64 value -> Ssa_type.f32 value
val index_to_f64 : t -> Ssa_type.index value -> Ssa_type.f64 value
val index_to_i64 : t -> Ssa_type.index value -> Ssa_type.i64 value

(** {1 Arithmetic} *)

val f64_binary :
  t ->
  Expr.Value.binary_op ->
  Ssa_type.f64 value ->
  Ssa_type.f64 value ->
  Ssa_type.f64 value

val index_add :
  t -> Ssa_type.index value -> Ssa_type.index value -> Ssa_type.index value
(** Checked: the first operation to leave the index domain fails with its
    operands. *)

val index_scale : t -> int64 -> Ssa_type.index value -> Ssa_type.index value

(** {1 Memory and accounting} *)

val load_f64 :
  t -> Ssa_id.Buffer.t -> decode:Ssa_op.Decode.t -> access -> Ssa_type.f64 value
(** [decode] must produce a float. *)

val load_i64 : t -> Ssa_id.Buffer.t -> access -> Ssa_type.i64 value

val store_f64 :
  t ->
  Ssa_id.Buffer.t ->
  encode:Ssa_op.Encode.t ->
  access ->
  Ssa_type.f64 value ->
  unit

val store_i64 : t -> Ssa_id.Buffer.t -> access -> Ssa_type.i64 value -> unit
val mark : t -> Ssa_mark.t -> unit

(** {1 Structured control} *)

val for_ :
  t ->
  lo:Ssa_type.index value ->
  hi:Ssa_type.index value ->
  init:'s pack ->
  (t -> Ssa_type.index value -> 's pack -> 's pack) ->
  's pack
(** A loop over [lo, hi) with step one. The body receives the induction value
    and the carried values and returns the next carried values. *)

val if_ :
  t ->
  Ssa_type.pred value ->
  then_:(t -> 's pack) ->
  else_:(t -> 's pack) ->
  's pack
(** Only the selected region executes. Both return the same signature. *)

val ordered_sum :
  t ->
  lo:Ssa_type.index value ->
  hi:Ssa_type.index value ->
  seed:Ssa_type.f64 value ->
  (t -> Ssa_type.index value -> Ssa_type.f64 value) ->
  Ssa_type.f64 value
(** The left fold of the terms the body returns, from [seed]. *)
