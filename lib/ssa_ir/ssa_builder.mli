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
  ?scan_limits:Expr.Scan_limits.t ->
  buffers:Ssa_buffer.t list ->
  (t -> unit) ->
  (Ssa_program.t, Ssa_verify.error) Err.t
(** Builds the entry region, then verifies the finished revision. *)

val probe : t -> (t -> 'a) -> unit
(** Runs the function against a block that is discarded, to learn what building
    something would involve before building it. It leaves gaps in the id space
    and nothing else. *)

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

val float_to_i64 : t -> Ssa_type.f64 value -> Ssa_type.i64 value
(** Checked: NaN, an infinity or a value from 2^63 up, or below -2^63, fails. *)

val i64_arith :
  t ->
  Ssa_op.I64_op.t ->
  Ssa_type.i64 value ->
  Ssa_type.i64 value ->
  Ssa_type.i64 value
(** Modular: wraps, and cannot fail. *)

val i64_div :
  t -> Ssa_type.i64 value -> Ssa_type.i64 value -> Ssa_type.i64 value
(** Truncating toward zero; fails on a zero divisor, then on [min_int / -1]. *)

val i64_compare :
  t ->
  Ssa_op.Compare.t ->
  Ssa_type.i64 value ->
  Ssa_type.i64 value ->
  Ssa_type.pred value

val i64_to_f64 : t -> Ssa_type.i64 value -> Ssa_type.f64 value
(** One rounding, at binary64. *)

val i64_to_f32 : t -> Ssa_type.i64 value -> Ssa_type.f32 value
(** One rounding, at binary32: not a conversion through binary64. *)

val index_of_i64 : t -> Ssa_type.i64 value -> Ssa_type.index value
(** For a value already inside the index domain; outside it is a defect. *)

val f64_max :
  t -> Ssa_type.f64 value -> Ssa_type.f64 value -> Ssa_type.f64 value
(** [Float.max], with the NaN and signed-zero behavior of [Expr.Max_op]. *)

val f64_unary :
  t -> Expr.Value.unary_op -> Ssa_type.f64 value -> Ssa_type.f64 value

val float_compare :
  t ->
  Ssa_op.Compare.t ->
  Ssa_type.f64 value ->
  Ssa_type.f64 value ->
  Ssa_type.pred value
(** Ordered: false when either side is NaN, and signed zeros are equal. *)

val pool_better :
  t -> Ssa_type.f64 value -> Ssa_type.f64 value -> Ssa_type.pred value
(** [pool_better best value]: the candidate wins on strict greater-than or on
    NaN. *)

val index_add :
  t -> Ssa_type.index value -> Ssa_type.index value -> Ssa_type.index value
(** Checked: the first operation to leave the index domain fails with its
    operands. *)

val index_scale : t -> int64 -> Ssa_type.index value -> Ssa_type.index value

val index_floor_div : t -> int64 -> Ssa_type.index value -> Ssa_type.index value
(** By a positive literal: the mathematical floor, also for a negative
    numerator. *)

val index_ceil_div : t -> int64 -> Ssa_type.index value -> Ssa_type.index value

val index_min :
  t -> Ssa_type.index value -> Ssa_type.index value -> Ssa_type.index value

val index_max :
  t -> Ssa_type.index value -> Ssa_type.index value -> Ssa_type.index value

val index_clamp_low : t -> Ssa_type.index value -> Ssa_type.index value
(** [max 0 x]: total in the index domain. *)

val index_compare :
  t ->
  Ssa_op.Compare.t ->
  Ssa_type.index value ->
  Ssa_type.index value ->
  Ssa_type.pred value

val pred_not : t -> Ssa_type.pred value -> Ssa_type.pred value

val pred_or :
  t -> Ssa_type.pred value -> Ssa_type.pred value -> Ssa_type.pred value
(** Both operands are already computed; not a short-circuit. *)

val select : t -> Ssa_type.pred value -> 'a value -> 'a value -> 'a value
(** Chooses between values already computed; an arm that must stay lazy is an
    [if]. *)

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

(** {1 Scratch locals and the scan meter} *)

val local_alloc :
  ?var:Expr.Local_var.t -> t -> slots:int64 -> Ssa_type.local value
(** A fresh object of [slots] binary64 cells, every cell unset: reading one that
    was never written is a defect. [var] names the variable an out-of-range read
    reports as unbound. *)

val local_read :
  t -> Ssa_type.local value -> Ssa_type.index value -> Ssa_type.f64 value

val local_write :
  t ->
  Ssa_type.local value ->
  Ssa_type.index value ->
  Ssa_type.f64 value ->
  unit

val check_local :
  t -> var:Expr.Local_var.t -> extent:int64 -> Ssa_type.index value -> unit
(** The position must lie between zero and [extent] (exclusive) of the
    variable's own range. *)

val check_scan :
  t ->
  var:Expr.Local_var.t option ->
  row:Ssa_type.index value ->
  lane:Ssa_type.index value ->
  row_extent:int64 ->
  lane_extent:int64 ->
  unit
(** The row, then the lane, against their extents: the row wins. *)

val meter_reset : t -> unit
val meter_charge : t -> unit
val meter_reserve : t -> width:int64 -> unit
val meter_release : t -> width:int64 -> unit

val check_gather : t -> Ssa_type.i64 value -> extent:int64 -> unit
(** A gather's raw index must lie between [-extent] and [extent - 1]. *)

val check_access : t -> Ssa_id.Buffer.t -> access -> unit
(** The bounds check of a coordinate load, without the read. *)

(** {1 Structured control} *)

val for_ :
  t ->
  lo:Ssa_type.index value ->
  hi:Ssa_type.index value ->
  init:'s pack ->
  (t -> Ssa_type.index value -> 's pack -> 's pack) ->
  's pack
(** A loop from [lo] up to, not including, [hi], with step one. The body
    receives the induction value and the carried values and returns the next
    carried values. *)

val if_ :
  t ->
  Ssa_type.pred value ->
  then_:(t -> 's pack) ->
  else_:(t -> 's pack) ->
  's pack
(** Only the selected region executes. Both return the same signature. *)

(** {1 Untyped control flow}

    For a converter whose carried values are not known statically. The signature
    is checked at run time, and the verifier checks it again. *)

val for_dyn :
  t ->
  lo:Ssa_type.index value ->
  hi:Ssa_type.index value ->
  init:Ssa_value.t list ->
  (t -> Ssa_type.index value -> Ssa_value.t list -> Ssa_value.t list) ->
  Ssa_value.t list

val if_dyn :
  t ->
  Ssa_type.pred value ->
  then_:(t -> Ssa_value.t list) ->
  else_:(t -> Ssa_value.t list) ->
  Ssa_value.t list

val as_f64 : Ssa_value.t -> Ssa_type.f64 value
(** A typed handle for a value of that type; [Invalid_argument] otherwise. *)

val as_i64 : Ssa_value.t -> Ssa_type.i64 value
val as_index : Ssa_value.t -> Ssa_type.index value
val as_pred : Ssa_value.t -> Ssa_type.pred value
val as_local : Ssa_value.t -> Ssa_type.local value

val ordered_sum :
  t ->
  lo:Ssa_type.index value ->
  hi:Ssa_type.index value ->
  seed:Ssa_type.f64 value ->
  (t -> Ssa_type.index value -> Ssa_type.f64 value) ->
  Ssa_type.f64 value
(** The left fold of the terms the body returns, from [seed]. *)
