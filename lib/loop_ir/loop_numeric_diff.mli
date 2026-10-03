(** How far a binary32-kernel result is from the binary64 reference, as numbers
    a tolerance can be calibrated against: absolute, relative and
    scale-normalized error, ULP distance, and every nonfinite cell accounted
    for. Sampled agreement, never a proof.

    A finite pair passes when
    [|actual - reference| <= atol + rtol * |reference|]. A nonfinite cell passes
    only when it is the same kind in both (NaN with NaN, an infinity with the
    same infinity): subtracting nonfinite values and letting a NaN compare false
    would pass it by accident. *)

type t = {
  cells : int;
  failing : int;
      (** cells outside the tolerance, nonfinite mismatches included *)
  nonfinite_actual : int;
  nonfinite_reference : int;
  nonfinite_mismatch : int;
      (** cells where exactly one side, or a different kind, is nonfinite *)
  max_abs : float;  (** over finite pairs *)
  max_rel : float;  (** over finite pairs with a nonzero reference *)
  max_normalized : float;
      (** [max_abs] over the largest finite [|reference|]: error against the
          scale of the output, which stays meaningful near zero *)
  ulp_buckets : (string * int) list;
      (** finite pairs by binary32 distance: [0], [1], [2-4], [5-16], [17-256],
          [>256] *)
}

val compare :
  atol:float ->
  rtol:float ->
  actual:Tensor.packed ->
  reference:Tensor.packed ->
  t
(** Both tensors must have the same shape. *)

val outputs :
  atol:float ->
  rtol:float ->
  reference:(Tensor_id.t -> Tensor.packed) ->
  (Tensor_id.t * Tensor.packed) list ->
  (Tensor_id.t * t) list
(** {!compare} of every graph output against its reference, by id. *)

val total : (Tensor_id.t * t) list -> t option
(** The outputs' figures accumulated; [None] for no outputs. *)

val merge : t -> t -> t
(** Accumulates two outputs' figures (the empty [compare] of zero cells is the
    unit). *)

val pp : Format.formatter -> t -> unit
