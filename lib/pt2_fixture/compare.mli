(** Compare an actual tensor with its reference, element by element, in logical
    coordinates. Both arrive as {!Logical.t}, so no layout is in play.

    Integer and Boolean elements must be equal. Floating elements follow
    [torch.testing.assert_close] with [equal_nan = false], the producer's own
    rule: an element is close when it is equal to the reference (infinities and
    signed zeros included) or when
    [|actual - reference| <= atol + rtol * |reference|]; a NaN on either side
    never matches. For binary32 the three arithmetic steps are rounded to
    binary32 as torch does, with [atol] and [rtol] first rounded to binary32.
    Every element is checked; nothing is sampled. *)

module Dtype = Pt2_checkpoint_map.Dtype

module Diff : sig
  type t = {
    actual : string;
    expected : string;
    index : int64 list;  (** Logical coordinates. *)
  }
end

type verdict =
  | Dtype_differs of { actual : Dtype.t; expected : Dtype.t }
  | Pass
  | Shape_differs of { actual : int64 list; expected : int64 list }
  | Unsupported_dtype of Dtype.t
  | Values_differ

type t = {
  elements : int64;
  first : Diff.t list;  (** The first few mismatches, in row-major order. *)
  max_abs_error : float;
      (** Over elements that are both finite; [0.] when none are. *)
  max_rel_error : float;  (** [|a - r| / |r|] over finite, nonzero [r]. *)
  mismatches : int64;
  name : string;
  verdict : verdict;
}

val tensor :
  atol:float ->
  rtol:float ->
  name:string ->
  expected:Logical.t ->
  actual:Logical.t ->
  t

val passed : t -> bool
val max_reported : int
