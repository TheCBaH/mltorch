(* Ranked class predictions from a pure-engine inference result. The dual of
   [Interp.top_predictions], which does the same job through ATen's [_softmax]
   and [topk]; this one is reachable from js_of_ocaml. *)

(** {1 Error payloads}

    Each is data the caller can branch on, per the payloads-carry-data rule. *)

module Too_few_classes : sig
  type t = { classes : int; wanted : int }
end

module Non_finite : sig
  type t = { index : int; value : float }
  (** The FIRST offending class in index order, so the report is stable. *)
end

type error =
  [ `Invalid_k of int  (** [k <= 0]; caught before the [classes >= k] test *)
  | `Non_finite_logit of Non_finite.t
  | `Not_class_logits of Vec6.shape
    (** Some non-class extent <> 1. PT2 sizes right-align into [Vec6], so a
        [[1; classes]] logits tensor is [~w:1 ~c:classes] and a [[2; classes]]
        batch is [~w:2 ~c:classes] — the rejected extent is [W], not [N]. *)
  | `Output_count of int  (** outputs <> 1 *)
  | `Too_few_classes of Too_few_classes.t ]

val pp_error : Format.formatter -> [< error ] -> unit

val top_predictions :
  Tensor.packed list -> int -> ((int * float) list, error) Err.t
(** [top_predictions outputs k] is the [k] highest-scoring
    [(class index, probability)] pairs of the single output tensor in [outputs],
    descending. Requires [k >= 1].

    {b Ranking is by raw logit, not by probability.} Softmax is monotone, so the
    two orders agree mathematically — but a finite probability can underflow to
    [0.], which would collapse distinct logits into a tie the index tie-break
    then orders wrongly. Ties in the logits themselves break by ascending class
    index, so the result is deterministic on both backends.

    {b The softmax denominator spans every class}, not the selected [k], so the
    reported probabilities match what [Interp.top_predictions] reports for the
    same logits and do not sum to 1 over the selection. *)
