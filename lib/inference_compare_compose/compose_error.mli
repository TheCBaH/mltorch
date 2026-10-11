(** Why two mapping artifacts do not compose. Payloads name the offending node,
    tensor or value; none carries prose. *)

open Inference_compare_schema

type absent_from = First | Second

type t =
  | Ambiguous_route of { destination : Tensor_ref.t; source : Tensor_ref.t }
      (** Two different conversion step lists connect the same tensors. *)
  | Conflicting_constant of Id.Canonical_value.t
      (** A pinned constant of the second map clashes with the first's value of
          the same key. *)
  | Intermediate_graph_mismatch of {
      first_right : Graph_key.t;
      second_left : Graph_key.t;
    }
  | Invalid_identifier of string
  | Lossy_comparison of Id.Member_pair.t
      (** The first pair converts its right member before comparing, which a
          composition cannot undo. *)
  | Missing_binding of Tensor_ref.t
  | Missing_check of Id.Canonical_value.t
      (** A constant the second map pins has no right-side check there. *)
  | Missing_intermediate_member of {
      absent_from : absent_from;
      node : Node_ref.t;
    }
      (** A node of the intermediate graph that one map covers and the other
          does not mention. *)
  | Missing_route of { tensor : Tensor_ref.t; toward : Side.t }
      (** No supplied route reaches [tensor] from the [toward] side's graph, so
          that direction of the composed relation is unavailable. *)
  | Output_mismatch of { position : int }

val pp : Format.formatter -> t -> unit
val fail : t -> ('a, t) Err.t
