(** The producer's call contract for one artifact: the keyword inputs and the
    ordered tuple of outputs, with dtypes, shapes and the tolerance each output
    is compared under. Only a plain functional tensor call is accepted --
    positional arguments, state mutation or a non-tuple result are refused here,
    not discovered during replay. *)

module Dtype = Pt2_checkpoint_map.Dtype

module Tensor_spec : sig
  type t = { dtype : Dtype.t; name : string; shape : int64 list }
end

type t = {
  artifact_id : string;
  atol : float;
  dynamic : bool;  (** The contract carries dynamic-shape constraints. *)
  graph_sha256 : Pt2_sha256.Digest.t;
  inputs : Tensor_spec.t list;  (** In call (keyword) order. *)
  outputs : Tensor_spec.t list;  (** In tuple order. *)
  rtol : float;
  verified_cases : int;
}

val of_string : string -> (t, [> Fault.error ]) Err.t
(** Requires: no positional arguments, a [tensor_tuple] result, no mutations,
    keyword order equal to the input order, unique names. *)
