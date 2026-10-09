(** [cases.json]: the published reference cases of an artifact, each naming its
    input and output tensors in contract order and carrying the content digest
    of each side. *)

module Case : sig
  type t = {
    id : string;
    inputs : string list;
    inputs_sha256 : Pt2_sha256.Digest.t;
    outputs : string list;
    outputs_sha256 : Pt2_sha256.Digest.t;
  }
end

type t = {
  artifact_id : string;
  atol : float;
  cases : Case.t list;
  rtol : float;
}

val of_string : string -> (t, [> Fault.error ]) Err.t

val check : Contract.t -> t -> (unit, [> Fault.error ]) Err.t
(** Same artifact, as many cases as the contract verified, ids [case-00],
    [case-01], ..., every case naming exactly the contract's inputs and outputs
    in the contract's order, and the contract's tolerances. *)
