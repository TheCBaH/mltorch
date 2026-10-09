(** The versioned, machine-readable result of replaying an artifact. A run that
    could not start (a refusal) and a run that compared and failed are distinct
    statuses; neither is [Passed]. *)

type status = Failed | Passed | Refused

type case = {
  error : string option;
      (** The case could not be run or checked (a refusal inside the case). *)
  id : string;
  inputs_digest_ok : bool;
  outputs : Compare.t list;
  outputs_digest_ok : bool;
}

type t = {
  artifact_id : string;
  atol : float;
  backend : string;  (** The route that was executed, e.g. ["native-direct"]. *)
  cases : case list;
  consumer : string;  (** The consumer revision, with its workspace state. *)
  pins : (string * string) list;
      (** Named digests: graph, contract, map, ... *)
  refusal : string option;
  rtol : float;
  status : status;
}

val schema_version : int

val status_of_cases : case list -> status
(** [Passed] only if there is at least one case and every case has both digests
    verified, no error, every output present and passing. *)

val case_passed : case -> bool
val to_string : t -> string
