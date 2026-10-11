(** Persisted digests: an algorithm identifier plus canonically encoded bytes.
    Each domain is its own type, so a graph digest cannot stand where a payload
    digest is required. Equality includes the algorithm: digests under different
    algorithms never compare equal and never alias a content-addressed path. *)

module type S = sig
  type t

  val make : Id.Digest_algorithm.t -> hex:string -> (t, Digest_error.t) Err.t
  (** The encoding must be nonempty, even-length lowercase hexadecimal. *)

  val of_string : string -> (t, Digest_error.t) Err.t
  (** Parses [algorithm:hex]. *)

  val to_string : t -> string
  val algorithm : t -> Id.Digest_algorithm.t
  val hex : t -> string
  val compare : t -> t -> int
  val equal : t -> t -> bool
  val pp : Format.formatter -> t -> unit
end

module Build : S
module Configuration : S
module Comparison_policy : S
module Contract : S
module Conversion : S
module Derivation : S
module Dtype_parameters : S
module Engine_artifact : S
module Environment : S
module Extracted_graph_payload : S
module Frontier_mapping : S
module Graph : S
module Implementation : S
module Input_domain : S
module Logical_value : S
module Mapping : S
module Metric_support : S
module Output_specs : S
module Payload : S
module Plan : S
module Policy_bindings : S
module Policy_selection : S
module Run_evidence : S
