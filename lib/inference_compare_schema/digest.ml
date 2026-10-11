module type S = sig
  type t

  val make : Id.Digest_algorithm.t -> hex:string -> (t, Digest_error.t) Err.t
  val of_string : string -> (t, Digest_error.t) Err.t
  val to_string : t -> string
  val algorithm : t -> Id.Digest_algorithm.t
  val hex : t -> string
  val compare : t -> t -> int
  val equal : t -> t -> bool
  val pp : Format.formatter -> t -> unit
end

let max_hex_length = 256
let is_lower_hex c = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')

let valid_hex s =
  let n = String.length s in
  n > 0 && n <= max_hex_length && n mod 2 = 0 && String.for_all is_lower_hex s

module Make () : S = struct
  type t = { algorithm : Id.Digest_algorithm.t; hex : string }

  let make algorithm ~hex =
    if valid_hex hex then Ok { algorithm; hex }
    else Err.fail ~pp_error:Digest_error.pp Digest_error.Invalid_encoding

  let of_string s =
    match String.index_opt s ':' with
    | None -> Err.fail ~pp_error:Digest_error.pp Digest_error.Missing_separator
    | Some i -> (
        let algorithm = String.sub s 0 i in
        let hex = String.sub s (i + 1) (String.length s - i - 1) in
        match Id.Digest_algorithm.of_string algorithm with
        | Error e ->
            Err.fail ~pp_error:Digest_error.pp
              (Digest_error.Invalid_algorithm (Err.Error.kind e))
        | Ok algorithm -> make algorithm ~hex)

  let algorithm d = d.algorithm
  let hex d = d.hex
  let to_string d = Id.Digest_algorithm.to_string d.algorithm ^ ":" ^ d.hex

  let compare a b =
    match Id.Digest_algorithm.compare a.algorithm b.algorithm with
    | 0 -> String.compare a.hex b.hex
    | c -> c

  let equal a b = compare a b = 0
  let pp ppf d = Fmt.string ppf (to_string d)
end

module Build = Make ()
module Configuration = Make ()
module Comparison_policy = Make ()
module Contract = Make ()
module Conversion = Make ()
module Derivation = Make ()
module Dtype_parameters = Make ()
module Engine_artifact = Make ()
module Environment = Make ()
module Extracted_graph_payload = Make ()
module Frontier_mapping = Make ()
module Graph = Make ()
module Implementation = Make ()
module Input_domain = Make ()
module Logical_value = Make ()
module Mapping = Make ()
module Metric_support = Make ()
module Output_specs = Make ()
module Payload = Make ()
module Plan = Make ()
module Policy_bindings = Make ()
module Policy_selection = Make ()
module Run_evidence = Make ()
