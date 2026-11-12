open Inference_compare_schema
module Hasher = Inference_compare_codecs.Payload_hasher

let algorithm = Id.Digest_algorithm.of_literal "sha256"

module Backend = struct
  type state = Pt2_sha256.t

  let algorithm = algorithm
  let create = Pt2_sha256.create
  let update st s ~pos ~len = Pt2_sha256.add_string st ~pos ~len s
  let finish st = Pt2_sha256.Digest.to_hex (Pt2_sha256.finish st)
end

let registry =
  Err.or_raise
    ~pp_error:(fun ppf (`Duplicate_algorithm a) ->
      Fmt.pf ppf "duplicate digest algorithm %a" Id.Digest_algorithm.pp a)
    (Hasher.Registry.register Hasher.Registry.empty (module Backend))
