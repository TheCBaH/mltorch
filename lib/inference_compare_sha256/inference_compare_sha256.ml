open Inference_compare_schema
module Hasher = Inference_compare_codecs.Payload_hasher

let algorithm =
  match Id.Digest_algorithm.of_string "sha256" with
  | Ok a -> a
  | Error _ -> invalid_arg "Inference_compare_sha256.algorithm"

module Backend = struct
  type state = Pt2_sha256.t

  let algorithm = algorithm
  let create = Pt2_sha256.create
  let update st s ~pos ~len = Pt2_sha256.add_string st ~pos ~len s
  let finish st = Pt2_sha256.Digest.to_hex (Pt2_sha256.finish st)
end

let registry =
  match Hasher.Registry.register Hasher.Registry.empty (module Backend) with
  | Ok r -> r
  | Error _ -> invalid_arg "Inference_compare_sha256.registry"
