open Inference_compare_schema
open Util
open Components

let compose ~hashers ~algorithm (first : Mapping_artifact.t)
    (second : Mapping_artifact.t) : (Mapping_artifact.t, E.t) Err.t =
  let ctx = { hashers; algorithm } in
  let* () =
    if Graph_key.equal first.right_graph second.left_graph then Ok ()
    else
      fail
        (E.Intermediate_graph_mismatch
           { first_right = first.right_graph; second_left = second.left_graph })
  in
  let* comps = components first second in
  let* units = Units.build ctx ~first ~second comps in
  let* relations, fwd, rev = Relations.build ctx ~first ~second in
  let* relations, matched_units =
    Pairs.build ctx ~first ~second ~units ~relations
  in
  let* dispositions = Dispositions.build ~first ~second ~comps ~units in
  let* frontier = Frontier.build ctx ~first ~second ~units ~fwd ~rev in
  let derivation m =
    match
      Codecs.Payload_hasher.digest_string hashers algorithm
        (Mapping_artifact.preimage m)
    with
    | Error _ -> fail (E.Invalid_identifier "digest backend unavailable")
    | Ok d -> (
        match Digest.Derivation.make algorithm ~hex:(Digest.Payload.hex d) with
        | Ok v -> Ok v
        | Error _ -> fail (E.Invalid_identifier "derivation digest"))
  in
  let* d1 = derivation first in
  let* d2 = derivation second in
  Ok
    {
      Mapping_artifact.schema_version = first.schema_version;
      left_graph = first.left_graph;
      right_graph = second.right_graph;
      units = matched_units;
      tensor_relations = relations;
      dispositions;
      frontier;
      derivation_digest =
        List.sort_uniq Digest.Derivation.compare
          (first.derivation_digest @ second.derivation_digest @ [ d1; d2 ]);
    }
