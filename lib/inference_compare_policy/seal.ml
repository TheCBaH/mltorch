open Inference_compare_schema
module H = Inference_compare_codecs.Payload_hasher

type error = Hasher of H.error

let pp_error ppf (Hasher e) = H.pp_error ppf e
let ( let* ) = Err.bind
let lift r = Err.map_error (fun e -> Hasher e) r

let hex ~hashers ~algorithm preimage =
  let* d = lift (H.digest_string hashers algorithm preimage) in
  Ok (Digest.Payload.hex d)

let make_digest make algorithm hex =
  match make algorithm ~hex with
  | Ok d -> Ok d
  | Error _ ->
      (* A backend returned non-canonical hex: a registration fault. *)
      Err.fail ~pp_error (Hasher (Unknown_algorithm algorithm))

let placeholder algorithm =
  match Digest.Comparison_policy.make algorithm ~hex:(String.make 64 '0') with
  | Ok d -> d
  | Error _ -> invalid_arg "Seal.placeholder"

let selection_digest ~hashers ~algorithm selection =
  let* hex = hex ~hashers ~algorithm (Policy_selection.preimage selection) in
  make_digest Digest.Policy_selection.make algorithm hex

let seal ~hashers ~algorithm ~selection bindings =
  let* selection_digest = selection_digest ~hashers ~algorithm selection in
  let rec each acc = function
    | [] -> Ok (List.rev acc)
    | (b : Policy_binding.t) :: rest ->
        let* h =
          hex ~hashers ~algorithm (Comparison_policy.preimage b.policy)
        in
        let* policy_digest =
          make_digest Digest.Comparison_policy.make algorithm h
        in
        each ({ b with policy_digest } :: acc) rest
  in
  let* bindings = each [] bindings in
  let unsealed : Policy_bindings.t =
    {
      selection_digest;
      bindings_digest =
        (match
           Digest.Policy_bindings.make algorithm ~hex:(String.make 64 '0')
         with
        | Ok d -> d
        | Error _ -> invalid_arg "Seal.seal");
      bindings;
    }
  in
  let* h = hex ~hashers ~algorithm (Policy_bindings.preimage unsealed) in
  let* bindings_digest = make_digest Digest.Policy_bindings.make algorithm h in
  Ok (Policy_bindings.canonicalize { unsealed with bindings_digest })
