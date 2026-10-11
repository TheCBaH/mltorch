open Inference_compare_schema
module Codecs = Inference_compare_codecs
module H = Codecs.Payload_hasher
module C = Codec_fixtures

let sha = Inference_compare_sha256.registry
let sha_id = Inference_compare_sha256.algorithm
let pp_hasher_error = Err.Error.pp_kind H.pp_error

let hex_of r =
  match r with
  | Ok d -> Digest.Payload.hex d
  | Error e -> Format.asprintf "ERROR %a" pp_hasher_error e

let%expect_test "known answer and chunk invariance" =
  let whole = hex_of (H.digest_string sha sha_id "abc") in
  Format.printf "abc: %s@." whole;
  let chunked =
    let ( let* ) = Err.bind in
    let r =
      let* st = H.create sha sha_id in
      let* () = H.update st "xxabcxx" ~pos:2 ~len:1 in
      let* () = H.update st "abc" ~pos:1 ~len:1 in
      let* () = H.update st "abc" ~pos:2 ~len:1 in
      H.finish st
    in
    hex_of r
  in
  Format.printf "chunked equals whole: %b@." (String.equal whole chunked);
  [%expect
    {|
    abc: ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
    chunked equals whole: true |}]

let%expect_test "lifecycle" =
  let show name r =
    match r with
    | Ok _ -> Format.printf "%s: ok@." name
    | Error e -> Format.printf "%s: %a@." name H.pp_error (Err.Error.kind e)
  in
  let st = Result.get_ok (H.create sha sha_id) in
  ignore (H.update_string st "a");
  ignore (H.finish st);
  show "update after finish" (H.update_string st "b");
  show "second finish" (Result.map ignore (H.finish st));
  let st2 = Result.get_ok (H.create sha sha_id) in
  show "range outside the chunk" (H.update st2 "abc" ~pos:2 ~len:5);
  show "negative position" (H.update st2 "abc" ~pos:(-1) ~len:1);
  let missing = C.text (Id.Digest_algorithm.of_string "md5") in
  show "unregistered algorithm" (Result.map ignore (H.create sha missing));
  [%expect
    {|
    update after finish: digest state already finished
    second finish: digest state already finished
    range outside the chunk: range pos=2 len=5 outside length 3
    negative position: range pos=-1 len=1 outside length 3
    unregistered algorithm: unknown digest algorithm md5 |}]

(* A deliberately weak backend whose digest of everything is the SHA-256 of
   "abc" -- the same hex as the real backend's answer for "abc". It stands in
   for a second algorithm to show that equal hex under two algorithm ids can
   never alias. *)
module Constant = struct
  type state = unit

  let algorithm = C.text (Id.Digest_algorithm.of_string "constant")
  let create () = ()
  let update () _ ~pos:_ ~len:_ = ()

  let finish () =
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
end

let%expect_test "algorithm tags cannot alias" =
  let registry =
    match H.Registry.register sha (module Constant) with
    | Ok r -> r
    | Error _ -> failwith "register"
  in
  let real = Result.get_ok (H.digest_string registry sha_id "abc")
  and fake =
    Result.get_ok (H.digest_string registry Constant.algorithm "anything")
  in
  Format.printf "same hex: %b@."
    (String.equal (Digest.Payload.hex real) (Digest.Payload.hex fake));
  Format.printf "equal digests: %b@." (Digest.Payload.equal real fake);
  Format.printf "same content path: %b@."
    (String.equal (H.content_path real) (H.content_path fake));
  Format.printf "paths: %s | %s@." (H.content_path real) (H.content_path fake);
  (match H.Registry.register registry (module Constant) with
  | Ok _ -> Format.printf "duplicate registration accepted@."
  | Error _ -> Format.printf "duplicate registration refused@.");
  Format.printf "algorithms: %a@."
    Fmt.(list ~sep:comma Id.Digest_algorithm.pp)
    (H.Registry.algorithms registry);
  [%expect
    {|
    same hex: true
    equal digests: false
    same content path: false
    paths: sha256/ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad | constant/ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
    duplicate registration refused
    algorithms: sha256,
    constant |}]
