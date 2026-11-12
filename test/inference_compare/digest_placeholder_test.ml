(* A placeholder digest is the all-zero value under the algorithm and is a
   digest [make] accepts, so sealing can swap it for the real one. *)

open Inference_compare_schema

let%expect_test "placeholder is the zero digest under the algorithm" =
  let alg = Inference_compare_sha256.algorithm in
  let p = Digest.Plan.placeholder alg in
  print_endline (Digest.Plan.to_string p);
  Format.printf "%b@."
    (Digest.Plan.equal p
       (Err.or_raise ~pp_error:Digest_error.pp
          (Digest.Plan.make alg ~hex:(String.make 64 '0'))));
  [%expect
    {|
    sha256:0000000000000000000000000000000000000000000000000000000000000000
    true |}]
