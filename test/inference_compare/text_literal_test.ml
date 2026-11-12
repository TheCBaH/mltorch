(* [of_literal] is for text fixed in the source: it accepts what [of_string]
   accepts and otherwise raises naming the role and the rejected text. *)

open Inference_compare_schema

let%expect_test "of_literal agrees with of_string and names role and value" =
  print_endline (Id.Engine.to_string (Id.Engine.of_literal "native"));
  (try ignore (Id.Engine.of_literal " native")
   with Invalid_argument m -> print_endline m);
  (try ignore (Id.Dialect.of_literal "")
   with Invalid_argument m -> print_endline m);
  [%expect
    {|
    native
    engine.of_literal " native": invalid character
    dialect.of_literal "": empty |}]
