open Transformers_metadata.Json_util
open World

let%expect_test "equal cache shapes never authorize different model weights" =
  let model =
    json
      {|{"model_id":"toy","weights":{"repo":"o/r","revision":"abc","config_sha256":"def"}}|}
  in
  let other field value =
    if field = "model_id" then replace field (J.string value) model
    else
      replace "weights"
        (replace field (J.string value) (unwrap (member "weights" model)))
        model
  in
  show (T.Chaining.contracts model model);
  List.iter
    (fun (field, value) ->
      match Err.payload (T.Chaining.contracts model (other field value)) with
      | Ok () -> print_endline "accepted"
      | Error _ -> print_endline "refused")
    [
      ("model_id", "other");
      ("repo", "other");
      ("revision", "other");
      ("config_sha256", "other");
    ];
  [%expect {|
    ok
    refused
    refused
    refused
    refused |}]
