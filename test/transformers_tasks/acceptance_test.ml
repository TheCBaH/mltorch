open World
open Transformers_metadata.Json_util

let%expect_test "partial task cases and full-tensor failures cannot be promoted"
    =
  let expected = get (T.Input.f32 [ 1L ] [| 1. |]) in
  let report value =
    let actual = get (T.Input.f32 [ 1L ] [| value |]) in
    let check =
      F.Compare.tensor ~atol:0. ~rtol:0. ~name:"logits" ~expected ~actual
    in
    get (T.Acceptance.report "fixture" "case" ~atol:0. ~rtol:0. [ check ])
  in
  let compared value =
    obj
      [
        ("status", J.string "compared");
        ("adapter", report 1.);
        ("model", report value);
      ]
  in
  let show cases =
    Printf.printf "accepted=%b\n"
      (T.Acceptance.passed (obj [ ("cases", J.list cases) ]))
  in
  show [ compared 1. ];
  show [ compared 1.0001 ];
  show
    [
      compared 1.;
      obj
        [
          ("status", J.string "refused");
          ("error", J.string "unsupported recipe");
        ];
    ];
  show [];
  let empty = get (T.Acceptance.report "fixture" "case" ~atol:0. ~rtol:0. []) in
  show
    [
      obj
        [
          ("status", J.string "compared");
          ("adapter", empty);
          ("model", J.null ());
        ];
    ];
  [%expect
    {|
    accepted=true
    accepted=false
    accepted=false
    accepted=false
    accepted=false |}]
