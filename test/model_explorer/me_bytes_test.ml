let render = function
  | Error e -> (
      let kind = Err.Error.kind e in
      match kind with
      | `Bad_length n -> Printf.sprintf "bad %.0f" n
      | `Over_limit n -> Printf.sprintf "over %Ld" n)
  | Ok n -> Printf.sprintf "ok %d" n

let check n =
  print_endline
    (render
       (Me_bytes.checked_length n
          ~max:(Int64.of_int Me_limits.Hard.max_request_json_bytes)))

let%expect_test "checked JavaScript lengths never narrow before validation" =
  List.iter check
    [
      0.;
      65_536.;
      65_537.;
      -1.;
      1.5;
      infinity;
      neg_infinity;
      nan;
      2_147_483_647.;
      2_147_483_648.;
      9_007_199_254_740_991.;
      9_007_199_254_740_992.;
    ];
  [%expect
    {|
    ok 0
    ok 65536
    over 65537
    bad -1
    bad 2
    bad inf
    bad -inf
    bad nan
    over 2147483647
    over 2147483648
    over 9007199254740991
    bad 9007199254740992
    |}]
