let pp_ints = Fmt.brackets (Fmt.list ~sep:Fmt.comma Fmt.int)

let%expect_test "to_string + option_or + result" =
  print_endline (Core.Pretty.to_string pp_ints [ 1; 2; 3 ]);
  Format.printf "%a@." (Core.Pretty.option_or ~none:"none" Fmt.int) (Some 7);
  Format.printf "%a@." (Core.Pretty.option_or ~none:"none" Fmt.int) None;
  Format.printf "%a@."
    (Core.Pretty.result ~ok:Fmt.int ~error:Fmt.string)
    (Ok 42);
  Format.printf "%a@."
    (Core.Pretty.result ~ok:Fmt.int ~error:Fmt.string)
    (Error "bad");
  [%expect {|
    [1, 2, 3]
    7
    none
    42
    bad |}]

let%expect_test "error_kind + core_result" =
  let err : (_, [> `Msg of string ]) Core.result =
    Core.fail (`Msg "bad input")
  in
  Format.printf "%a@."
    (Core.Pretty.error_kind (fun ppf (`Msg msg) -> Fmt.string ppf msg))
    (match err with Ok _ -> assert false | Error e -> e);
  Format.printf "%a@."
    (Core.Pretty.core_result ~ok:Fmt.int ~error:(fun ppf (`Msg msg) ->
         Fmt.string ppf msg))
    err;
  [%expect {|
    bad input
    bad input |}]

let%expect_test "capture_to_string" =
  let rendered =
    Core.Pretty.capture_to_string (fun ppf -> Fmt.pf ppf "@[<v>alpha@,beta@]")
  in
  print_string rendered;
  [%expect {|
    alpha
    beta |}]
