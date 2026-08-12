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

let%expect_test "error_kind + err_result" =
  let err : (_, [> `Msg of string ]) Err.t = Err.fail (`Msg "bad input") in
  Format.printf "%a@."
    (Core.Pretty.error_kind (fun ppf (`Msg msg) -> Fmt.string ppf msg))
    (match err with Ok _ -> assert false | Error e -> e);
  Format.printf "%a@."
    (Core.Pretty.err_result ~ok:Fmt.int ~error:(fun ppf (`Msg msg) ->
         Fmt.string ppf msg))
    err;
  [%expect {|
    bad input
    bad input |}]

let contains s sub =
  let n = String.length s and m = String.length sub in
  let rec go i = i + m <= n && (String.sub s i m = sub || go (i + 1)) in
  go 0

(* [or_raise] raises [Err.Exn.E], NOT [Failure]. The exception carries the whole
   wrapper and registers a [Printexc] printer, so a handler holding nothing but
   [Printexc.to_string] still gets payload and provenance instead of a bare
   message -- which is the point, and also why any boundary rendering into a
   wire response has to match [Err.Exn.E] and print the payload itself.

   Asserts the SHAPE and that the payload survives, never the rendered text: it
   embeds a backtrace, and a golden holding one shifts on every build. *)
let%expect_test "or_raise raises a structured exception carrying the payload" =
  let pp_msg ppf (`Msg msg) = Fmt.string ppf msg in
  print_endline
    (Err.or_raise ~pp_error:pp_msg
       (Err.return "ok" : (_, [> `Msg of string ]) Err.t));
  (match Err.or_raise ~pp_error:pp_msg (Err.fail (`Msg "bad input")) with
  | (_ : string) -> print_endline "unexpectedly returned"
  | exception Err.Exn.E packed ->
      Printf.printf "structured=true payload=%b\n"
        (contains (Core.Pretty.to_string Err.Exn.pp packed) "bad input")
  | exception Failure _ -> print_endline "raised a bare Failure");
  [%expect {|
    ok
    structured=true payload=true |}]

let%expect_test "capture_to_string" =
  let rendered =
    Core.Pretty.capture_to_string (fun ppf -> Fmt.pf ppf "@[<v>alpha@,beta@]")
  in
  print_string rendered;
  [%expect {|
    alpha
    beta |}]
