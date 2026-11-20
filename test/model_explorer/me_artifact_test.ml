(* [Me_artifact]: the lowering inspector's answer as the protocol carries it.

   What a round trip cannot show: that the text ceiling cuts at EXACTLY the
   ceiling (not a byte early, not a byte late) and says so, that a cut never
   lands inside a UTF-8 sequence, and that a refusal is data with a tag. The
   truncation test follows the [max_attr_chars] one in the detail suite. *)

module A = Me_artifact
module L = Me_limits.Limits

let key =
  Lowering_key.make
    (Graph_ir.Node_id.of_int 2)
    (Result.get_ok (Lowering_key.of_name "c:ssa"))

let tight n =
  Err.or_raise ~pp_error:Me_limits.pp_error
    (L.create ~max_artifact_bytes:n L.untrusted)

let show_body = function
  | A.Body.Text { language; text; truncated } ->
      Printf.printf "%s len=%d truncated=%b %S\n" language (String.length text)
        truncated text
  | A.Body.Refused t ->
      Printf.printf "refused %s [%s]\n" t.A.Tag.kind
        (String.concat "; " t.A.Tag.data)

let%expect_test "text is cut at exactly max_artifact_bytes" =
  let limits = tight 8 in
  List.iter
    (fun text -> show_body (A.text ~limits ~language:"js" text))
    [ ""; "1234567"; "12345678"; "123456789"; String.make 100 'x' ];
  [%expect
    {|
    js len=0 truncated=false ""
    js len=7 truncated=false "1234567"
    js len=8 truncated=false "12345678"
    js len=8 truncated=true "12345678"
    js len=8 truncated=true "xxxxxxxx" |}]

let%expect_test "a cut never lands inside a UTF-8 sequence" =
  (* U+00E9 is two bytes: a ceiling of 3 over "ab" ^ e-acute ^ "c" would cut
     between the two bytes of the accent, so the cut backs up to 2. *)
  let limits = tight 3 in
  show_body (A.text ~limits ~language:"js" "ab\xC3\xA9c");
  show_body (A.text ~limits:(tight 4) ~language:"js" "ab\xC3\xA9c");
  [%expect
    {|
    js len=2 truncated=true "ab"
    js len=4 truncated=true "ab\195\169" |}]

let round_trip body notes =
  let a = { A.key; body; notes } in
  match Jsont_bytesrw.encode_string A.jsont a with
  | Error e -> print_endline e
  | Ok text -> (
      print_endline text;
      match Jsont_bytesrw.decode_string A.jsont text with
      | Error e -> print_endline e
      | Ok back -> Printf.printf "round trip: %b\n" (back = a))

let%expect_test "an artifact survives the wire" =
  round_trip
    (A.Body.Text { language = "wat"; text = "(module)\n"; truncated = false })
    [ { A.Tag.kind = "numerics"; data = [ "binary32" ] } ];
  round_trip
    (A.Body.Refused { A.Tag.kind = "unsupported_op"; data = [ "lstm" ] })
    [];
  [%expect
    {|
    {"key":{"dialect":"native","node":2,"variant":"c:ssa"},"body":{"kind":"text","language":"wat","text":"(module)\n","truncated":false},"notes":[{"kind":"numerics","data":["binary32"]}]}
    round trip: true
    {"key":{"dialect":"native","node":2,"variant":"c:ssa"},"body":{"kind":"refused","refusal":{"kind":"unsupported_op","data":["lstm"]}},"notes":[]}
    round trip: true |}]

let%expect_test "a body naming no kind, or one it does not know, is refused" =
  List.iter
    (fun text ->
      match Jsont_bytesrw.decode_string A.Body.jsont text with
      | Ok _ -> print_endline "decoded"
      | Error _ -> print_endline "rejected")
    [
      {|{"kind":"text"}|};
      {|{"kind":"binary"}|};
      {|{"kind":"refused"}|};
      {|{"kind":"text","language":"js","text":"x","truncated":false}|};
    ];
  [%expect {|
    rejected
    rejected
    rejected
    decoded |}]
