(* [Me_limits.Diagnostic]: the closed code vocabulary, and the two bounds that
   make [create] total (.ai/model_explorer_design.md).

   Under node as well as natively, because both bounds are byte counts and one
   of them is applied to a buffer whose length is an [int].

   The interesting inputs here are the ones a caller cannot produce on purpose:
   [of_exn] admits whatever bytes an exception message happens to carry, which
   need not be valid UTF-8, and the wire format is JSON. So the fixtures are
   deliberately ill-formed rather than merely long. *)

(* Jsont styles its decode errors with ANSI escapes unless the ambient
   environment says otherwise, and cram/expect goldens below quote that text.
   Pinned here so a bare [dune runtest] agrees with [make runtest], which sets
   NO_COLOR itself -- a golden that depends on the caller's environment is a
   golden that passes for the wrong reason. *)
let () = Jsont.Error.disable_ansi_styler ()

module D = Me_limits.Diagnostic

let limits = Me_limits.Limits.untrusted
let pp_graph = Core.Pretty.option_or ~none:"absent" (Fmt.fmt "%S")

let show d =
  Format.printf "%a@." D.pp d;
  Format.printf "  message %d bytes, graph %a, truncated %b@."
    (String.length d.D.message)
    pp_graph d.D.graph d.D.truncated

(* --- the vocabulary --- *)

let%expect_test "the vocabulary is closed and round-trips" =
  List.iter
    (fun c ->
      let s = D.Code.to_string c in
      match D.Code.of_string s with
      | Some c' when c' = c -> print_string (s ^ " ")
      | _ -> Printf.printf "\nBROKEN %s\n" s)
    D.Code.all;
  Printf.printf "\n%d codes\n" (List.length D.Code.all);
  [%expect
    {|
    over_limit malformed_request invalid_limits invalid_source malformed_response request_in_flight inconsistent_mount settlement_mismatch buffer_mismatch not_an_array_buffer stale_epoch unsupported_detail_key key_disagrees_with_ids unsupported_operator unsupported_input unsupported_graph_shape outside_dialect_domain requires_payloads prerequisite_unavailable not_implemented internal
    21 codes
    |}]

let%expect_test "an unknown tag is a named failure, not a silent drop" =
  let pp_code =
    Core.Pretty.option_or ~none:"unknown" (Fmt.of_to_string D.Code.to_string)
  in
  let show s = Format.printf "%-24s %a@." s pp_code (D.Code.of_string s) in
  show "over_limit";
  show "Over_limit";
  show "some_future_code";
  show "";
  [%expect
    {|
    over_limit               over_limit
    Over_limit               unknown
    some_future_code         unknown
                             unknown
    |}]

(* --- the message bound --- *)

let%expect_test "a long message is cut, and says so" =
  show (D.create ~limits D.Code.Over_limit (String.make 32 'x'));
  show (D.create ~limits D.Code.Over_limit (String.make 1024 'x'));
  [%expect
    {|
    over_limit: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
      message 32 bytes, graph absent, truncated false
    over_limit: xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx (truncated)
      message 512 bytes, graph absent, truncated true
    |}]

let%expect_test "bytes that are not UTF-8 at all are replaced, not passed on" =
  (* [of_exn]'s domain, and the reason replacement is not optional: see the
     jsont-behaviour case at the bottom of this file. *)
  let raw = "before \xff\xfe after" in
  show (D.create ~limits D.Code.Internal raw);
  (* Control bytes, including the ones that are valid UTF-8. *)
  show (D.create ~limits D.Code.Internal "a\tb\nc\x00d\x7fe");
  (* Valid multi-byte input survives unchanged. *)
  show (D.create ~limits D.Code.Internal "naïve — 世界");
  [%expect
    {|
    internal: before �� after
      message 19 bytes, graph absent, truncated false
    internal: a�b�c�d�e
      message 17 bytes, graph absent, truncated false
    internal: naïve — 世界
      message 17 bytes, graph absent, truncated false |}]

let%expect_test "replacement widens, so the cut is measured on what is written"
    =
  (* One stray byte becomes three. A ceiling tested against the bytes READ
     would admit three times the bytes it promised. *)
  let tight =
    Err.or_raise ~pp_error:Me_limits.pp_error
      (Me_limits.Limits.create ~max_diagnostic_bytes:16 limits)
  in
  let d = D.create ~limits:tight D.Code.Internal (String.make 32 '\xff') in
  Printf.printf "%d bytes, truncated %b\n"
    (String.length d.D.message)
    d.D.truncated;
  (* And the result is still whole scalar values: 16 bytes cannot hold six
     three-byte replacements, so it stops at five. *)
  Printf.printf "%S\n" d.D.message;
  [%expect
    {|
    15 bytes, truncated true
    "\239\191\189\239\191\189\239\191\189\239\191\189\239\191\189" |}]

(* --- the graph id --- *)

let%expect_test "an id that does not fit is dropped, never truncated" =
  let tight =
    Err.or_raise ~pp_error:Me_limits.pp_error
      (Me_limits.Limits.create ~max_id_bytes:8 limits)
  in
  show (D.create ~limits:tight ~graph:"g/native" D.Code.Over_limit "m");
  show (D.create ~limits:tight ~graph:"g/native/0001" D.Code.Over_limit "m");
  show (D.create ~limits:tight ~graph:"g/\xffve" D.Code.Over_limit "m");
  show (D.create ~limits:tight ~graph:"g\x00ve" D.Code.Over_limit "m");
  [%expect
    {|
    over_limit: m [g/native]
      message 1 bytes, graph "g/native", truncated false
    over_limit: m (truncated)
      message 1 bytes, graph absent, truncated true
    over_limit: m (truncated)
      message 1 bytes, graph absent, truncated true
    over_limit: m (truncated)
      message 1 bytes, graph absent, truncated true |}]

(* --- the exception bridge --- *)

let%expect_test "of_exn is the one bridge, and it is lossy on purpose" =
  show (D.of_exn ~limits (Failure "something went wrong"));
  show (D.of_exn ~limits Not_found);
  [%expect
    {|
    internal: Failure("something went wrong")
      message 31 bytes, graph absent, truncated false
    internal: Not_found
      message 9 bytes, graph absent, truncated false |}]

(* [Err.Exn.E] is the one exception this bridge must NOT hand to
   [Printexc.to_string]. [Err] registers a printer, so the default path renders
   payload + detection stack + semantic trace -- and this value is a wire
   response, so that would ship frame-by-frame provenance of the server's own
   source to the browser. The sanitiser would truncate it, not remove it.

   The message is asserted structurally rather than as a golden: it must CONTAIN
   the payload and must NOT contain the provenance markers, and the frames
   themselves shift per build and vanish entirely under node (this suite runs
   both ways). Byte counts are out for the same reason. *)
let%expect_test "of_exn renders an Err payload without its provenance" =
  let pp_msg ppf (`Msg m) = Fmt.string ppf m in
  let raised =
    match Err.or_raise ~pp_error:pp_msg (Err.fail (`Msg "lowering failed")) with
    | (_ : int) -> Failure "or_raise did not raise"
    | exception e -> e
  in
  let d = D.of_exn ~limits raised in
  let contains sub =
    let s = d.D.message in
    let n = String.length s and m = String.length sub in
    let rec go i = i + m <= n && (String.sub s i m = sub || go (i + 1)) in
    go 0
  in
  Printf.printf "payload=%b detected_at=%b trace=%b source_named=%b\n"
    (contains "lowering failed")
    (contains "detected at") (contains "trace:")
    (contains "me_diagnostic_test.ml" || contains "err.ml");
  [%expect {| payload=true detected_at=false trace=false source_named=false |}]

(* --- the wire --- *)

let%expect_test "round trip" =
  (* A named crossing out of jsont's [(_, string) result], per
     .ai/printer_conventions.md — not a hand-rolled print. Every value here is
     one [create] built, so an encoder that refuses it is a defect rather than
     an outcome, and [Err.or_raise ~pp_error:] cannot serve because this result carries
     no [Err.Error.t]. *)
  let encode d =
    match Jsont_bytesrw.encode_string D.jsont d with
    | Ok s -> s
    | Error e -> failwith ("encode: " ^ e)
  in
  let pp_reencoded s =
    Core.Pretty.result
      ~ok:(fun fmt d' -> Fmt.pf fmt "re-encoded equal: %b" (encode d' = s))
      ~error:(Fmt.fmt "decode failed: %s")
  in
  let roundtrip d =
    let s = encode d in
    Format.printf "%s@.  %a@." s (pp_reencoded s)
      (Jsont_bytesrw.decode_string D.jsont s)
  in
  roundtrip (D.create ~limits D.Code.Over_limit "too big");
  roundtrip
    (D.create ~limits ~graph:"g/native/0001" D.Code.Unsupported_operator "no");
  (* [Printexc.to_string] escapes as it renders, so an exception carrying raw
     bytes reaches the sanitiser as printable text and proves nothing about it.
     The case that matters is the one that hands [create] the bytes directly —
     which is what a message assembled from any other source can do. *)
  roundtrip (D.of_exn ~limits (Failure "raw \xff bytes"));
  roundtrip (D.create ~limits D.Code.Internal "raw \xff\xfe bytes");
  [%expect
    {|
    {"code":"over_limit","message":"too big","truncated":false}
      re-encoded equal: true
    {"code":"unsupported_operator","message":"no","graph":"g/native/0001","truncated":false}
      re-encoded equal: true
    {"code":"internal","message":"Failure(\"raw \\255 bytes\")","truncated":false}
      re-encoded equal: true
    {"code":"internal","message":"raw �� bytes","truncated":false}
      re-encoded equal: true |}]

let%expect_test "an unknown code on the wire is refused" =
  let pp_decoded =
    Core.Pretty.result ~ok:D.pp ~error:(Fmt.fmt "rejected: %s")
  in
  let show s =
    Format.printf "%a@." pp_decoded (Jsont_bytesrw.decode_string D.jsont s)
  in
  show {|{"code":"over_limit","message":"m","truncated":false}|};
  show {|{"code":"some_future_code","message":"m","truncated":false}|};
  (* [truncated] absent decodes as false rather than failing: a producer that
     never cut anything need not say so. *)
  show {|{"code":"internal","message":"m"}|};
  [%expect
    {|
    over_limit: m
    rejected: Unexpected diagnostic_code enum string value: some_future_code.
    Must be over_limit, malformed_request, invalid_limits, invalid_source,
    malformed_response, request_in_flight, inconsistent_mount,
    settlement_mismatch, buffer_mismatch, not_an_array_buffer, stale_epoch,
    unsupported_detail_key, key_disagrees_with_ids, unsupported_operator,
    unsupported_input, unsupported_graph_shape, outside_dialect_domain,
    requires_payloads, prerequisite_unavailable, not_implemented or internal.
    File "-":
    File "-": in member code of
    File "-", line 1, characters 0-27: diagnostic object
    internal: m |}]

(* --- the dependency behaviour the sanitiser exists for --- *)

let%expect_test "jsont emits invalid UTF-8 rather than refusing it" =
  (* The premise behind [create]'s replacement step, pinned rather than
     asserted. If [Jsont_bytesrw] REFUSED these bytes, sanitising would be
     merely tidy — the encoder would catch anything that slipped through. It
     does not: it passes them along, so the output is not JSON, no OCaml code
     on the path sees a problem, and the first thing to notice is the browser's
     parser. That turns a diagnostic about some other condition into a
     malformed response, which is the one failure a diagnostic must not cause.

     A jsont upgrade that starts validating makes this case fail, which is the
     point: the justification above would then need rewriting, not just the
     golden. *)
  let raw = "raw \xff\xfe bytes" in
  Format.printf "direct: %a@."
    (Core.Pretty.result ~ok:(Fmt.fmt "%S") ~error:(Fmt.fmt "refused: %s"))
    (Jsont_bytesrw.encode_string Jsont.string raw);
  (* Through [create], the same bytes are clean before they ever reach it. *)
  Format.printf "sanitised: %a@."
    (Core.Pretty.result ~ok:(Fmt.fmt "%S") ~error:(Fmt.fmt "refused: %s"))
    (Jsont_bytesrw.encode_string Jsont.string
       (D.create ~limits D.Code.Internal raw).D.message);
  [%expect
    {|
    direct: "\"raw \255\254 bytes\""
    sanitised: "\"raw \239\191\189\239\191\189 bytes\"" |}]
