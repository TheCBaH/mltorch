(* [Err_host.parse]: the host's trace knob, exercised through a lookup function
   rather than the real environment so the cases are independent of each other
   and of whatever the test runner was started with. *)

let show env =
  let lookup name = List.assoc_opt name env in
  match Err_host.parse lookup with
  | Ok None -> print_endline "unset (policy left alone)"
  | Ok (Some config) -> Format.printf "%a@." Err.Config.pp config
  | Error e -> Format.printf "error: %a@." Err_host.pp_error (Err.Error.kind e)

(* The distinction the whole design turns on: no variable set means DO NOT
   TOUCH the policy, which is not the same as installing Err's defaults. A host
   that already chose [fast] must keep it. *)
let%expect_test "no variable set leaves the policy alone" =
  show [];
  [%expect {| unset (policy left alone) |}]

let%expect_test "one variable set fills the rest from Err's defaults" =
  show [ ("MLTORCH_ERROR_TRACE", "off") ];
  show [ ("MLTORCH_ERROR_BACKTRACE", "events") ];
  [%expect
    {|
    { actions = off; backtrace = origin; max_events = 32; max_frames = 32;
      max_external_bytes = 16384 }
    { actions = map, filter, catch, raise, import, export; backtrace = events;
      max_events = 32; max_frames = 32; max_external_bytes = 16384 }
    |}]

let%expect_test "the three trace presets, and an explicit action list" =
  List.iter
    (fun v -> show [ ("MLTORCH_ERROR_TRACE", v) ])
    [ "off"; "boundaries"; "all"; "map,catch" ];
  [%expect
    {|
    { actions = off; backtrace = origin; max_events = 32; max_frames = 32;
      max_external_bytes = 16384 }
    { actions = map, filter, catch, raise, import, export; backtrace = origin;
      max_events = 32; max_frames = 32; max_external_bytes = 16384 }
    { actions = detect, map, filter, catch, raise, import, export;
      backtrace = origin; max_events = 32; max_frames = 32;
      max_external_bytes = 16384 }
    { actions = map, catch; backtrace = origin; max_events = 32; max_frames = 32;
      max_external_bytes = 16384 }
    |}]

let%expect_test "the limits are read, and bounded" =
  show
    [
      ("MLTORCH_ERROR_MAX_EVENTS", "4");
      ("MLTORCH_ERROR_MAX_FRAMES", "8");
      ("MLTORCH_ERROR_MAX_EXTERNAL_BYTES", "256");
    ];
  [%expect
    {|
    { actions = map, filter, catch, raise, import, export; backtrace = origin;
      max_events = 4; max_frames = 8; max_external_bytes = 256 }
    |}]

(* Every rejection is a NAMED payload rather than a string, so the CLI can
   report which setting was wrong without re-deriving it from prose. A bad
   setting must be an error and not a silent fallback: falling back would leave
   the operator debugging under a policy they did not ask for, which is exactly
   what this knob exists to prevent. *)
let%expect_test "bad settings are rejected, one message per kind" =
  show [ ("MLTORCH_ERROR_TRACE", "bogus") ];
  show [ ("MLTORCH_ERROR_BACKTRACE", "nope") ];
  show [ ("MLTORCH_ERROR_MAX_EVENTS", "twelve") ];
  show [ ("MLTORCH_ERROR_MAX_FRAMES", "-4") ];
  (* Bounded so a 32-bit runtime can hold it — the repo-wide rule in CLAUDE.md,
     enforced by Err itself rather than by each host. *)
  show [ ("MLTORCH_ERROR_MAX_EXTERNAL_BYTES", "2147483647") ];
  [%expect
    {|
    error: unknown trace action "bogus"
    error: unknown backtrace mode "nope" (expected off, never, origin, or events)
    error: max_events must be an integer (got "twelve")
    error: max_frames must be non-negative (got -4)
    error: max_external_bytes must fit on a 32-bit OCaml runtime (got 2147483647)
    |}]
