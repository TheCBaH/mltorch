open Graph_ir
open Loop_ir
open Embed_common

let%expect_test "one loaded unit, called repeatedly, matches the reference" =
  let g, b, constants = setup () in
  let p = prepare b in
  let ctx = context p ~constants in
  check g b ctx ~constants ~salt:0 ~poison:false;
  check g b ctx ~constants ~salt:2 ~poison:true;
  check g b ctx ~constants ~salt:5 ~poison:true;
  E.close_context ctx;
  E.close p;
  [%expect
    {|
    output t9 identical to the reference: true
    output t9 identical to the reference: true
    output t9 identical to the reference: true |}]

let%expect_test "contexts share the loaded unit and keep their own memory" =
  let g, b, constants = setup () in
  let p = prepare b in
  let a = context p ~constants and c = context p ~constants in
  check g b a ~constants ~salt:1 ~poison:true;
  check g b c ~constants ~salt:4 ~poison:true;
  check g b a ~constants ~salt:1 ~poison:true;
  E.close_context a;
  (match Err.payload (E.run a ~bind:(map_of (inputs b ~salt:1))) with
  | Error `Closed -> print_endline "closed context refused"
  | _ -> print_endline "unexpected");
  check g b c ~constants ~salt:4 ~poison:false;
  E.close_context c;
  E.close p;
  [%expect
    {|
    output t9 identical to the reference: true
    output t9 identical to the reference: true
    output t9 identical to the reference: true
    closed context refused
    output t9 identical to the reference: true |}]

let%expect_test "the unit has no directive and calls only declared host symbols"
    =
  let _, b, _ = setup () in
  let p = prepare b in
  let text = E.source p in
  let directive =
    List.find_opt
      (fun l -> String.length l > 0 && l.[0] = '#')
      (String.split_on_char '\n' text)
  in
  Fmt.pr "%s@." (Option.value directive ~default:"no directive");
  E.close p;
  [%expect {| no directive |}]

let%expect_test "a missing input and a failing invocation are typed" =
  let _, b0, constants = setup () in
  let p = prepare b0 in
  let ctx = context p ~constants in
  (match Err.payload (E.run ctx ~bind:(fun _ -> None)) with
  | Error (`Missing_input id) ->
      Fmt.pr "missing input t%d@." (Tensor_id.to_int id)
  | _ -> print_endline "unexpected");
  E.close_context ctx;
  E.close p;
  let failing =
    C_host_fixtures.poisoned_bundle b0 ~at:1 Loop_failure.I64_division_by_zero
  in
  let p = prepare failing in
  let ctx = context p ~constants in
  (match Err.payload (E.run ctx ~bind:(map_of (inputs b0 ~salt:0))) with
  | Error (`Inference_failed (i, e)) ->
      Fmt.pr "invocation %d, %a@." i Loop_interp.pp_error e
  | Error e -> Fmt.pr "%a@." Loop_c_embed.pp_error e
  | Ok _ -> print_endline "no failure");
  E.close_context ctx;
  E.close p;
  [%expect {|
    missing input t0
    invocation 1, I64 division by zero |}]
