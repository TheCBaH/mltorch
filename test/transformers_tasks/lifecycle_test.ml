open World
module L = T.Lifecycle

let outputs values =
  [ ("logits", get (T.Input.f32 [ 1L; 1L; 3L ] (Array.of_list values))) ]

let show result =
  match Err.payload result with
  | Ok _ -> print_endline "accepted"
  | Error _ -> print_endline "refused"

let%expect_test "greedy ties, EOS, limit and independent prompt resets" =
  let fresh () = get (L.fresh ~eos:[ 2L ] ~maximum:2) in
  let state, token, stop =
    get (L.advance (fresh ()) (outputs [ 0.; 1.; 1. ]))
  in
  Printf.printf "first tie: %Ld, continue=%b\n" token (stop = L.Continue);
  let state, _, stop = get (L.advance state (outputs [ 0.; 1.; 0. ])) in
  Printf.printf "second token: count=%d, limit=%b\n"
    (List.length state.generated)
    (stop = L.Limit);
  show (L.advance state (outputs [ 0.; 1.; 0. ]));
  let state, _, stop = get (L.advance (fresh ()) (outputs [ 0.; 0.; 1. ])) in
  Printf.printf "EOS=%b\n" (stop = L.Eos);
  show (L.advance state (outputs [ 0.; 1.; 0. ]));
  Printf.printf "fresh state=%d\n" (List.length (fresh ()).generated);
  show (L.fresh ~eos:[ 2L ] ~maximum:3);
  show (L.advance (fresh ()) (outputs [ nan; 0.; 1. ]));
  [%expect
    {|
    first tie: 1, continue=true
    second token: count=2, limit=true
    refused
    EOS=true
    refused
    fresh state=0
    refused
    refused |}]

let%expect_test
    "cache sizes are derived; masks and every K/V binding are required" =
  let history : F.History.t =
    {
      history = 4;
      attention_length = 5;
      maximum_input_history = 4;
      state_inputs = [ "past_0_key"; "past_0_value" ];
      state_outputs = [ "present_0_key"; "present_0_value" ];
    }
  in
  let kv = get (T.Input.f32 [ 1L; 2L; 4L; 3L ] (Array.make 24 1.)) in
  let values = [ ("present_0_key", kv); ("present_0_value", kv) ] in
  let mask values = get (T.Input.i64 [ 1L; 4L ] values) in
  let inputs = get (L.next ~history values (mask [ 1L; 1L; 1L; 1L ]) 1L) in
  let shape = (get (L.named "past_0_key" inputs)).F.Logical.shape in
  Printf.printf "cache=%s\n"
    (String.concat "," (List.map Int64.to_string shape));
  show (L.next ~history (List.tl values) (mask [ 1L; 1L; 1L; 1L ]) 1L);
  show (L.next ~history values (mask [ 1L; 1L; 1L; 0L ]) 1L);
  print_endline
    (match F.History.check_feed history ~requested:5 with
    | Ok () -> "accepted"
    | Error _ -> "refused");
  [%expect {|
    cache=1,2,4,3
    refused
    refused
    refused |}]

let%expect_test "input construction refuses malformed extents and byte lengths"
    =
  show (T.Input.i64 [ 1L; 4L ] [ 1L ]);
  show (T.Input.i64 [ -1L ] []);
  show (T.Input.i64 [ 4294967296L; 4294967296L ] []);
  show (T.Input.i64 [ 0L; 1_000_000L; 1_000_000L; 1_000_000L; 1_000_000L ] []);
  [%expect {|
    refused
    refused
    refused
    refused |}]
