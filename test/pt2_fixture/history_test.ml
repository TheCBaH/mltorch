(* What a static-history decode artifact covers, on the shapes the released
   SmolLM2 prefill and decode artifacts actually carry. *)

module H = Pt2_fixture.History

let decode_contract ~history =
  Printf.sprintf
    {|{"state":{"host":"h","input":["past_0_key","past_0_value"],"maximum_input_history":%d,"output":["present_0_key","present_0_value"],"semantics":"s"},"variant":{"attention_length":%d,"history":%d,"kind":"static-history"}}|}
    history (history + 1) history

let decode history =
  match H.of_contract_string (decode_contract ~history) with
  | Ok (Some h) -> h
  | _ -> failwith "decode contract"

let kv history = [ 1L; 3L; Int64.of_int history; 64L ]

let show = function
  | Ok () -> print_endline "ok"
  | Error f -> Fmt.pr "%a@." H.pp_fault f

let%expect_test "only contracts that are static-history snapshots carry a scope"
    =
  (match H.of_contract_string (decode_contract ~history:4) with
  | Ok (Some h) -> print_endline (H.scope h)
  | _ -> print_endline "none");
  List.iter
    (fun text ->
      match H.of_contract_string text with
      | Ok None -> print_endline "none"
      | Ok (Some _) -> print_endline "some"
      | Error _ -> print_endline "malformed")
    [
      {|{"state":{"input":[],"maximum_input_history":4,"output":[]}}|};
      {|{"variant":{"attention_length":5,"history":4,"kind":"dynamic"},"state":{"input":[],"maximum_input_history":4,"output":[]}}|};
      {|{"call":{}}|};
      "not json";
    ];
  [%expect
    {|
    static snapshot at history 4 (attention length 5, capacity 4): other histories are not covered
    none
    none
    none
    malformed |}]

let%expect_test "a feed is covered only at the exported history" =
  let h = decode 4 in
  show (H.check_feed h ~requested:4);
  show (H.check_feed h ~requested:3);
  show (H.check_feed h ~requested:5);
  show (H.check_feed h ~requested:0);
  [%expect
    {|
    ok
    history 3 is not covered: this artifact is a static snapshot at history 4, and no other length is established by it
    history 5 exceeds the artifact's capacity of 4
    history 0 is not covered: this artifact is a static snapshot at history 4, and no other length is established by it |}]

let%expect_test "a prefill chains to the decode whose history is its length" =
  let prefill len =
    [
      ("logits", [ 1L; Int64.of_int len; 49152L ]);
      ("present_0_key", kv len);
      ("present_0_value", kv len);
    ]
  in
  let inputs history =
    [
      ("input_ids", [ 1L; 1L ]);
      ("past_0_key", kv history);
      ("past_0_value", kv history);
    ]
  in
  let chain ~prefill_len ~history =
    show
      (H.chain ~prefill:(prefill prefill_len) ~decode:(decode history)
         ~decode_inputs:(inputs history))
  in
  chain ~prefill_len:4 ~history:4;
  chain ~prefill_len:4 ~history:8;
  (* heads differ: a prefill from another model *)
  show
    (H.chain
       ~prefill:
         [
           ("present_0_key", [ 1L; 9L; 4L; 64L ]);
           ("present_0_value", [ 1L; 9L; 4L; 64L ]);
         ]
       ~decode:(decode 4) ~decode_inputs:(inputs 4));
  (* names differ: a prefill that lacks the second layer's state *)
  show
    (H.chain
       ~prefill:[ ("present_0_key", kv 4) ]
       ~decode:(decode 4) ~decode_inputs:(inputs 4));
  [%expect
    {|
    ok
    prefill and decode do not meet on history length of 0_key: prefill has 4, decode 8
    prefill and decode do not meet on batch, heads or head dimension of 0_key: prefill has 1,9,64, decode 1,3,64
    prefill and decode do not meet on state names: prefill has 0_key, decode 0_key,0_value |}]
