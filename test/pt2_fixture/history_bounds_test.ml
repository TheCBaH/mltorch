module H = Pt2_fixture.History

let%expect_test "history metadata is bounded before narrowing, including JS" =
  let show text =
    match H.of_contract_string text with
    | Ok _ -> print_endline "accepted"
    | Error _ -> print_endline "refused"
  in
  List.iter
    (fun (history, capacity, attention) ->
      show
        (Printf.sprintf
           {|{"variant":{"kind":"static-history","history":%s,"attention_length":%s},"state":{"input":["past_0_key"],"output":["present_0_key"],"maximum_input_history":%s}}|}
           history attention capacity))
    [
      ("4", "4", "5");
      ("-1", "4", "5");
      ("5", "4", "6");
      ("4", "4", "4");
      ("4294967300", "4294967300", "4294967301");
      ("4", "-1", "5");
    ];
  show
    {|{"variant":{"kind":"static-history","history":4,"attention_length":5}}|};
  List.iter
    (fun shape ->
      print_endline
        (match H.history_of_shape shape with
        | None -> "refused"
        | Some _ -> "accepted"))
    [ [ 1L; 3L; 4L; 64L ]; [ 1L; 3L; 4294967300L; 64L ]; [ 1L; 3L; -1L; 64L ] ];
  [%expect
    {|
    accepted
    refused
    refused
    refused
    refused
    refused
    refused
    accepted
    refused
    refused |}]

let%expect_test "cache chaining requires exact dtype, even with equal shapes" =
  let spec name dtype : Pt2_fixture.Contract.Tensor_spec.t =
    { name; dtype; shape = [ 1L; 3L; 4L; 64L ] }
  in
  let decode : H.t =
    {
      history = 4;
      attention_length = 5;
      maximum_input_history = 4;
      state_inputs = [ "past_0_key" ];
      state_outputs = [ "present_0_key" ];
    }
  in
  List.iter
    (fun dtype ->
      match
        H.chain_tensors
          ~prefill:[ spec "present_0_key" Pt2_checkpoint_map.Dtype.F32 ]
          ~decode
          ~decode_inputs:[ spec "past_0_key" dtype ]
      with
      | Ok () -> print_endline "accepted"
      | Error _ -> print_endline "refused")
    [ Pt2_checkpoint_map.Dtype.F32; Pt2_checkpoint_map.Dtype.I64 ];
  [%expect {|
    accepted
    refused |}]
