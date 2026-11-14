open Band_schedule

let c = Count.of_int
let r = Row.of_int

let kernel ?(stride = 1) ?(pad = 0) ?(dilation = 1) ?(chunk = 1) name extent =
  {
    name;
    extent = c extent;
    stride = c stride;
    pad = c pad;
    dilation = c dilation;
    chunk = c chunk;
  }

let rows k ~lo ~hi ~height =
  let a, b = input_rows k ~lo:(r lo) ~hi:(r hi) ~height:(c height) in
  Fmt.pr "[%d,%d) -> [%d,%d)@." lo hi (Row.to_int a) (Row.to_int b)

let%expect_test "a kernel maps output rows to the input rows it reads" =
  let conv3 = kernel "conv3" 3 ~pad:1 in
  rows conv3 ~lo:0 ~hi:1 ~height:8;
  rows conv3 ~lo:3 ~hi:5 ~height:8;
  rows conv3 ~lo:7 ~hi:8 ~height:8;
  (* a 1x1 convolution: the identity at stride 1, [2lo, 2(hi-1)+1) at stride 2 *)
  rows (kernel "p1" 1) ~lo:2 ~hi:5 ~height:8;
  rows (kernel "p1s2" 1 ~stride:2) ~lo:2 ~hi:5 ~height:16;
  [%expect
    {|
    [0,1) -> [0,2)
    [3,5) -> [2,6)
    [7,8) -> [6,8)
    [2,5) -> [2,5)
    [2,5) -> [4,9) |}]

let%expect_test "output height follows the window, stride and padding" =
  let h k input = Count.to_int (output_height k ~input:(c input)) in
  Fmt.pr "%d %d %d %d@."
    (h (kernel "a" 3 ~pad:1) 8)
    (h (kernel "b" 3 ~stride:2 ~pad:1) 8)
    (h (kernel "c" 3 ~dilation:2) 8)
    (h (kernel "d" 3) 2);
  [%expect {| 8 4 4 0 |}]

let%expect_test "a producer that runs ahead keeps the row a later demand needs"
    =
  (* chunks of 4, a 3x3 stride 1 pad 1 consumer, one output row a call: the
     first demand [0,2) makes [0,4); row 3 stays for the demand [1,4) *)
  let producer = kernel "producer" 1 ~chunk:4 in
  let consumer = kernel "consumer" 3 ~pad:1 in
  let s = run [ producer; consumer ] ~input:(c 8) ~block:(c 1) in
  Fmt.pr "%a" pp s;
  [%expect
    {|
    producer produces [0,4) from [0,4)
    consumer produces [0,1) from [0,2)
    consumer produces [1,2) from [0,3)
    consumer produces [2,3) from [1,4)
    producer produces [4,8) from [4,8)
    consumer produces [3,4) from [2,5)
    consumer produces [4,5) from [3,6)
    consumer produces [5,6) from [4,7)
    consumer produces [6,7) from [5,8)
    consumer produces [7,8) from [6,8)
    windows: 8 0 |}]

let%expect_test "interior blocks overlap by the window less the stride" =
  let consumed k =
    let s = run [ kernel "p" 1 ~chunk:16; k ] ~input:(c 16) ~block:(c 1) in
    List.filter_map
      (fun e ->
        if String.equal e.kernel k.name then
          Some (Row.to_int (fst e.consumed), Row.to_int (snd e.consumed))
        else None)
      s.events
  in
  let show name k =
    let l = consumed k in
    let overlap =
      match l with
      | (_, b0) :: (a1, _) :: (_ :: _ :: _ :: _ as rest)
        when List.length rest > 2 ->
          max 0 (b0 - a1)
      | _ -> 0
    in
    Fmt.pr "%s: overlap of interior blocks %d@." name overlap
  in
  show "k3 d1 s1" (kernel "c" 3);
  show "k3 d2 s1" (kernel "c" 3 ~dilation:2);
  show "k3 d1 s2" (kernel "c" 3 ~stride:2);
  show "k1 s2" (kernel "c" 1 ~stride:2);
  [%expect
    {|
    k3 d1 s1: overlap of interior blocks 2
    k3 d2 s1: overlap of interior blocks 4
    k3 d1 s2: overlap of interior blocks 1
    k1 s2: overlap of interior blocks 0 |}]

let%expect_test
    "a segment is admitted only as one spine, with covered side outputs" =
  let producer = kernel "producer" 3 ~pad:1 ~chunk:2 in
  let consume ?(stride = 1) ?(extent = 1) () =
    kernel "consumer" extent ~stride
  in
  let show name result =
    match result with
    | Ok s ->
        Fmt.pr "%s: admitted, windows %a@." name
          Fmt.(list ~sep:(any " ") int)
          (List.map Count.to_int s.windows)
    | Error r -> Fmt.pr "%s: refused: %a@." name pp_refusal r
  in
  let chain = [ producer; consume () ] in
  let spine = [ ("producer", [ "consumer" ]) ] in
  let admit ?(consumers = spine) ?(side = []) kernels =
    admit kernels ~consumers ~side_outputs:side ~input:(c 8) ~block:(c 1)
  in
  show "spine" (admit chain);
  show "side output, every row demanded" (admit chain ~side:[ "producer" ]);
  show "side output, a stride-2 consumer skips rows"
    (admit [ producer; consume ~stride:2 () ] ~side:[ "producer" ]);
  show "no side output, a stride-2 consumer"
    (admit [ producer; consume ~stride:2 () ]);
  show "fan-out"
    (admit chain ~consumers:[ ("producer", [ "consumer"; "other" ]) ]);
  show "second sink" (admit chain ~consumers:[]);
  show "unknown side output" (admit chain ~side:[ "nothing" ]);
  [%expect
    {|
    spine: admitted, windows 2 0
    side output, every row demanded: admitted, windows 2 0
    side output, a stride-2 consumer skips rows: refused: a side output of producer: row 1 is never demanded
    no side output, a stride-2 consumer: admitted, windows 2 0
    fan-out: refused: producer feeds two consumers in the segment
    second sink: refused: producer is a second sink
    unknown side output: refused: nothing is no kernel of the segment |}]

let%expect_test "the verifier accepts a schedule and refuses each mutation" =
  let producer = kernel "producer" 1 ~chunk:4 in
  let consumer = kernel "consumer" 3 ~pad:1 in
  let chain = [ producer; consumer ] in
  let s = run chain ~input:(c 8) ~block:(c 1) in
  let show name t =
    match verify chain ~input:(c 8) t with
    | Ok () -> Fmt.pr "%s: valid@." name
    | Error v -> Fmt.pr "%s: %a@." name pp_violation v
  in
  show "the schedule" s;
  List.iter
    (fun (name, m) -> show name (mutate m s))
    [
      ("dropped event", Drop_event);
      ("repeated event", Repeat_event);
      ("consumer first", Run_consumer_first);
      ("shifted read", Shift_consumed);
    ];
  [%expect
    {|
    the schedule: valid
    dropped event: consumer skips to row 1
    repeated event: producer produces row 0 twice
    consumer first: event 0 of consumer reads row 0 before it is produced
    shifted read: event 1 of consumer reads the wrong input rows |}]
