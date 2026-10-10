open World
module L = F.Logical

let floats values =
  get (T.Input.f32 [ 1L; Int64.of_int (Array.length values) ] values)

let refused result = Result.is_error (Err.payload result)

let%expect_test
    "classification uses actual logits and refuses ambiguous rankings" =
  let output = get (T.Host.top5 (floats [| -9.; 2.; 7.; -1.; 4.; 5.; -8. |])) in
  let ids = get (T.Lifecycle.named "top5_ids" output) in
  let logits = get (T.Lifecycle.named "top5_logits" output) in
  for i = 0 to 4 do
    Printf.printf "%Ld:%g " (L.get_int64 ids i) (L.get_float logits i)
  done;
  Printf.printf "\n";
  List.iter
    (fun values ->
      Printf.printf "refused=%b\n" (refused (T.Host.top5 (floats values))))
    [
      [| 1.; 2.; 3.; 4. |];
      [| 1.; 2.; 3.; 4.; nan |];
      [| 1.; 2.; 3.; 4.; infinity |];
      [| 5.; 5.; 4.; 3.; 2.; 1. |];
      [| 6.; 5.; 4.; 3.; 2.; 2. |];
    ];
  [%expect
    {|
    2:7 5:5 4:4 1:2 3:-1
    refused=true
    refused=true
    refused=true
    refused=true
    refused=true |}]

let%expect_test
    "CLIP host checks features, unit normalization, scale and both scores" =
  let image_features = floats [| 3.; 4. |] in
  let text_features = floats [| 4.; 3. |] in
  let log_scale = get (T.Input.f32 [] [| 0. |]) in
  let actual = get (T.Host.clip ~log_scale ~image_features ~text_features) in
  let expected =
    [
      ("image_features", image_features);
      ("text_features", text_features);
      ("image_embeds", floats [| 0.6; 0.8 |]);
      ("text_embeds", floats [| 0.8; 0.6 |]);
      ("logit_scale", get (T.Input.f32 [] [| 1. |]));
      ("logits_per_image", get (T.Input.f32 [ 1L; 1L ] [| 0.96 |]));
      ("logits_per_text", get (T.Input.f32 [ 1L; 1L ] [| 0.96 |]));
    ]
  in
  let checks =
    get (T.Diagnostic.compare ~atol:0.0000001 ~rtol:0. expected actual)
  in
  Printf.printf "all seven outputs pass=%b\n"
    (List.for_all F.Compare.passed checks);
  let doubled = get (T.Input.f32 [] [| log 2. |]) in
  let outputs =
    get (T.Host.clip ~log_scale:doubled ~image_features ~text_features)
  in
  let scores = get (T.Lifecycle.named "logits_per_image" outputs) in
  Printf.printf "checkpoint scale changes scores=%g\n" (L.get_float scores 0);
  Printf.printf "zero norm=%b, width=%b, scalar=%b, overflow=%b\n"
    (refused
       (T.Host.clip ~log_scale
          ~image_features:(floats [| 0.; 0. |])
          ~text_features))
    (refused
       (T.Host.clip ~log_scale ~image_features ~text_features:(floats [| 1. |])))
    (refused
       (T.Host.clip ~log_scale:(floats [| 0. |]) ~image_features ~text_features))
    (refused
       (T.Host.clip
          ~log_scale:(get (T.Input.f32 [] [| 1000. |]))
          ~image_features ~text_features));
  [%expect
    {|
    all seven outputs pass=true
    checkpoint scale changes scores=1.92
    zero norm=true, width=true, scalar=true, overflow=true |}]
