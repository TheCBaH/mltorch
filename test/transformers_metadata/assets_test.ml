open Transformers_metadata.Json_util
open World
module J = Jsont.Json
module Spec = Pt2_fixture.Contract.Tensor_spec

let contract crop : Pt2_fixture.Contract.t =
  {
    artifact_id = W.artifact_id;
    atol = 1e-5;
    rtol = 1e-4;
    dynamic = false;
    graph_sha256 = Pt2_sha256.string "graph";
    verified_cases = 1;
    outputs = [];
    inputs =
      [
        {
          Spec.name = "pixel_values";
          dtype = Pt2_checkpoint_map.Dtype.F32;
          shape = [ 1L; 3L; crop; crop ];
        };
      ];
  }

let processor =
  json
    {|{"crop_size":4,"size":6,"do_center_crop":true,"do_resize":true,"do_flip_channels":true,"feature_extractor_type":"MobileViTFeatureExtractor","resample":2}|}

let labels =
  obj
    (List.init 1000 (fun i ->
         (string_of_int i, J.string ("label" ^ string_of_int i))))

let run_image ?(processor = processor) ?(crop = 4L) ?(recipe = processor) dir =
  let processor_bytes = bytes processor in
  let p =
    json (W.pin_json ~name:"preprocessor_config.json" ~data:processor_bytes)
  in
  let p = obj (List.remove_assoc "name" (unwrap (members p))) in
  let source =
    obj
      [
        ("files", obj [ ("preprocessor_config.json", p) ]);
        ( "recipe",
          obj
            [
              ("image", obj [ ("preprocessor", recipe) ]);
              ("output_decoding", obj [ ("num_labels", J.int 1000) ]);
            ] );
      ]
  in
  let cache = unwrap (U.Cache.create dir) in
  let transport : U.Transport.t =
   fun ~url:_ ~dest ->
    write dest processor_bytes;
    Ok ()
  in
  T.Assets.image
    (U.Bundle.config ~transport cache)
    source (contract crop)
    (obj [ ("id2label", labels) ])
    T.Assets.Mobilevit

let%expect_test
    "processor bytes determine crop/resize; recipe and contract changes refuse"
    =
  with_dir (fun dir ->
      let derived = unwrap (run_image dir) in
      print_endline (bytes derived));
  with_dir (fun dir -> show (run_image ~crop:5L dir));
  with_dir (fun dir ->
      show (run_image ~recipe:(replace "resample" (J.int 3) processor) dir));
  with_dir (fun dir ->
      show
        (run_image
           ~processor:(replace "do_flip_channels" (J.bool false) processor)
           dir));
  [%expect
    {|
    {
      "crop_size": 4,
      "flip_channels": true,
      "preprocessor_config": {
        "name": "preprocessor_config.json",
        "sha256": "cb912f39d780a42c3621bf4582fe51278b360768c358feee27179ab17f09083c",
        "size": 182,
        "url": "https://example.org/preprocessor_config.json"
      },
      "resample": "bilinear",
      "shortest_edge": 6
    }
    invalid metadata: pixel recipe differs from release input contract
    processor recipe: resample is 2, expected 3
    invalid metadata: MobileViT requires bilinear resize and BGR flip |}]

let%expect_test
    "malformed adapter requests, unsafe outputs and incompatible IDs are typed"
    =
  let row =
    obj
      [
        ("adapter", J.string "mobilevit");
        ("artifact_id", J.string W.artifact_id);
        ("output", J.string "../escape.json");
      ]
  in
  show
    (T.Assets.requests
       (obj [ ("schema_version", J.int 1); ("adapters", J.list [ row ]) ]));
  show
    (T.Assets.requests
       (obj
          [
            ("schema_version", J.int 1);
            ( "adapters",
              J.list
                [
                  replace "output" (J.string "asset.json")
                    (replace "adapter" (J.string "unknown") row);
                ] );
          ]));
  with_dir (fun dir ->
      let c = U.Bundle.config (unwrap (U.Cache.create dir)) in
      let cohort = W.cohort in
      show
        (T.Assets.build c producer cohort (json W.cohort_bytes)
           {
             T.Assets.Request.adapter = T.Assets.Mobilevit;
             artifact_id = W.artifact_id;
             output = "asset.json";
           }));
  [%expect
    {|
    unsafe relative path: ../escape.json
    invalid metadata: unsupported adapter: unknown
    toy/task/reference/forward/fp32/dynamo/static/ckpt-aaaaaaaaaaaa: adapter model is toy, expected mobilevit-xxs |}]

let%expect_test "corrupt processor bytes and wrong cohort schema refuse" =
  with_dir (fun dir ->
      let derived = unwrap (run_image dir) in
      let pinned = unwrap (path [ "preprocessor_config" ] derived >>= pin_of) in
      let cache = unwrap (U.Cache.create dir) in
      write (U.Cache.blob_path cache pinned.sha256) "corrupt";
      show
        (U.Fetch.ensure ~layer:(Pt2_fixture.Fault.Source pinned.name) cache
           pinned));
  with_dir (fun dir ->
      let config = U.Bundle.config (unwrap (U.Cache.create dir)) in
      show
        (T.Assets.generate config producer
           (replace "schema_version" (J.int 2) (json W.cohort_bytes))
           []));
  [%expect
    {|
    source "preprocessor_config.json": size is 7 bytes, expected 182
    invalid metadata: unsupported schema_version |}]
