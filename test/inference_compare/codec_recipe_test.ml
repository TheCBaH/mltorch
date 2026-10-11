open Inference_compare_schema
module Codecs = Inference_compare_codecs
module L = Codecs.Logical_value
module E = Codecs.Recipe_eval
module C = Codec_fixtures

let key = C.canonical "x"

(* Run a one-input recipe over [value] (as canonical "x") and print its
   integers, or the failure. *)
let run_ints ctx value recipe =
  let env = C.env ~canonical:[ (key, L.spec value) ] () in
  match C.run ctx env (C.inputs ~canonical:[ (key, value) ] ()) recipe with
  | Error f -> Format.printf "%a@." C.pp_failure f
  | Ok (_, v, _) ->
      Format.printf "%a : %a@." C.pp_ints (C.to_integers ctx v)
        Fmt.(brackets (list ~sep:semi int64))
        (L.spec v).logical_shape;
      L.drop v

let single ?lossless id step spec =
  C.recipe ?lossless id
    [ C.step_node "n" step [ Canonical key ] ]
    (Conversion_result (C.node "n"))
    spec

let%expect_test "layout steps are exact" =
  let ctx = C.context () in
  (* NCHW [1;2;2;3] -> NHWC [1;2;3;2]: channel is the fastest axis. *)
  let nchw =
    C.integers ctx
      (C.i32 [ 1L; 2L; 2L; 3L ])
      [ 0L; 1L; 2L; 3L; 4L; 5L; 10L; 11L; 12L; 13L; 14L; 15L ]
  in
  run_ints ctx nchw
    (single "to_nhwc" (Permute [ 0; 2; 3; 1 ]) (C.i32 [ 1L; 2L; 3L; 2L ]));
  run_ints ctx nchw (single "flat" (Reshape [ 12L ]) (C.i32 [ 12L ]));
  let m = C.integers ctx (C.i32 [ 4L; 3L ]) (List.init 12 Int64.of_int) in
  run_ints ctx m
    (single "rows"
       (Slice { axis = 0; start = 1L; length = 2L })
       (C.i32 [ 2L; 3L ]));
  run_ints ctx m
    (single "cols"
       (Slice { axis = 1; start = 2L; length = 1L })
       (C.i32 [ 4L; 1L ]));
  run_ints ctx m
    (single "padded rank"
       (Reshape [ 1L; 4L; 1L; 3L ])
       (C.i32 [ 1L; 4L; 1L; 3L ]));
  [%expect
    {|
    [0; 10; 1; 11; 2; 12; 3; 13; 4; 14; 5; 15] : [1; 2; 3; 2]
    [0; 1; 2; 3; 4; 5; 10; 11; 12; 13; 14; 15] : [12]
    [3; 4; 5; 6; 7; 8] : [2; 3]
    [2; 5; 8; 11] : [4; 1]
    [0; 1; 2; 3; 4; 5; 6; 7; 8; 9; 10; 11] : [1; 4; 1; 3] |}]

let%expect_test "pack and split roundtrip exactly" =
  let ctx = C.context () in
  let a = C.integers ctx (C.i64 [ 2L; 2L ]) [ 1L; 2L; 3L; 4L ] in
  let b =
    C.integers ctx
      (C.i64 [ 2L; 1L ])
      [ Int64.add (Int64.shift_left 1L 53) 1L; 6L ]
  in
  let ka = C.canonical "a" and kb = C.canonical "b" in
  let env = C.env ~canonical:[ (ka, L.spec a); (kb, L.spec b) ] () in
  let inputs = C.inputs ~canonical:[ (ka, a); (kb, b) ] () in
  let pack =
    C.recipe "pack"
      [ C.step_node "cat" (Concat 1) [ Canonical ka; Canonical kb ] ]
      (Conversion_result (C.node "cat"))
      (C.i64 [ 2L; 3L ])
  in
  (match C.run ctx env inputs pack with
  | Error f -> Format.printf "%a@." C.pp_failure f
  | Ok (_, packed, lineage) ->
      Format.printf "packed: %a@." C.pp_ints (C.to_integers ctx packed);
      Format.printf "lineage: %d entries@." (List.length lineage);
      let kp = C.canonical "p" in
      let env2 = C.env ~canonical:[ (kp, L.spec packed) ] () in
      let inputs2 = C.inputs ~canonical:[ (kp, packed) ] () in
      let split name start length spec original =
        let r =
          C.recipe name
            [
              C.step_node "s"
                (Slice { axis = 1; start; length })
                [ Canonical kp ];
            ]
            (Conversion_result (C.node "s"))
            spec
        in
        match C.run ctx env2 inputs2 r with
        | Error f -> Format.printf "%a@." C.pp_failure f
        | Ok (_, part, _) ->
            Format.printf "%s recovered exactly: %b@." name
              (L.equal part original);
            L.drop part
      in
      split "split a" 0L 2L (C.i64 [ 2L; 2L ]) a;
      split "split b" 2L 1L (C.i64 [ 2L; 1L ]) b;
      L.drop packed);
  [%expect
    {|
    packed: [1; 2; 9007199254740993; 3; 4; 6]
    lineage: 1 entries
    split a recovered exactly: true
    split b recovered exactly: true |}]

let%expect_test "casts" =
  let ctx = C.context () in
  let cast ?(lossless = true) ?(rounding = Conversion_step.Nearest_even)
      ?(overflow = Conversion_step.Reject) target spec =
    single ~lossless "cast"
      (Cast
         {
           target = { codec = target; parameters_digest = None };
           rounding;
           overflow;
         })
      spec
  in
  let i32s = C.integers ctx (C.i32 [ 3L ]) [ -5L; 0L; 2147483647L ] in
  run_ints ctx i32s (cast Codecs.Element_codec.i64 (C.i64 [ 3L ]));
  let i64s = C.integers ctx (C.i64 [ 3L ]) [ 7L; 2147483648L; -2147483649L ] in
  Format.printf "-- narrowing a lossless-declared cast@.";
  run_ints ctx i64s (cast Codecs.Element_codec.i32 (C.i32 [ 3L ]));
  Format.printf "-- declared lossy, reject@.";
  run_ints ctx i64s
    (cast ~lossless:false Codecs.Element_codec.i32 (C.i32 [ 3L ]));
  Format.printf "-- saturate@.";
  run_ints ctx i64s
    (cast ~lossless:false ~overflow:Saturate Codecs.Element_codec.i32
       (C.i32 [ 3L ]));
  Format.printf "-- wrap@.";
  run_ints ctx i64s
    (cast ~lossless:false ~overflow:Wrap Codecs.Element_codec.i32 (C.i32 [ 3L ]));
  let fs = C.floats ctx (C.f64 [ 5L ]) [ 2.5; 3.5; -2.5; 0.5; -0.5 ] in
  Format.printf "-- float to int, nearest even@.";
  run_ints ctx fs (cast ~lossless:false Codecs.Element_codec.i32 (C.i32 [ 5L ]));
  Format.printf "-- float to int, toward negative@.";
  run_ints ctx fs
    (cast ~lossless:false ~rounding:Toward_negative Codecs.Element_codec.i32
       (C.i32 [ 5L ]));
  let nonfinite = C.floats ctx (C.f64 [ 2L ]) [ 1.0; Float.nan ] in
  Format.printf "-- nonfinite to int@.";
  run_ints ctx nonfinite
    (cast ~lossless:false Codecs.Element_codec.i32 (C.i32 [ 2L ]));
  Format.printf "-- int64 to float32 is not admitted@.";
  run_ints ctx i64s
    (cast ~lossless:false Codecs.Element_codec.f32 (C.f32 [ 3L ]));
  Format.printf "-- float to bool is not admitted@.";
  run_ints ctx fs
    (cast ~lossless:false Codecs.Element_codec.bool
       (C.spec Codecs.Element_codec.bool [ 5L ]));
  Format.printf "-- wrap from float is not admitted@.";
  run_ints ctx fs
    (cast ~lossless:false ~overflow:Wrap Codecs.Element_codec.i32 (C.i32 [ 5L ]));
  [%expect
    {|
    [-5; 0; 2147483647] : [3]
    -- narrowing a lossless-declared cast
    check: recipe declared lossless but is lossy
    -- declared lossy, reject
    eval: cast overflow at element 1
    -- saturate
    [7; 2147483647; -2147483648] : [3]
    -- wrap
    [7; -2147483648; 2147483647] : [3]
    -- float to int, nearest even
    [2; 4; -2; 0; 0] : [5]
    -- float to int, toward negative
    [2; 3; -3; 0; -1] : [5]
    -- nonfinite to int
    eval: cast of a nonfinite value at element 1
    -- int64 to float32 is not admitted
    check: unsupported cast i64 -> f32
    -- float to bool is not admitted
    check: unsupported cast f64 -> bool
    -- wrap from float is not admitted
    check: unsupported overflow policy |}]

let%expect_test "float widening keeps nan payloads and zero signs" =
  let ctx = C.context () in
  let src =
    let b = Bytes.create 16 in
    List.iteri
      (fun i bits -> Bytes.set_int32_le b (i * 4) (Int64.to_int32 bits))
      [ 0x7FA00001L; 0xFFC00000L; 0x80000000L; 0x7F800000L ];
    C.ok (L.make ctx (C.f32 [ 4L ]) (Bytes.to_string b))
  in
  let to64 =
    single "widen"
      (Cast
         {
           target =
             { codec = Codecs.Element_codec.f64; parameters_digest = None };
           rounding = Nearest_even;
           overflow = Reject;
         })
      (C.f64 [ 4L ])
  in
  let env = C.env ~canonical:[ (key, L.spec src) ] () in
  (match C.run ctx env (C.inputs ~canonical:[ (key, src) ] ()) to64 with
  | Error f -> Format.printf "%a@." C.pp_failure f
  | Ok (_, wide, _) ->
      Format.printf "f64 bits: %a@." C.pp_hex_bits (C.to_bits ctx wide);
      let kw = C.canonical "w" in
      let back =
        C.recipe ~lossless:false "narrow"
          [
            C.step_node "n"
              (Cast
                 {
                   target =
                     {
                       codec = Codecs.Element_codec.f32;
                       parameters_digest = None;
                     };
                   rounding = Nearest_even;
                   overflow = Reject;
                 })
              [ Canonical kw ];
          ]
          (Conversion_result (C.node "n"))
          (C.f32 [ 4L ])
      in
      let env2 = C.env ~canonical:[ (kw, L.spec wide) ] () in
      (match C.run ctx env2 (C.inputs ~canonical:[ (kw, wide) ] ()) back with
      | Error f -> Format.printf "%a@." C.pp_failure f
      | Ok (_, narrow, _) ->
          Format.printf "f32 roundtrip bit-identical: %b@." (L.equal narrow src);
          L.drop narrow);
      L.drop wide);
  [%expect
    {|
    f64 bits: [0x7ff4000020000000; 0xfff8000000000000; 0x8000000000000000;
               0x7ff0000000000000]
    f32 roundtrip bit-identical: true |}]

let%expect_test "structural refusals" =
  let ctx = C.context () in
  let v = C.integers ctx (C.i32 [ 2L; 3L ]) [ 0L; 1L; 2L; 3L; 4L; 5L ] in
  let try_ name step spec =
    Format.printf "%s: " name;
    run_ints ctx v (single name step spec)
  in
  try_ "reshape changes the count" (Reshape [ 7L ]) (C.i32 [ 7L ]);
  (* Two different shapes whose products agree modulo 2^64: a wrapped product
     would accept this reshape; the checked product refuses it. *)
  let wide = C.spec Codecs.Element_codec.i32 [ 4294967296L ] in
  let wide_env = C.env ~canonical:[ (key, wide) ] () in
  (match
     C.run (C.context ()) wide_env (C.inputs ())
       (single "wrapping reshape"
          (Reshape [ 4294967296L; 4294967297L ])
          (C.i32 [ 4294967296L; 4294967297L ]))
   with
  | Error f -> Format.printf "wrapping reshape: %a@." C.pp_failure f
  | Ok _ -> Format.printf "wrapping reshape: ACCEPTED@.");
  try_ "permute with a repeated axis" (Permute [ 0; 0 ]) (C.i32 [ 2L; 2L ]);
  try_ "permute with a short list" (Permute [ 0 ]) (C.i32 [ 2L; 3L ]);
  try_ "permute out of range" (Permute [ 0; 2 ]) (C.i32 [ 2L; 3L ]);
  try_ "slice past the end"
    (Slice { axis = 1; start = 2L; length = 2L })
    (C.i32 [ 2L; 2L ]);
  try_ "slice length overflow"
    (Slice { axis = 1; start = 2L; length = Int64.max_int })
    (C.i32 [ 2L; 1L ]);
  try_ "slice bad axis"
    (Slice { axis = 2; start = 0L; length = 1L })
    (C.i32 [ 2L; 1L ]);
  try_ "concat with one operand and a bad axis" (Concat 5) (C.i32 [ 2L; 3L ]);
  try_ "result spec differs from the declared one"
    (Reshape [ 3L; 2L ])
    (C.i32 [ 2L; 3L ]);
  [%expect
    {|
    reshape changes the count: check: element count changed from 6 to 7
    wrapping reshape: check: element count overflow
    permute with a repeated axis: check: duplicate axis 0
    permute with a short list: check: permutation has 1 axes, rank is 2
    permute out of range: check: axis 2 out of range for rank 2
    slice past the end: check: slice [2, +2) outside extent 3 on axis 1
    slice length overflow: check: slice [2, +9223372036854775807) outside extent 3 on axis 1
    slice bad axis: check: axis 2 out of range for rank 2
    concat with one operand and a bad axis: check: axis 5 out of range for rank 2
    result spec differs from the declared one: check: tensor spec mismatch |}]

let%expect_test "graph-level refusals" =
  let ctx = C.context () in
  let v = C.integers ctx (C.i32 [ 2L ]) [ 1L; 2L ] in
  let env = C.env ~canonical:[ (key, L.spec v) ] () in
  let inputs = C.inputs ~canonical:[ (key, v) ] () in
  let show name r =
    match C.run ctx env inputs r with
    | Error f -> Format.printf "%s: %a@." name C.pp_failure f
    | Ok (_, out, _) ->
        Format.printf "%s: ok %a@." name C.pp_ints (C.to_integers ctx out);
        L.drop out
  in
  let spec = C.i32 [ 2L ] in
  let reshape = Conversion_step.Reshape [ 2L ] in
  show "cycle"
    (C.recipe "cycle"
       [
         C.step_node "a" reshape [ Conversion_result (C.node "b") ];
         C.step_node "b" reshape [ Conversion_result (C.node "a") ];
       ]
       (Conversion_result (C.node "a"))
       spec);
  show "unknown node"
    (C.recipe "ghost" [] (Conversion_result (C.node "nope")) spec);
  show "unknown canonical"
    (C.recipe "missing" [] (Canonical (C.canonical "absent")) spec);
  show "unknown boundary"
    (C.recipe "boundary" [] (Boundary (C.boundary "t")) spec);
  show "passthrough with the identical spec"
    (C.recipe "same" [] (Canonical key) spec);
  show "passthrough relabelled to another layout"
    (C.recipe "relabel" [] (Canonical key)
       { spec with layout_id = C.text (Id.Layout.of_string "nhwc") });
  show "duplicate node"
    (C.recipe "dup"
       [
         C.step_node "a" reshape [ Canonical key ];
         C.step_node "a" reshape [ Canonical key ];
       ]
       (Conversion_result (C.node "a"))
       spec);
  show "unreachable node is never run"
    (C.recipe "dead"
       [
         C.step_node "live" reshape [ Canonical key ];
         C.step_node "dead"
           (Slice { axis = 0; start = 9L; length = 1L })
           [ Canonical key ];
       ]
       (Conversion_result (C.node "live"))
       spec);
  [%expect
    {|
    cycle: check: cyclic recipe at a
    unknown node: check: unknown node nope
    unknown canonical: check: unknown canonical value absent
    unknown boundary: check: unknown boundary tensor native/v1@sha256:0000000000000000000000000000000000000000000000000000000000000000:/t
    passthrough with the identical spec: ok [1; 2]
    passthrough relabelled to another layout: check: tensor spec mismatch
    duplicate node: check: duplicate node a
    unreachable node is never run: ok [1; 2] |}]

let%expect_test "intermediates are released and bounded" =
  let ctx = C.context () in
  let v = C.integers ctx (C.i32 [ 4L ]) [ 1L; 2L; 3L; 4L ] in
  let r =
    C.recipe "chain"
      [
        C.step_node "a" (Reshape [ 2L; 2L ]) [ Canonical key ];
        C.step_node "b" (Permute [ 1; 0 ]) [ Conversion_result (C.node "a") ];
        C.step_node "c" (Reshape [ 4L ]) [ Conversion_result (C.node "b") ];
      ]
      (Conversion_result (C.node "c"))
      (C.i32 [ 4L ])
  in
  let env = C.env ~canonical:[ (key, L.spec v) ] () in
  (match C.run ctx env (C.inputs ~canonical:[ (key, v) ] ()) r with
  | Error f -> Format.printf "%a@." C.pp_failure f
  | Ok (checked, out, lineage) ->
      Format.printf "result: %a@." C.pp_ints (C.to_integers ctx out);
      Format.printf "lossless: %b, reads: %a@."
        (E.Checked.lossless checked)
        Fmt.(list ~sep:comma Id.Canonical_value.pp)
        (E.Checked.canonical_reads checked);
      Format.printf "lineage nodes: %a@."
        Fmt.(
          list ~sep:(any ",")
            (using
               (fun (e : E.Lineage.entry) ->
                 Id.Conversion_node.to_string e.node)
               string))
        lineage;
      Format.printf "live blobs (input + result): %d@."
        (Codecs.Owned_blob.Ledger.live_blobs ctx.ledger);
      L.drop out);
  L.drop v;
  Format.printf "balanced: %b@." (Codecs.Owned_blob.Ledger.balanced ctx.ledger);
  (* A limit that fits the input and one intermediate but not two. *)
  let tight = C.context ~limit:40L () in
  let v = C.integers tight (C.i32 [ 4L ]) [ 1L; 2L; 3L; 4L ] in
  let env = C.env ~canonical:[ (key, L.spec v) ] () in
  (match C.run tight env (C.inputs ~canonical:[ (key, v) ] ()) r with
  | Error f -> Format.printf "tight: %a@." C.pp_failure f
  | Ok (_, out, _) ->
      Format.printf "tight: ok@.";
      L.drop out);
  L.drop v;
  Format.printf "balanced after failure: %b@."
    (Codecs.Owned_blob.Ledger.balanced tight.ledger);
  [%expect
    {|
    result: [1; 3; 2; 4]
    lossless: true, reads: x
    lineage nodes: a,b,c
    live blobs (input + result): 2
    balanced: true
    tight: eval: resource: requested 16, limit 40
    balanced after failure: true |}]
