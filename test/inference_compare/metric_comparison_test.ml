open Inference_compare_schema
module M = Inference_compare_metrics
module F = Metric_fixtures
module T = M.Tensor_metrics

let cid = Codec_fixtures.text (Id.Comparison.of_string "cmp")

let pdigest =
  Result.get_ok
    (Digest.Comparison_policy.make Plan_fixtures.sha256
       ~hex:(String.make 64 '0'))

let metric ?(partition = Tensor_metric.Whole_tensor) id kind role :
    Tensor_metric.t =
  {
    id = Codec_fixtures.text (Id.Tensor_metric.of_string id);
    kind;
    partition;
    role;
  }

let verdict_name : Comparison_result.verdict -> string = function
  | Metadata_error -> "metadata_error"
  | Metric_unavailable -> "metric_unavailable"
  | Mismatch -> "mismatch"
  | Pass -> "pass"
  | Unsupported -> "unsupported"

let compare_ ?(env = T.empty_env) name policy left right =
  match
    M.Comparison.compare F.ctx env policy ~comparison_id:cid
      ~policy_digest:pdigest ~left ~right
  with
  | Error e ->
      Format.printf "%s: CONFIG %a@." name M.Comparison.pp_error
        (Err.Error.kind e)
  | Ok r ->
      Format.printf "%s: %s checked=%Ld/%Ld element-violations=%s metrics=%d@."
        name (verdict_name r.verdict) r.elements_checked r.elements_expected
        (match Option.bind r.element (fun e -> e.tolerance_violations) with
        | Some v -> Int64.to_string v
        | None -> "-")
        (List.length r.tensor_metrics)

let rmse role = metric "rmse" Root_mean_square_error role
let cos role = metric "cos" (Cosine_similarity { zero_norm = Undefined }) role

let%expect_test "the verdict is the conjunction of every required check" =
  let l = F.f64 [ 1.0; 2.0; 3.0; 4.0 ] in
  let element = F.floating ~atol:0.5 () in
  let p ?(element = element) metrics = F.policy ~element ~metrics () in
  compare_ "identical" (p [ rmse (Required (At_most 0.1)) ]) l l;
  compare_ "within element tolerance" (p []) l (F.f64 [ 1.1; 2.0; 3.0; 4.0 ]);
  compare_ "element violated" (p []) l (F.f64 [ 2.0; 2.0; 3.0; 4.0 ]);
  (* element tolerance is loose, the required aggregate is not *)
  compare_ "element passes, required rmse fails"
    (p [ rmse (Required (At_most 0.01)) ])
    l
    (F.f64 [ 1.4; 2.4; 3.4; 4.4 ]);
  compare_ "diagnostic metric failing changes nothing"
    (p [ rmse Diagnostic_only ])
    l
    (F.f64 [ 1.4; 2.4; 3.4; 4.4 ]);
  compare_ "metrics only, no element rule, threshold met"
    (F.policy ~metrics:[ rmse (Required (At_most 1.0)) ] ())
    l
    (F.f64 [ 1.4; 2.4; 3.4; 4.4 ]);
  compare_ "metrics only, threshold missed"
    (F.policy ~metrics:[ rmse (Required (At_most 0.1)) ] ())
    l
    (F.f64 [ 1.4; 2.4; 3.4; 4.4 ]);
  [%expect
    {|
    identical: pass checked=4/4 element-violations=0 metrics=1
    within element tolerance: pass checked=4/4 element-violations=0 metrics=0
    element violated: mismatch checked=4/4 element-violations=1 metrics=0
    element passes, required rmse fails: mismatch checked=4/4 element-violations=0 metrics=1
    diagnostic metric failing changes nothing: pass checked=4/4 element-violations=0 metrics=1
    metrics only, no element rule, threshold met: pass checked=4/4 element-violations=- metrics=1
    metrics only, threshold missed: mismatch checked=4/4 element-violations=- metrics=1 |}]

let%expect_test "unavailable outranks mismatch and cannot pass" =
  let zeros = F.f64 [ 0.0; 0.0 ] in
  (* equal zero vectors pass the element rule, yet the required cosine is
     undefined: the comparison cannot establish agreement *)
  compare_ "required cosine undefined"
    (F.policy ~element:(F.floating ())
       ~metrics:[ cos (Required (At_least 0.9)) ]
       ())
    zeros zeros;
  compare_ "unavailable and mismatching"
    (F.policy ~element:(F.floating ())
       ~metrics:[ cos (Required (At_least 0.9)) ]
       ())
    zeros
    (F.f64 [ 0.0; 1.0 ]);
  compare_ "diagnostic cosine undefined is harmless"
    (F.policy ~element:(F.floating ()) ~metrics:[ cos Diagnostic_only ] ())
    zeros zeros;
  compare_ "required metric on a nonfinite operand"
    (F.policy
       ~element:(F.floating ~nan:Both_nan ())
       ~metrics:[ rmse (Required (At_most 1.0)) ]
       ())
    (F.f64 [ Float.nan; 1.0 ])
    (F.f64 [ Float.nan; 1.0 ]);
  [%expect
    {|
    required cosine undefined: metric_unavailable checked=2/2 element-violations=0 metrics=1
    unavailable and mismatching: metric_unavailable checked=2/2 element-violations=1 metrics=1
    diagnostic cosine undefined is harmless: pass checked=2/2 element-violations=0 metrics=1
    required metric on a nonfinite operand: metric_unavailable checked=2/2 element-violations=0 metrics=1 |}]

let%expect_test "metadata differences are never zero-element successes" =
  let p = F.policy ~element:(F.floating ()) () in
  compare_ "different shape" p (F.f64 [ 1.0; 2.0 ]) (F.f64 [ 1.0; 2.0; 3.0 ]);
  compare_ "different dtype" p (F.f64 [ 1.0 ]) (F.f32 [ 1.0 ]);
  compare_ "genuine empty tensors" p (F.f64 []) (F.f64 []);
  [%expect
    {|
    different shape: metadata_error checked=0/2 element-violations=- metrics=0
    different dtype: metadata_error checked=0/1 element-violations=- metrics=0
    genuine empty tensors: pass checked=0/0 element-violations=0 metrics=0 |}]

let%expect_test "integer and boolean sites require exactness" =
  let ints = F.i64 [ 1L; 2L; 3L ] in
  compare_ "exact equal" (F.policy ~element:(Some Exact_integer) ()) ints ints;
  compare_ "exact differs"
    (F.policy ~element:(Some Exact_integer) ())
    ints
    (F.i64 [ 1L; 2L; 4L ]);
  compare_ "no element rule on integers"
    (F.policy
       ~metrics:[ metric "m" Mean_absolute_error (Required (At_most 1.0)) ]
       ())
    ints ints;
  compare_ "float rule on integers"
    (F.policy ~element:(F.floating ()) ())
    ints ints;
  compare_ "exact rule on floats"
    (F.policy ~element:(Some Exact_integer) ())
    (F.f64 [ 1.0 ]) (F.f64 [ 1.0 ]);
  [%expect
    {|
    exact equal: pass checked=3/3 element-violations=0 metrics=0
    exact differs: mismatch checked=3/3 element-violations=1 metrics=0
    no element rule on integers: CONFIG codec i64 requires an exact element rule
    float rule on integers: CONFIG element rule does not fit codec i64
    exact rule on floats: CONFIG element rule does not fit codec f64 |}]

let%expect_test "policy validation" =
  let l = F.f64 [ 1.0; 2.0 ] in
  compare_ "no required check"
    (F.policy ~metrics:[ rmse Diagnostic_only ] ())
    l l;
  compare_ "nothing at all" (F.policy ()) l l;
  compare_ "scale differs from the element rule"
    (F.policy
       ~element:(F.floating ~scale:Left_scale ())
       ~relative_scale:Right_scale ())
    l l;
  compare_ "metric id reused with another threshold"
    (F.policy ~element:(F.floating ())
       ~metrics:[ rmse (Required (At_most 1.0)); rmse (Required (At_most 2.0)) ]
       ())
    l l;
  compare_ "identical duplicate evaluated once"
    (F.policy ~element:(F.floating ())
       ~metrics:[ rmse (Required (At_most 1.0)); rmse (Required (At_most 1.0)) ]
       ())
    l l;
  compare_ "registered element comparator without an implementation"
    (F.policy
       ~element:
         (Some
            (Registered
               {
                 comparator =
                   Codec_fixtures.text (Id.Comparator.of_string "custom");
                 policy_digest = pdigest;
               }))
       ~metrics:[ rmse (Required (At_most 1.0)) ]
       ())
    l l;
  [%expect
    {|
    no required check: CONFIG policy has no required check
    nothing at all: CONFIG policy has no required check
    scale differs from the element rule: CONFIG relative scale differs from the element rule's scale
    metric id reused with another threshold: CONFIG metric id rmse reused with a different specification
    identical duplicate evaluated once: pass checked=2/2 element-violations=0 metrics=1
    registered element comparator without an implementation: unsupported checked=0/2 element-violations=- metrics=1 |}]

let%expect_test "per-group coverage cannot be narrowed" =
  (* token 1 is undefined (zero vectors); tokens 0 and 2 pass. A required
     per-token cosine must not pass on the two scored groups. *)
  let shape = [ 3L; 2L ] in
  let l = F.f64_shaped shape [ 1.; 0.; 0.; 0.; 1.; 0. ]
  and r = F.f64_shaped shape [ 1.; 0.; 0.; 0.; 1.; 0. ] in
  compare_ "one group unscored"
    (F.policy ~element:(F.floating ())
       ~metrics:
         [
           metric
             ~partition:(Reduce_axes [ Option.get (Tensor_axis.of_index 1) ])
             "cos"
             (Cosine_similarity { zero_norm = Undefined })
             (Required (At_least 0.9));
         ]
       ())
    l r;
  compare_ "same with zero norms declared equal"
    (F.policy ~element:(F.floating ())
       ~metrics:
         [
           metric
             ~partition:(Reduce_axes [ Option.get (Tensor_axis.of_index 1) ])
             "cos"
             (Cosine_similarity { zero_norm = Both_zero_equal })
             (Required (At_least 0.9));
         ]
       ())
    l r;
  [%expect
    {|
    one group unscored: metric_unavailable checked=6/6 element-violations=0 metrics=1
    same with zero norms declared equal: pass checked=6/6 element-violations=0 metrics=1 |}]
