open Inference_compare_schema
module M = Inference_compare_metrics
module F = Metric_fixtures
module T = M.Tensor_metrics

let metric ?(partition = Tensor_metric.Whole_tensor)
    ?(role = Tensor_metric.Diagnostic_only) id kind : Tensor_metric.t =
  {
    id = Codec_fixtures.text (Id.Tensor_metric.of_string id);
    kind;
    partition;
    role;
  }

let axes l =
  Tensor_metric.Reduce_axes
    (List.map (fun i -> Option.get (Tensor_axis.of_index i)) l)

let show_result name (r : Metric_result.t) =
  let st =
    match r.status with
    | Complete -> "complete"
    | Undefined -> "undefined"
    | Unsupported -> "unsupported"
    | Nonfinite -> "nonfinite"
    | Resource_failure -> "resource"
    | Numerical_failure -> "numerical"
  in
  let em : Error_metric.t -> string = function
    | No_finite_value -> "-"
    | Finite_value v -> Printf.sprintf "%.6g" v
    | Overflow -> "overflow"
  in
  Format.printf
    "%s: %s groups %Ld/%Ld min=%s max=%s violations=%s samples=[%s]@." name st
    r.groups_checked r.groups_expected (em r.minimum) (em r.maximum)
    (match r.violations with Some v -> Int64.to_string v | None -> "-")
    (String.concat "; "
       (List.map
          (fun (c, v) ->
            Printf.sprintf "(%s)=%.4g"
              (String.concat "," (List.map Int64.to_string c))
              v)
          r.samples))

let run ?(env = T.empty_env) name m left right =
  match T.compute F.ctx env m ~left ~right with
  | Ok r -> show_result name r
  | Error e -> Format.printf "%s: ERROR %a@." name T.pp_error (Err.Error.kind e)

let kind_cases =
  [
    ("mae", Tensor_metric.Mean_absolute_error);
    ("rmse", Root_mean_square_error);
    ("bias", Absolute_bias Right_minus_left);
    ( "std",
      Error_standard_deviation
        { direction = Right_minus_left; variance = Population } );
    ("l2", Normalized_l2 { scale = Left_scale; zero_norm = Undefined });
    ("cos", Cosine_similarity { zero_norm = Undefined });
  ]

let%expect_test "metrics on hand-derived inputs" =
  let left = F.f64 [ 1.0; 2.0; 3.0; 4.0 ] in
  let cases name right =
    Format.printf "-- %s@." name;
    List.iter
      (fun (id, k) -> run id (metric id k) left (F.f64 right))
      kind_cases
  in
  cases "equal" [ 1.0; 2.0; 3.0; 4.0 ];
  cases "constant offset +1" [ 2.0; 3.0; 4.0; 5.0 ];
  cases "opposing errors" [ 2.0; 1.0; 4.0; 3.0 ];
  cases "scale by 10" [ 10.0; 20.0; 30.0; 40.0 ];
  [%expect
    {|
    -- equal
    mae: complete groups 1/1 min=0 max=0 violations=- samples=[()=0]
    rmse: complete groups 1/1 min=0 max=0 violations=- samples=[()=0]
    bias: complete groups 1/1 min=0 max=0 violations=- samples=[()=0]
    std: complete groups 1/1 min=0 max=0 violations=- samples=[()=0]
    l2: complete groups 1/1 min=0 max=0 violations=- samples=[()=0]
    cos: complete groups 1/1 min=1 max=1 violations=- samples=[()=1]
    -- constant offset +1
    mae: complete groups 1/1 min=1 max=1 violations=- samples=[()=1]
    rmse: complete groups 1/1 min=1 max=1 violations=- samples=[()=1]
    bias: complete groups 1/1 min=1 max=1 violations=- samples=[()=1]
    std: complete groups 1/1 min=0 max=0 violations=- samples=[()=0]
    l2: complete groups 1/1 min=0.365148 max=0.365148 violations=- samples=[()=0.3651]
    cos: complete groups 1/1 min=0.993808 max=0.993808 violations=- samples=[()=0.9938]
    -- opposing errors
    mae: complete groups 1/1 min=1 max=1 violations=- samples=[()=1]
    rmse: complete groups 1/1 min=1 max=1 violations=- samples=[()=1]
    bias: complete groups 1/1 min=0 max=0 violations=- samples=[()=0]
    std: complete groups 1/1 min=1 max=1 violations=- samples=[()=1]
    l2: complete groups 1/1 min=0.365148 max=0.365148 violations=- samples=[()=0.3651]
    cos: complete groups 1/1 min=0.933333 max=0.933333 violations=- samples=[()=0.9333]
    -- scale by 10
    mae: complete groups 1/1 min=22.5 max=22.5 violations=- samples=[()=22.5]
    rmse: complete groups 1/1 min=24.6475 max=24.6475 violations=- samples=[()=24.65]
    bias: complete groups 1/1 min=22.5 max=22.5 violations=- samples=[()=22.5]
    std: complete groups 1/1 min=10.0623 max=10.0623 violations=- samples=[()=10.06]
    l2: complete groups 1/1 min=9 max=9 violations=- samples=[()=9]
    cos: complete groups 1/1 min=1 max=1 violations=- samples=[()=1] |}]

let%expect_test
    "a constant offset has bias but no spread, opposing errors the reverse" =
  let left = F.f64 [ 0.0; 0.0 ] in
  run "offset: bias"
    (metric "b" (Absolute_bias Right_minus_left))
    left
    (F.f64 [ 1.0; 1.0 ]);
  run "offset: std"
    (metric "s"
       (Error_standard_deviation
          { direction = Right_minus_left; variance = Population }))
    left
    (F.f64 [ 1.0; 1.0 ]);
  run "opposing: bias"
    (metric "b" (Absolute_bias Right_minus_left))
    left
    (F.f64 [ -1.0; 1.0 ]);
  run "opposing: std"
    (metric "s"
       (Error_standard_deviation
          { direction = Right_minus_left; variance = Population }))
    left
    (F.f64 [ -1.0; 1.0 ]);
  [%expect
    {|
    offset: bias: complete groups 1/1 min=1 max=1 violations=- samples=[()=1]
    offset: std: complete groups 1/1 min=0 max=0 violations=- samples=[()=0]
    opposing: bias: complete groups 1/1 min=0 max=0 violations=- samples=[()=0]
    opposing: std: complete groups 1/1 min=1 max=1 violations=- samples=[()=1] |}]

let%expect_test "per-group thresholds are not hidden by a global average" =
  (* [batch=1; token=3; feature=2]: tokens 0 and 2 identical, token 1 opposite. *)
  let shape = [ 1L; 3L; 2L ] in
  let left = F.f64_shaped shape [ 1.0; 0.0; 1.0; 0.0; 1.0; 0.0 ] in
  let right = F.f64_shaped shape [ 1.0; 0.0; -1.0; 0.0; 1.0; 0.0 ] in
  let cos partition role =
    metric ~partition ~role "cos" (Cosine_similarity { zero_norm = Undefined })
  in
  run "whole tensor" (cos Whole_tensor (Required (At_least 0.5))) left right;
  run "per token (reduce feature)"
    (cos (axes [ 2 ]) (Required (At_least 0.5)))
    left right;
  run "per feature (reduce token)" (cos (axes [ 1 ]) Diagnostic_only) left right;
  run "reduce [1;2]" (cos (axes [ 1; 2 ]) Diagnostic_only) left right;
  run "reduce [2;1], listed order is irrelevant"
    (cos (axes [ 2; 1 ]) Diagnostic_only)
    left right;
  (* a different reduced set groups differently: mixed axis order [2;0] on a
     [2;2;2] tensor gives one group per middle index *)
  let cube = [ 2L; 2L; 2L ] in
  let l = F.f64_shaped cube [ 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. ]
  and r = F.f64_shaped cube [ 1.; 2.; 3.; 4.; 5.; 6.; 7.; 9. ] in
  run "groups follow complementary axes"
    (metric ~partition:(axes [ 2; 0 ]) "m" Mean_absolute_error)
    l r;
  [%expect
    {|
    whole tensor: complete groups 1/1 min=0.333333 max=0.333333 violations=1 samples=[()=0.3333]
    per token (reduce feature): complete groups 3/3 min=-1 max=1 violations=1 samples=[(0,0)=1; (0,1)=-1; (0,2)=1]
    per feature (reduce token): undefined groups 1/2 min=0.333333 max=0.333333 violations=- samples=[(0,0)=0.3333]
    reduce [1;2]: complete groups 1/1 min=0.333333 max=0.333333 violations=- samples=[(0)=0.3333]
    reduce [2;1], listed order is irrelevant: complete groups 1/1 min=0.333333 max=0.333333 violations=- samples=[(0)=0.3333]
    groups follow complementary axes: complete groups 2/2 min=0 max=0.25 violations=- samples=[(0)=0; (1)=0.25] |}]

let%expect_test "cosine is scale invariant but the element error remains" =
  let left = F.f64 [ 1.0; 2.0 ] and right = F.f64 [ 10.0; 20.0 ] in
  run "cosine"
    (metric "c" (Cosine_similarity { zero_norm = Undefined }))
    left right;
  let p =
    F.policy ~element:(F.floating ~atol:1e-9 ())
      ~metrics:
        [
          metric "c"
            (Cosine_similarity { zero_norm = Undefined })
            ~role:(Required (At_least 0.999));
        ]
      ()
  in
  (match
     M.Comparison.compare F.ctx T.empty_env p
       ~comparison_id:(Codec_fixtures.text (Id.Comparison.of_string "cmp"))
       ~policy_digest:
         (Result.get_ok
            (Digest.Comparison_policy.make Plan_fixtures.sha256
               ~hex:(String.make 64 '0')))
       ~left ~right
   with
  | Ok r ->
      Format.printf
        "verdict with element rule: %s (element violations %a, cosine %s)@."
        (match r.verdict with
        | Pass -> "pass"
        | Mismatch -> "mismatch"
        | _ -> "other")
        (Fmt.option Fmt.int64)
        (Option.bind r.element (fun e -> e.tolerance_violations))
        (match
           T.satisfied
             (metric "c"
                (Cosine_similarity { zero_norm = Undefined })
                ~role:(Required (At_least 0.999)))
             (List.hd r.tensor_metrics)
         with
        | Some true -> "satisfied"
        | _ -> "?")
  | Error e ->
      Format.printf "ERROR %a@." M.Comparison.pp_error (Err.Error.kind e));
  [%expect
    {|
    cosine: complete groups 1/1 min=1 max=1 violations=- samples=[()=1]
    verdict with element rule: mismatch (element violations 2, cosine satisfied) |}]

let%expect_test "zero norms, undefined groups and nonfinite operands" =
  let c z = metric "c" (Cosine_similarity { zero_norm = z }) in
  let zeros = F.f64 [ 0.0; 0.0 ] and ones = F.f64 [ 1.0; 1.0 ] in
  run "both zero, undefined" (c Undefined) zeros zeros;
  run "both zero, equal" (c Both_zero_equal) zeros zeros;
  run "one zero, equal policy" (c Both_zero_equal) zeros ones;
  let l2 z =
    metric "l2" (Normalized_l2 { scale = Left_scale; zero_norm = z })
  in
  run "l2 zero denominator, zero residual, equal" (l2 Both_zero_equal) zeros
    zeros;
  run "l2 zero denominator, nonzero residual" (l2 Both_zero_equal) zeros ones;
  run "nan operand"
    (metric "m" Mean_absolute_error)
    (F.f64 [ 1.0; Float.nan ])
    ones;
  run "inf operand"
    (metric "m" Mean_absolute_error)
    (F.f64 [ 1.0; Float.infinity ])
    ones;
  run "empty group" (metric "m" Mean_absolute_error) (F.f64 []) (F.f64 []);
  run "sample std of one element"
    (metric "s"
       (Error_standard_deviation
          { direction = Right_minus_left; variance = Sample }))
    (F.f64 [ 1.0 ]) (F.f64 [ 2.0 ]);
  [%expect
    {|
    both zero, undefined: undefined groups 0/1 min=- max=- violations=- samples=[]
    both zero, equal: complete groups 1/1 min=1 max=1 violations=- samples=[()=1]
    one zero, equal policy: undefined groups 0/1 min=- max=- violations=- samples=[]
    l2 zero denominator, zero residual, equal: complete groups 1/1 min=0 max=0 violations=- samples=[()=0]
    l2 zero denominator, nonzero residual: undefined groups 0/1 min=- max=- violations=- samples=[]
    nan operand: nonfinite groups 0/1 min=- max=- violations=- samples=[]
    inf operand: nonfinite groups 0/1 min=- max=- violations=- samples=[]
    empty group: undefined groups 0/1 min=- max=- violations=- samples=[]
    sample std of one element: undefined groups 0/1 min=- max=- violations=- samples=[] |}]

let%expect_test "overflow is reported, not clamped" =
  let big = Float.max_float in
  run "mae overflow"
    (metric "m" Mean_absolute_error)
    (F.f64 [ big; big ])
    (F.f64 [ -.big; -.big ]);
  (* squares would overflow; the scaled recurrence still gets RMSE of 1e200 *)
  run "rmse of large finite residuals"
    (metric "r" Root_mean_square_error)
    (F.f64 [ 0.0; 0.0 ])
    (F.f64 [ 1e200; 1e200 ]);
  run "cosine of huge vectors"
    (metric "c" (Cosine_similarity { zero_norm = Undefined }))
    (F.f64 [ big; big ])
    (F.f64 [ big; big ]);
  [%expect
    {|
    mae overflow: numerical groups 0/1 min=- max=- violations=- samples=[]
    rmse of large finite residuals: complete groups 1/1 min=1e+200 max=1e+200 violations=- samples=[()=1e+200]
    cosine of huge vectors: complete groups 1/1 min=1 max=1 violations=- samples=[()=1] |}]

let%expect_test "distribution metrics" =
  let w = metric "w" (Wasserstein_1d Empirical_values) in
  run "permutation of the same values" w
    (F.f64 [ 1.0; 2.0; 3.0 ])
    (F.f64 [ 3.0; 1.0; 2.0 ]);
  run "shifted values" w (F.f64 [ 1.0; 2.0; 3.0 ]) (F.f64 [ 2.0; 3.0; 4.0 ]);
  Format.printf "element-wise on the permutation: ";
  run "mae"
    (metric "m" Mean_absolute_error)
    (F.f64 [ 1.0; 2.0; 3.0 ])
    (F.f64 [ 3.0; 1.0; 2.0 ]);
  let support positions values : Metric_support.t =
    ignore values;
    {
      id = Codec_fixtures.text (Id.Metric_support.of_string "bins");
      payload_digest =
        Result.get_ok
          (Digest.Metric_support.make Plan_fixtures.sha256
             ~hex:(String.make 64 '5'));
      positions;
    }
  in
  let hist ?(mass = Tensor_metric.Normalize_positive_total) positions =
    metric "h"
      (Wasserstein_1d
         (Histogram_masses { support = support positions (); mass }))
  in
  let env xs : T.env =
    { registry = M.Metric_registry.empty; supports = (fun _ -> Some xs) }
  in
  let left = F.f64 [ 1.0; 0.0; 0.0 ] and right = F.f64 [ 0.0; 1.0; 0.0 ] in
  run ~env:(env [| 0.0; 1.0; 2.0 |]) "unit spacing moves 1" (hist 3L) left right;
  run
    ~env:(env [| 0.0; 100.0; 200.0 |])
    "support [0,100,200] costs 100" (hist 3L) left right;
  run
    ~env:(env [| 0.0; 1.0; 2.0 |])
    "masses scaled, normalized" (hist 3L)
    (F.f64 [ 2.0; 0.0; 0.0 ])
    (F.f64 [ 0.0; 5.0; 0.0 ]);
  run
    ~env:(env [| 0.0; 1.0; 2.0 |])
    "unit mass required, violated"
    (hist ~mass:(Require_unit_mass { absolute_tolerance = 1e-6 }) 3L)
    (F.f64 [ 2.0; 0.0; 0.0 ])
    right;
  run
    ~env:(env [| 0.0; 1.0; 2.0 |])
    "negative mass" (hist 3L)
    (F.f64 [ 2.0; -1.0; 0.0 ])
    right;
  run
    ~env:(env [| 0.0; 1.0; 2.0 |])
    "zero total" (hist 3L)
    (F.f64 [ 0.0; 0.0; 0.0 ])
    right;
  run
    ~env:(env [| 0.0; 0.0; 2.0 |])
    "support not strictly increasing" (hist 3L) left right;
  run "support payload unavailable" (hist 3L) left right;
  run
    ~env:(env [| 0.0; 1.0; 2.0 |])
    "declared positions differ from group size" (hist 2L) left right;
  [%expect
    {|
    permutation of the same values: complete groups 1/1 min=0 max=0 violations=- samples=[()=0]
    shifted values: complete groups 1/1 min=1 max=1 violations=- samples=[()=1]
    element-wise on the permutation: mae: complete groups 1/1 min=1.33333 max=1.33333 violations=- samples=[()=1.333]
    unit spacing moves 1: complete groups 1/1 min=1 max=1 violations=- samples=[()=1]
    support [0,100,200] costs 100: complete groups 1/1 min=100 max=100 violations=- samples=[()=100]
    masses scaled, normalized: complete groups 1/1 min=1 max=1 violations=- samples=[()=1]
    unit mass required, violated: undefined groups 0/1 min=- max=- violations=- samples=[]
    negative mass: undefined groups 0/1 min=- max=- violations=- samples=[]
    zero total: undefined groups 0/1 min=- max=- violations=- samples=[]
    support not strictly increasing: undefined groups 0/1 min=- max=- violations=- samples=[]
    support payload unavailable: unsupported groups 0/1 min=- max=- violations=- samples=[]
    declared positions differ from group size: ERROR support has 2 positions, group size is 3 |}]

let%expect_test "configuration faults are caught before running" =
  let spec = Codec_fixtures.f64 [ 2L; 3L ] in
  let check name m =
    match T.validate T.empty_env spec m with
    | Ok () -> Format.printf "%s: valid@." name
    | Error e ->
        Format.printf "%s: %a@." name M.Metric_config_error.pp
          (Err.Error.kind e)
  in
  let cos role =
    metric ~role "c" (Cosine_similarity { zero_norm = Undefined })
  in
  let mae ?(partition = Tensor_metric.Whole_tensor) role =
    metric ~partition ~role "m" Mean_absolute_error
  in
  check "cosine at_least in range" (cos (Required (At_least 0.9)));
  check "cosine at_most" (cos (Required (At_most 0.9)));
  check "cosine bound above 1" (cos (Required (At_least 1.5)));
  check "distance at_least" (mae (Required (At_least 0.1)));
  check "distance at_most" (mae (Required (At_most 0.1)));
  check "negative distance bound" (mae (Required (At_most (-0.1))));
  check "nonfinite bound" (mae (Required (At_most Float.infinity)));
  check "empty axes" (mae ~partition:(axes []) Diagnostic_only);
  check "duplicate axes" (mae ~partition:(axes [ 1; 1 ]) Diagnostic_only);
  check "axis out of range" (mae ~partition:(axes [ 2 ]) Diagnostic_only);
  check "valid axes" (mae ~partition:(axes [ 1 ]) (Required (At_most 0.1)));
  check "unregistered comparator"
    (metric "r"
       (Registered
          {
            comparator = Codec_fixtures.text (Id.Comparator.of_string "emd2d");
            implementation_digest =
              Result.get_ok
                (Digest.Implementation.make Plan_fixtures.sha256
                   ~hex:(String.make 64 '1'));
            configuration_digest =
              Result.get_ok
                (Digest.Comparison_policy.make Plan_fixtures.sha256
                   ~hex:(String.make 64 '2'));
          }));
  (match
     T.validate T.empty_env (Codec_fixtures.i64 [ 2L ]) (mae Diagnostic_only)
   with
  | Error e ->
      Format.printf "integer tensor: %a@." M.Metric_config_error.pp
        (Err.Error.kind e)
  | Ok () -> Format.printf "integer tensor: valid@.");
  [%expect
    {|
    cosine at_least in range: valid
    cosine at_most: threshold direction does not fit the metric
    cosine bound above 1: cosine bound outside [-1, 1]
    distance at_least: threshold direction does not fit the metric
    distance at_most: valid
    negative distance bound: distance bound must be nonnegative
    nonfinite bound: threshold must be finite
    empty axes: empty reduction axis list
    duplicate axes: duplicate axis 1
    axis out of range: axis 2 out of range for rank 2
    valid axes: valid
    unregistered comparator: unregistered comparator emd2d
    integer tensor: metric unsupported for codec i64 |}]

let%expect_test "registered metrics are pinned and budgeted" =
  let sha ch =
    Result.get_ok
      (Digest.Implementation.make Plan_fixtures.sha256 ~hex:(String.make 64 ch))
  in
  let cfg ch =
    Result.get_ok
      (Digest.Comparison_policy.make Plan_fixtures.sha256
         ~hex:(String.make 64 ch))
  in
  let comparator = Codec_fixtures.text (Id.Comparator.of_string "max_abs") in
  let impl : M.Metric_registry.impl =
    {
      comparator;
      implementation_digest = sha '1';
      configuration_digest = cfg '2';
      threshold_direction = At_most;
      max_group_elements = 4;
      compute =
        (fun a b ->
          Score
            (Array.fold_left Float.max 0.0
               (Array.mapi (fun i x -> Float.abs (x -. b.(i))) a)));
    }
  in
  let registry =
    Result.get_ok (M.Metric_registry.register M.Metric_registry.empty impl)
  in
  let env : T.env = { registry; supports = (fun _ -> None) } in
  let kind ?(i = '1') ?(c = '2') () =
    Tensor_metric.Registered
      {
        comparator;
        implementation_digest = sha i;
        configuration_digest = cfg c;
      }
  in
  run ~env "pinned"
    (metric "r" (kind ()))
    (F.f64 [ 1.0; 5.0 ])
    (F.f64 [ 2.0; 2.0 ]);
  run ~env "other implementation digest"
    (metric "r" (kind ~i:'9' ()))
    (F.f64 [ 1.0 ]) (F.f64 [ 2.0 ]);
  run ~env "other configuration digest"
    (metric "r" (kind ~c:'9' ()))
    (F.f64 [ 1.0 ]) (F.f64 [ 2.0 ]);
  run ~env "over the group budget"
    (metric "r" (kind ()))
    (F.f64 [ 1.; 2.; 3.; 4.; 5. ])
    (F.f64 [ 1.; 2.; 3.; 4.; 5. ]);
  (match M.Metric_registry.register registry impl with
  | Error _ -> Format.printf "duplicate registration refused@."
  | Ok _ -> Format.printf "duplicate accepted@.");
  [%expect
    {|
    pinned: complete groups 1/1 min=3 max=3 violations=- samples=[()=3]
    other implementation digest: unsupported groups 0/1 min=- max=- violations=- samples=[]
    other configuration digest: unsupported groups 0/1 min=- max=- violations=- samples=[]
    over the group budget: resource groups 0/1 min=- max=- violations=- samples=[]
    duplicate registration refused |}]

let%expect_test "group element order is canonical whatever order the axes are listed" =
  let show name partition =
    match M.Tensor_groups.make ~shape:[ 2L; 2L; 2L ] partition with
    | Error e -> Format.printf "%s: %a@." name M.Metric_config_error.pp (Err.Error.kind e)
    | Ok g ->
        Format.printf "%s: %d groups of %d:" name (M.Tensor_groups.groups g)
          (M.Tensor_groups.group_size g);
        for i = 0 to M.Tensor_groups.groups g - 1 do
          Format.printf " (%s)->[%s]"
            (String.concat "," (List.map Int64.to_string (M.Tensor_groups.coords g i)))
            (String.concat ","
               (List.map string_of_int (Array.to_list (M.Tensor_groups.indices g i))))
        done;
        Format.printf "@."
  in
  show "reduce [0;2]" (axes [ 0; 2 ]);
  show "reduce [2;0]" (axes [ 2; 0 ]);
  show "reduce [1]" (axes [ 1 ]);
  show "whole tensor" Whole_tensor;
  show "zero extent" Whole_tensor;
  (match M.Tensor_groups.make ~shape:[ 3L; 0L ] (axes [ 1 ]) with
  | Ok g -> Format.printf "zero extent reduced: %d groups of %d@." (M.Tensor_groups.groups g) (M.Tensor_groups.group_size g)
  | Error _ -> Format.printf "zero extent reduced: error@.");
  [%expect {|
    reduce [0;2]: 2 groups of 4: (0)->[0,1,4,5] (1)->[2,3,6,7]
    reduce [2;0]: 2 groups of 4: (0)->[0,1,4,5] (1)->[2,3,6,7]
    reduce [1]: 4 groups of 2: (0,0)->[0,2] (0,1)->[1,3] (1,0)->[4,6] (1,1)->[5,7]
    whole tensor: 1 groups of 8: ()->[0,1,2,3,4,5,6,7]
    zero extent: 1 groups of 8: ()->[0,1,2,3,4,5,6,7]
    zero extent reduced: 3 groups of 0 |}]
