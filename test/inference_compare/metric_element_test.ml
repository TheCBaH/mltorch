open Inference_compare_schema
module M = Inference_compare_metrics
module F = Metric_fixtures

let nan32 = 0x7FC00000L

let%expect_test "statistics follow the hand-derived table" =
  (* Residual = right - left. *)
  let show name residuals =
    let left = F.f64 (List.map (fun _ -> 0.0) residuals) in
    let right = F.f64 residuals in
    let r = F.element (F.policy ()) ~left ~right in
    Format.printf "%s: %a nonzero=%Ld@." name F.pp_stats r.statistics
      (Int64.add r.statistics.positive_errors r.statistics.negative_errors)
  in
  show "[1,1]" [ 1.0; 1.0 ];
  show "[-1,1]" [ -1.0; 1.0 ];
  show "[0,0,6]" [ 0.0; 0.0; 6.0 ];
  (* sqrt 8 = 2.82843, sqrt 12 = 3.4641 *)
  [%expect
    {|
    [1,1]: pairs=2 +2 -0 0:0 bias=1 mae=1 rmse=1 std=0 nonzero=2
    [-1,1]: pairs=2 +1 -1 0:0 bias=0 mae=1 rmse=1 std=1 nonzero=2
    [0,0,6]: pairs=3 +1 -0 0:2 bias=2 mae=2 rmse=3.4641 std=2.82843 nonzero=1 |}]

let%expect_test "direction names the sign, not the donor" =
  let left = F.f64 [ 1.0; 2.0; 3.0 ] and right = F.f64 [ 2.0; 2.0; 1.0 ] in
  let run direction =
    let r = F.element (F.policy ~direction ()) ~left ~right in
    Format.printf "%s: +%Ld -%Ld bias=%a min=%a@."
      (match direction with
      | Right_minus_left -> "right-left"
      | Left_minus_right -> "left-right")
      r.statistics.positive_errors r.statistics.negative_errors F.pp_scalar
      r.statistics.bias
      (Fmt.option (fun ppf (e : Difference_statistics.extremum) ->
           F.pp_scalar ppf e.value))
      r.statistics.minimum_signed_error
  in
  run Right_minus_left;
  run Left_minus_right;
  [%expect
    {|
    right-left: +1 -1 bias=-0.333333 min=-2
    left-right: +1 -1 bias=0.333333 min=-1 |}]

let%expect_test "sample and population conventions, n = 0 and n = 1" =
  let run variance values =
    let left = F.f64 (List.map (fun _ -> 0.0) values) in
    let r = F.element (F.policy ~variance ()) ~left ~right:(F.f64 values) in
    Format.printf "%s n=%d std=%a bias=%a@."
      (match variance with
      | Difference_statistics.Population -> "pop"
      | Sample -> "sample")
      (List.length values) F.pp_scalar r.statistics.standard_deviation
      F.pp_scalar r.statistics.bias
  in
  run Population [ 1.0 ];
  run Sample [ 1.0 ];
  run Sample [ 1.0; 3.0 ];
  run Population [ 1.0; 3.0 ];
  (* an empty tensor is a complete zero-element comparison with no statistics *)
  run Population [];
  [%expect
    {|
    pop n=1 std=0 bias=1
    sample n=1 std=n/a(insufficient) bias=1
    sample n=2 std=1.41421 bias=2
    pop n=2 std=1 bias=2
    pop n=0 std=n/a(empty) bias=n/a(empty) |}]

let%expect_test "extrema keep the first row-major coordinate on ties" =
  let left = F.f64_shaped [ 2L; 2L ] [ 0.; 0.; 0.; 0. ]
  and right = F.f64_shaped [ 2L; 2L ] [ 1.; -1.; 1.; -1. ] in
  let r = F.element (F.policy ()) ~left ~right in
  let at = function
    | Some ({ index; _ } : Difference_statistics.extremum) ->
        Format.asprintf "%a" Fmt.(list ~sep:comma int64) index
    | None -> "-"
  in
  Format.printf "max at (%s) min at (%s) largest |e| at (%s)@."
    (at r.statistics.maximum_signed_error)
    (at r.statistics.minimum_signed_error)
    (at r.statistics.largest_absolute_error);
  [%expect
    {|
    max at (0,
    0) min at (0,
    1) largest |e| at (0,
    0) |}]

let%expect_test "exceptional operands are classified before arithmetic" =
  let n32 =
    0x7FC00001L
    (* different payload *)
  in
  let left =
    F.f32_of_bits
      [
        nan32;
        nan32;
        nan32;
        0x7F800000L;
        0x7F800000L;
        0xFF800000L;
        0x3F800000L;
        0x00000000L;
      ]
  and right =
    F.f32_of_bits
      [
        nan32;
        n32;
        0x3F800000L;
        0x7F800000L;
        0xFF800000L;
        0x3F800000L;
        0x7F800000L;
        0x80000000L;
      ]
  in
  let r = F.element (F.policy ()) ~left ~right in
  let nf = r.nonfinite in
  Format.printf
    "nan same=%Ld diff=%Ld left-only=%Ld right-only=%Ld | inf same=%Ld \
     opposite=%Ld left/finite=%Ld right/finite=%Ld | -0=%Ld@."
    nf.both_nan_same_bits nf.both_nan_different_bits nf.left_nan_only
    nf.right_nan_only nf.same_sign_infinities nf.opposite_sign_infinities
    nf.left_infinite_right_finite nf.right_infinite_left_finite
    nf.signed_zero_differences;
  Format.printf
    "finite=%Ld nan_pairs=%Ld inf_pairs=%Ld excluded=%Ld total=%Ld@."
    r.finite_pairs r.nan_pairs r.infinity_pairs
    r.statistics.excluded_nonfinite_pairs
    (Int64.add r.finite_pairs (Int64.add r.nan_pairs r.infinity_pairs));
  Format.printf
    "absolute: finite=%Ld nonfinite=%Ld; relative: finite=%Ld 0/0=%Ld@."
    r.absolute.finite_values r.absolute.nonfinite_operands
    r.relative.finite_values r.relative.zero_over_zero;
  Format.printf "bitwise mismatches=%Ld@." r.bitwise_mismatches;
  [%expect
    {|
    nan same=1 diff=1 left-only=1 right-only=0 | inf same=1 opposite=1 left/finite=1 right/finite=1 | -0=1
    finite=1 nan_pairs=3 inf_pairs=4 excluded=7 total=8
    absolute: finite=1 nonfinite=7; relative: finite=0 0/0=1
    bitwise mismatches=6 |}]

let%expect_test "nan and infinity acceptance by policy" =
  let left = F.f32_of_bits [ nan32; nan32; 0x7F800000L; 0x7F800000L ]
  and right = F.f32_of_bits [ nan32; 0x7FC00001L; 0x7F800000L; 0xFF800000L ] in
  let run name ?required nan =
    let p = F.policy ~element:(F.floating ?required ~nan ()) () in
    let r = F.element p ~left ~right in
    Format.printf "%s: violations=%a first=%a@." name (Fmt.option Fmt.int64)
      r.tolerance_violations
      (Fmt.option Fmt.(list ~sep:comma int64))
      r.first_violation_index
  in
  run "reject" Reject;
  run "both_nan" Both_nan;
  run "same_bits" Same_bits;
  (* Bitwise still rejects differing NaN payloads, and Reject still rejects
     identical NaN bits. *)
  run "bitwise+both_nan" ~required:Bitwise Both_nan;
  run "bitwise+reject" ~required:Bitwise Reject;
  [%expect
    {|
    reject: violations=3 first=0
    both_nan: violations=1 first=3
    same_bits: violations=2 first=1
    bitwise+both_nan: violations=2 first=1
    bitwise+reject: violations=3 first=0 |}]

let%expect_test "signed zeros" =
  let left = F.f32_of_bits [ 0x00000000L; 0x80000000L ]
  and right = F.f32_of_bits [ 0x80000000L; 0x80000000L ] in
  let run name zero required =
    let r =
      F.element
        (F.policy ~element:(F.floating ~signed_zero:zero ~required ()) ())
        ~left ~right
    in
    Format.printf "%s: violations=%a bitwise=%Ld zero-diff=%Ld ulp=%s@." name
      (Fmt.option Fmt.int64) r.tolerance_violations r.bitwise_mismatches
      r.nonfinite.signed_zero_differences
      (match r.max_ulp_distance with
      | Some u -> Ulp_distance.to_decimal_string u
      | None -> "-")
  in
  run "equal/tolerance" Equal Tolerance;
  run "distinct/tolerance" Distinct Tolerance;
  run "equal/bitwise" Equal Bitwise;
  [%expect
    {|
    equal/tolerance: violations=0 bitwise=1 zero-diff=1 ulp=0
    distinct/tolerance: violations=1 bitwise=1 zero-diff=1 ulp=1
    equal/bitwise: violations=1 bitwise=1 zero-diff=1 ulp=0 |}]

let%expect_test "absolute and relative measurements" =
  let show name scale left right =
    let p = F.policy ~relative_scale:scale () in
    let r =
      F.element ~max_samples:4 p ~left:(F.f64 [ left ]) ~right:(F.f64 [ right ])
    in
    match r.difference_samples with
    | s :: _ ->
        Format.printf "%s: abs=%a rel=%a@." name F.pp_mag s.absolute F.pp_mag
          s.relative
    | [] -> Format.printf "%s: identical@." name
  in
  show "left scale 2 vs 3" Left_scale 2.0 3.0;
  show "right scale 2 vs 3" Right_scale 2.0 3.0;
  show "symmetric 2 vs 3" Symmetric_max_scale 2.0 3.0;
  show "zero reference, nonzero" Left_scale 0.0 1.0;
  show "tiny divisor is not floored" Left_scale 1e-300 1.0;
  let big = Float.max_float in
  show "difference overflows, ratio recoverable" Symmetric_max_scale big (-.big);
  show "both overflow" Left_scale 1e-300 big;
  (* identical zeros: 0/0 *)
  let r =
    F.element (F.policy ()) ~left:(F.f64 [ 0.0 ]) ~right:(F.f64 [ 0.0 ])
  in
  Format.printf "0 vs 0: rel 0/0 count=%Ld abs finite=%Ld@."
    r.relative.zero_over_zero r.absolute.finite_values;
  [%expect
    {|
    left scale 2 vs 3: abs=1 rel=0.5
    right scale 2 vs 3: abs=1 rel=0.333333
    symmetric 2 vs 3: abs=1 rel=0.333333
    zero reference, nonzero: abs=1 rel=unbounded
    tiny divisor is not floored: abs=1 rel=1e+300
    difference overflows, ratio recoverable: abs=overflow rel=2
    both overflow: abs=1.79769e+308 rel=overflow
    0 vs 0: rel 0/0 count=1 abs finite=1 |}]

let%expect_test "ulp distance" =
  let show name l r =
    let res =
      F.element (F.policy ()) ~left:(F.f32_of_bits [ l ])
        ~right:(F.f32_of_bits [ r ])
    in
    Format.printf "%s: %s@." name
      (match res.max_ulp_distance with
      | Some u -> Ulp_distance.to_decimal_string u
      | None -> "-")
  in
  show "1.0 vs next" 0x3F800000L 0x3F800001L;
  show "across zero (-tiny, +tiny)" 0x80000001L 0x00000001L;
  show "max vs -max" 0x7F7FFFFFL 0xFF7FFFFFL;
  show "nan has none" nan32 0x3F800000L;
  let wide l r =
    let mk b = F.f64 [ Int64.float_of_bits b ] in
    let res = F.element (F.policy ()) ~left:(mk l) ~right:(mk r) in
    match res.max_ulp_distance with
    | Some u -> Ulp_distance.to_decimal_string u
    | None -> "-"
  in
  Format.printf "f64 max vs -max: %s@."
    (wide 0x7FEFFFFFFFFFFFFFL 0xFFEFFFFFFFFFFFFFL);
  [%expect
    {|
    1.0 vs next: 1
    across zero (-tiny, +tiny): 3
    max vs -max: 4278190079
    nan has none: -
    f64 max vs -max: 18437736874454810623 |}]

let%expect_test "exact integers never go through floating point" =
  let big = Int64.add (Int64.shift_left 1L 53) 1L in
  let left = F.i64 [ big; 1L; Int64.max_int ]
  and right = F.i64 [ Int64.shift_left 1L 53; 1L; Int64.max_int ] in
  let r = F.element (F.policy ~element:(Some Exact_integer) ()) ~left ~right in
  Format.printf "mismatches=%Ld violations=%a first=%a stats=%s@."
    r.bitwise_mismatches (Fmt.option Fmt.int64) r.tolerance_violations
    (Fmt.option Fmt.(list ~sep:comma int64))
    r.first_violation_index
    (match r.statistics.status with
    | Not_supported -> "not_supported"
    | _ -> "?");
  Format.printf "direction counts: +%Ld -%Ld 0:%Ld@."
    r.statistics.positive_errors r.statistics.negative_errors
    r.statistics.zero_errors;
  (match
     M.Element_compare.compare F.ctx
       (F.policy ~element:(F.floating ()) ())
       ~left ~right
   with
  | Error e ->
      Format.printf "float rule on integers: %a@." F.pp_error (Err.Error.kind e)
  | Ok _ -> Format.printf "float rule on integers: accepted@.");
  [%expect
    {|
    mismatches=1 violations=1 first=0 stats=not_supported
    direction counts: +0 -1 0:2
    float rule on integers: element policy does not fit the codec |}]

let%expect_test "spec mismatch is not a comparison" =
  (match
     M.Element_compare.compare F.ctx (F.policy ()) ~left:(F.f64 [ 1.0 ])
       ~right:(F.f32 [ 1.0 ])
   with
  | Error e -> Format.printf "%a@." F.pp_error (Err.Error.kind e)
  | Ok _ -> Format.printf "accepted@.");
  [%expect {| left and right specs differ |}]

(* The published producer rule, as reimplemented by Pt2_fixture.Compare, must
   agree element for element with the generic rule configured the same way
   (binary32, reference on the right, NaN rejected, signed zeros equal). *)
let%expect_test "agrees with the producer's published binary32 rule" =
  let module P = Pt2_fixture in
  Random.init 20261011;
  let specials =
    [
      0x7FC00000L;
      0x7F800000L;
      0xFF800000L;
      0x00000000L;
      0x80000000L;
      0x3F800000L;
      0x00000001L;
    ]
  in
  let random_bits () =
    if Random.int 6 = 0 then
      List.nth specials (Random.int (List.length specials))
    else
      (* Nearby values so the tolerance boundary is exercised. *)
      let base = 0x3F800000 + Random.int 0x100000 in
      Int64.of_int base
  in
  let n = 4000 in
  let actual_bits = List.init n (fun _ -> random_bits ()) in
  let expected_bits =
    List.map
      (fun b ->
        if Random.int 3 = 0 then b
        else if Random.int 4 = 0 then random_bits ()
        else Int64.add b (Int64.of_int (Random.int 40 - 20)))
      actual_bits
  in
  let atol = 1e-5 and rtol = 1.3e-6 in
  let left = F.f32_of_bits actual_bits
  and right = F.f32_of_bits expected_bits in
  let ours =
    F.element
      (F.policy
         ~element:
           (F.floating ~atol ~rtol ~scale:Right_scale ~arithmetic:Binary32 ())
         ())
      ~left ~right
  in
  let logical bits =
    let b = Bytes.create (4 * List.length bits) in
    List.iteri (fun i x -> Bytes.set_int32_le b (i * 4) (Int64.to_int32 x)) bits;
    Result.get_ok
      (P.Logical.of_bytes ~dtype:Pt2_checkpoint_map.Dtype.F32
         ~shape:[ Int64.of_int (List.length bits) ]
         (Bytes.to_string b))
  in
  let theirs =
    P.Compare.tensor ~atol ~rtol ~name:"x" ~expected:(logical expected_bits)
      ~actual:(logical actual_bits)
  in
  Format.printf "producer mismatches=%Ld ours=%a same=%b@." theirs.mismatches
    (Fmt.option Fmt.int64) ours.tolerance_violations
    (Some theirs.mismatches = ours.tolerance_violations);
  Format.printf "nontrivial: %b@."
    (theirs.mismatches > 100L && theirs.mismatches < 3900L);
  [%expect
    {|
    producer mismatches=862 ours=862 same=true
    nontrivial: true |}]

let%expect_test
    "the tolerance boundary is inclusive and binary32 rounds the tolerance" =
  let pass ?(arithmetic = Float_policy.Binary64) ~atol ~rtol l r mk =
    let p = F.policy ~element:(F.floating ~atol ~rtol ~arithmetic ()) () in
    let res = F.element p ~left:(mk [ l ]) ~right:(mk [ r ]) in
    res.tolerance_violations
  in
  (* |1.5 - 1| = 0.5 = atol exactly: accepted; a hair under: rejected. *)
  Format.printf "d = tol: %a@." (Fmt.option Fmt.int64)
    (pass ~atol:0.5 ~rtol:0.0 1.0 1.5 F.f64);
  Format.printf "d > tol: %a@." (Fmt.option Fmt.int64)
    (pass ~atol:0.4999999 ~rtol:0.0 1.0 1.5 F.f64);
  (* Relative part uses the scale: right = 2 -> tol = 0.1 * 2 = 0.2. *)
  Format.printf "relative on right scale: %a@." (Fmt.option Fmt.int64)
    (pass ~atol:0.0 ~rtol:0.1 1.8 2.0 F.f64);
  (* In binary32 atol 0.1 is first rounded to binary32 (0.100000001490116),
     so a difference of exactly that single value passes; in binary64 the
     tolerance is 0.1 and the same difference fails. *)
  let tenth32 = Int32.float_of_bits (Int32.bits_of_float 0.1) in
  Format.printf "binary32: %a, binary64: %a@." (Fmt.option Fmt.int64)
    (pass ~arithmetic:Binary32 ~atol:0.1 ~rtol:0.0 0.0 tenth32 F.f32)
    (Fmt.option Fmt.int64)
    (pass ~arithmetic:Binary64 ~atol:0.1 ~rtol:0.0 0.0 tenth32 F.f64);
  [%expect
    {|
    d = tol: 0
    d > tol: 1
    relative on right scale: 0
    binary32: 0, binary64: 1 |}]
