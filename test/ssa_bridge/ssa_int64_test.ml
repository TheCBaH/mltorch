open Ssa_bridge
open Loop_ir_test
open Loop_fixtures
open Loop_programs

(* The int64 track, direct: exact values past 2^53, modular arithmetic, checked
   division and float conversion, the reductions with their own seeds and the
   gather's range check, each through the SSA operation that owns it, against
   the reference. The cases are the Loop suite's own. *)

let floats = [| 2.9; -2.9; 0.; 0. |]

let show name ?(floats = floats) ~cells body =
  let plan = Fusion_plan.default (i64_body_kernel body) in
  Fmt.pr "%s: %a@." name Ssa_check.pp_verdict
    (Ssa_check.run plan ~bind:(i64_bind ~floats ~cells))

let big = Int64.add (Int64.shift_left 1L 53) 1L

let%expect_test "a value past 2^53 stays exact, and add, sub and mul wrap" =
  show "2^53+1, plus one"
    ~cells:[| big; Int64.max_int; Int64.min_int; 0L |]
    (Expr.Value.i64_add i64_here (Expr.Value.i64_const 1L));
  let cells = [| Int64.max_int; Int64.min_int; 3L; -5L |] in
  show "max_int + 1" ~cells
    (Expr.Value.i64_add i64_here (Expr.Value.i64_const 1L));
  show "min_int - 1" ~cells
    (Expr.Value.i64_sub i64_here (Expr.Value.i64_const 1L));
  show "cell * cell" ~cells (Expr.Value.i64_mul i64_here i64_here);
  [%expect
    {|
    2^53+1, plus one: agree
    max_int + 1: agree
    min_int - 1: agree
    cell * cell: agree
    |}]

let%expect_test
    "division truncates toward zero and fails on zero and min_int / -1" =
  let cells = [| -7L; 7L; -7L; 7L |] in
  show "x / 2" ~cells (Expr.Value.i64_div i64_here (Expr.Value.i64_const 2L));
  show "x / -2" ~cells
    (Expr.Value.i64_div i64_here (Expr.Value.i64_const (-2L)));
  show "x / 0" ~cells (Expr.Value.i64_div i64_here (Expr.Value.i64_const 0L));
  let edge = [| Int64.min_int; 1L; 1L; 1L |] in
  show "min_int / -1" ~cells:edge
    (Expr.Value.i64_div i64_here (Expr.Value.i64_const (-1L)));
  show "min_int / 1" ~cells:edge
    (Expr.Value.i64_div i64_here (Expr.Value.i64_const 1L));
  (* the zero divisor is reported before the overflow *)
  show "min_int / 0" ~cells:edge
    (Expr.Value.i64_div i64_here (Expr.Value.i64_const 0L));
  [%expect
    {|
    x / 2: agree
    x / -2: agree
    x / 0: agree on failure: i64_division_by_zero
    min_int / -1: agree on failure: i64_division_overflow
    min_int / 1: agree
    min_int / 0: agree on failure: i64_division_by_zero
    |}]

let%expect_test
    "float to int64 truncates, and refuses NaN, infinities and out of range" =
  let float_to_i64 =
    Expr.Value.float_to_i64
      (Expr.Value.load (Expr_bridge.source_of_id (tid 0)) out_coord)
  in
  List.iter
    (fun (name, x) ->
      show name ~floats:[| x; x; x; x |] ~cells:[| 0L; 0L; 0L; 0L |]
        float_to_i64)
    [
      ("2.9", 2.9);
      ("-2.9", -2.9);
      ("-0", -0.);
      ("-2^63", -.Float.pow 2. 63.);
      ("2^63", Float.pow 2. 63.);
      ("nan", nan);
      ("inf", infinity);
      ("-inf", neg_infinity);
      ("1e30", 1e30);
    ];
  show "float_of_int64 (2^53+1) then back" ~cells:[| big; 3L; -3L; 0L |]
    (Expr.Value.float_to_i64 (Expr.Value.i64_to_float i64_here));
  [%expect
    {|
    2.9: agree
    -2.9: agree
    -0: agree
    -2^63: agree
    2^63: agree on failure: i64_from_float_out_of_range
    nan: agree on failure: i64_from_float_nan
    inf: agree on failure: i64_from_float_infinite
    -inf: agree on failure: i64_from_float_infinite
    1e30: agree on failure: i64_from_float_out_of_range
    float_of_int64 (2^53+1) then back: agree
    |}]

let%expect_test "int64 reductions keep their own seeds" =
  let cells = [| 5L; -9L; 5L; 2L |] in
  let lo = Expr.Index.zero and hi = Expr.Index.const 0 in
  show "sum" ~cells (reduce_w Expr.Reduction.Sum ());
  show "max" ~cells (reduce_w Expr.Reduction.Max ());
  show "argmax value" ~cells (reduce_w Expr.Reduction.Argmax_value ());
  show "argmax index (first of the ties)" ~cells
    (reduce_w Expr.Reduction.Argmax_index ());
  show "empty sum" ~cells (reduce_w Expr.Reduction.Sum ~lo ~hi ());
  show "empty max" ~cells (reduce_w Expr.Reduction.Max ~lo ~hi ());
  show "empty argmax index" ~cells
    (reduce_w Expr.Reduction.Argmax_index ~lo ~hi ());
  show "modular sum"
    ~cells:[| Int64.max_int; 1L; 1L; 1L |]
    (reduce_w Expr.Reduction.Sum ());
  [%expect
    {|
    sum: agree
    max: agree
    argmax value: agree
    argmax index (first of the ties): agree
    empty sum: agree
    empty max: agree
    empty argmax index: agree
    modular sum: agree
    |}]

let gather name cells =
  let plan = Fusion_plan.default gather_kernel in
  let bind = i64_bind ~floats:[| 10.; 20.; 30.; 40. |] ~cells in
  Fmt.pr "%s: %a@." name Ssa_check.pp_verdict (Ssa_check.run plan ~bind)

let%expect_test "a gather index is checked in the int64 domain, then normalized"
    =
  gather "in range" [| 0L; 1L; 2L; 3L |];
  gather "negative wraps (-1 is the last)" [| -1L; -4L; -2L; -3L |];
  gather "-extent" [| -4L; -4L; -4L; -4L |];
  gather "extent - 1" [| 3L; 3L; 3L; 3L |];
  gather "extent" [| 4L; 0L; 0L; 0L |];
  gather "-extent - 1" [| -5L; 0L; 0L; 0L |];
  (* narrowed first, these would wrap into small in-range ints *)
  gather "near min_int" [| Int64.min_int; 0L; 0L; 0L |];
  gather "near max_int" [| Int64.max_int; 0L; 0L; 0L |];
  gather "2^32 (wraps to 0 in 32 bits)" [| Int64.shift_left 1L 32; 0L; 0L; 0L |];
  [%expect
    {|
    in range: agree
    negative wraps (-1 is the last): agree
    -extent: agree
    extent - 1: agree
    extent: agree on failure: gather_index_out_of_range
    -extent - 1: agree on failure: gather_index_out_of_range
    near min_int: agree on failure: gather_index_out_of_range
    near max_int: agree on failure: gather_index_out_of_range
    2^32 (wraps to 0 in 32 bits): agree on failure: gather_index_out_of_range
    |}]

let%expect_test
    "an int64 value runs between the float values it reads and is read by" =
  let bind =
    i64_bind ~floats:[| 1.5; 2.5; -1.5; 1e10 |] ~cells:[| 1L; 2L; 3L; 4L |]
  in
  Fmt.pr "%a@." Ssa_check.pp_verdict
    (Ssa_check.run (Fusion_plan.default mixed_kernel) ~bind);
  (* an int64 value that loads an F32 input is refused by format; the reference
     fails at run time, so the two are never compared *)
  (match
     Ssa_check.run (Fusion_plan.default i64_load_of_f32_kernel) ~bind:(fun _ ->
         None)
   with
  | Ssa_check.Refused u ->
      Fmt.pr "refused: %s@."
        (Ssa_lower.Ssa_unsupported.construct_name
           u.Ssa_lower.Ssa_unsupported.construct)
  | v -> Fmt.pr "%a@." Ssa_check.pp_verdict v);
  [%expect {|
    agree
    refused: load of format f32
    |}]

(* one rounding at binary32, against the Loop interpreter's own conversion *)
let%expect_test "an int64 converts to binary32 with a single rounding" =
  let samples =
    [
      0L;
      1L;
      16777217L;
      Int64.add (Int64.shift_left 1L 53) 1L;
      Int64.add (Int64.shift_left 1L 62) 1L;
      Int64.max_int;
      Int64.min_int;
      Int64.neg (Int64.add (Int64.shift_left 1L 40) 0x7FFFFFL);
      0x7FFFFF7FFFFFFFFFL;
      0x0020000020000001L;
    ]
  in
  let same =
    List.for_all
      (fun n ->
        Core.Float_bits.equal_portable
          (Ssa_ir.Ssa_const.round32_of_i64 n)
          (Loop_ir.Loop_numerics.round32_of_i64 n))
      samples
  in
  (* and it is not conversion through binary64, which double-rounds *)
  let n = 0x0020000020000001L in
  Fmt.pr "matches the Loop conversion: %b; differs from f64 then f32: %b@." same
    (not
       (Core.Float_bits.equal_portable
          (Ssa_ir.Ssa_const.round32_of_i64 n)
          (Ssa_ir.Ssa_const.round_f32 (Int64.to_float n))));
  [%expect
    {| matches the Loop conversion: true; differs from f64 then f32: true |}]
