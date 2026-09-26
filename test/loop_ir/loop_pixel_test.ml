open Loop_ir
open Loop_fixtures
open Loop_programs

(* Pixel-form kernels through all three executors: [Kernel_eval] is the oracle,
   the Loop interpreter runs the lowered program, and every case asserts BOTH
   that they agree and what the answer is, because two executors can agree on a
   wrong value. *)

let hex x =
  if Float.is_nan x then "nan"
  else Printf.sprintf "%016Lx" (Int64.bits_of_float x)

let bind_data ~shape data id =
  if Tensor_id.equal id (tid 0) then
    Some (f32_tensor shape (fun c -> data.((Vec6.offset shape c :> int))))
  else None

(* The verdict, then the first [n] cells of the Loop interpreter's output. *)
let show ?(n = 4) ~shape ~data kernel =
  let plan = Fusion_plan.default kernel in
  let bind = bind_data ~shape data in
  Fmt.pr "%a@." Loop_check.pp_verdict (Loop_check.run plan ~bind);
  match Loop_lower.lower plan with
  | Error _ -> ()
  | Ok program -> (
      match Err.payload (Loop_interp.run program ~bind) with
      | Error e -> Fmt.pr "loop failed: %a@." Loop_interp.pp_error e
      | Ok m ->
          let t = Tensor_id.Map.find (tid 1) m in
          let cells =
            List.init n (fun i ->
                let (Tensor.Tensor tt) = t in
                let coord =
                  if Dim.to_int (Vec6.get tt.Tensor.shape Axis.C) > 1 then
                    Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:i
                  else Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:i ~c:0
                in
                hex (Tensor.read t coord))
          in
          Fmt.pr "%s@." (String.concat " " cells))

let%expect_test "elementwise arithmetic" =
  show ~shape:(shape_w 4) ~data:[| 1.; 2.; 3.; 4. |]
    (pixel_kernel
       (Expr.Value.add
          (Expr.Value.mul load_t0 (Expr.Value.const 2.))
          (Expr.Value.const 1.)));
  [%expect
    {|
    agree
    4008000000000000 4014000000000000 401c000000000000 4022000000000000 |}]

let%expect_test "a filled input folds to its rounded value" =
  show ~shape:(shape_w 4) ~data:[| 1.; 1.; 1.; 1. |] (filled_kernel 16777217.);
  [%expect
    {|
    agree
    4170000000000000 4170000000000000 4170000000000000 4170000000000000 |}]

(* ---- reductions ------------------------------------------------------------ *)

let reduce_case kind ?(bounds = full) data =
  let lo, hi = bounds in
  show ~n:1 ~shape:(s1c 3) ~data (reduction_kernel kind ~lo ~hi)

let%expect_test "an all -0. sum is +0., not the first element" =
  reduce_case Expr.Reduction.Sum [| -0.; -0.; -0. |];
  [%expect {|
    agree
    0000000000000000 |}]

let%expect_test "sum and max propagate NaN" =
  reduce_case Expr.Reduction.Sum [| 1.; nan; 2. |];
  reduce_case Expr.Reduction.Max [| 1.; nan; 2. |];
  [%expect {|
    agree
    nan
    agree
    nan |}]

let%expect_test "max seeds at -inf and orders -0. below +0." =
  reduce_case Expr.Reduction.Max [| -0.; 0.; -0. |];
  reduce_case Expr.Reduction.Max [| -5.; -7.; -6. |];
  [%expect {|
    agree
    0000000000000000
    agree
    c014000000000000 |}]

let%expect_test "argmax keeps the first of equal maxima and the last NaN" =
  reduce_case Expr.Reduction.Argmax_index [| 1.; 3.; 3. |];
  reduce_case Expr.Reduction.Argmax_value [| 1.; 3.; 3. |];
  reduce_case Expr.Reduction.Argmax_index [| nan; 5.; nan |];
  [%expect
    {|
    agree
    3ff0000000000000
    agree
    4008000000000000
    agree
    4000000000000000 |}]

let%expect_test "an empty range returns the seed" =
  let empty = (Expr.Index.zero, Expr.Index.const 0) in
  reduce_case Expr.Reduction.Sum ~bounds:empty [| 1.; 2.; 3. |];
  reduce_case Expr.Reduction.Max ~bounds:empty [| 1.; 2.; 3. |];
  reduce_case Expr.Reduction.Argmax_index ~bounds:empty [| 1.; 2.; 3. |];
  [%expect
    {|
    agree
    0000000000000000
    agree
    fff0000000000000
    agree
    0000000000000000 |}]

(* ---- guards ---------------------------------------------------------------- *)

let count_fail_ifs plan =
  match Loop_lower.lower plan with
  | Error _ -> -1
  | Ok p ->
      let n = ref 0 in
      let rec go = function
        | Loop_stmt.Fail_if _ -> incr n
        | Loop_stmt.For { body; _ } -> List.iter go body
        | Loop_stmt.If (_, a, b) ->
            List.iter go a;
            List.iter go b
        | _ -> ()
      in
      List.iter go p.Loop_program.body;
      !n

let%expect_test
    "the interval proof discharges every check of an in-range kernel" =
  let elementwise =
    pixel_kernel (Expr.Value.mul load_t0 (Expr.Value.const 2.))
  in
  Fmt.pr "elementwise: %d@." (count_fail_ifs (Fusion_plan.default elementwise));
  Fmt.pr "shifted load: %d@."
    (count_fail_ifs (Fusion_plan.default Loop_programs.shifted_kernel));
  [%expect {|
    elementwise: 0
    shifted load: 1 |}]

let%expect_test "a load past the extent fails with the reference's own row" =
  show ~shape:(shape_w 4) ~data:[| 1.; 2.; 3.; 4. |]
    Loop_programs.shifted_kernel;
  [%expect
    {|
    agree on failure: coord_out_of_range
    loop failed: t0[0,0,0,0,4,0] out of range on axis W: 4 |}]

(* 2^30 * w leaves the index domain at w = 2. The reference's checked domain is
   its host int: 32 bits under js_of_ocaml, where it reports the same first
   operation, and wider natively, where it computes 2^31 without complaint. *)
let%expect_test "an index that leaves the domain is a failure, never a wrap" =
  let plan = Fusion_plan.default overflow_kernel in
  let bind = bind_data ~shape:(shape_w 4) [| 0.; 0.; 0.; 0. |] in
  (match Loop_lower.lower plan with
  | Error _ -> Fmt.pr "refused@."
  | Ok program -> (
      match Err.payload (Loop_interp.run program ~bind) with
      | Error e -> Fmt.pr "%s@." (Loop_check.kind (e :> Kernel_eval.error))
      | Ok _ -> Fmt.pr "no failure@."));
  let verdict = Loop_check.run plan ~bind in
  Fmt.pr "agrees where the reference's int is 32 bits: %b@."
    (Sys.int_size > 32
    || match verdict with Loop_check.Agree_on_failure _ -> true | _ -> false);
  [%expect
    {|
    index_overflow
    agrees where the reference's int is 32 bits: true |}]

(* ---- the max-pool intrinsic ------------------------------------------------- *)

let pool_case ?(result = Expr.Intrinsic.Max_pool.Value) ~input ~out ~kernel
    ~stride ~pad data =
  let shape = hw input input in
  let plan =
    Fusion_plan.default (pool_kernel ~input ~out ~kernel ~stride ~pad ~result)
  in
  let bind = bind_data ~shape data in
  Fmt.pr "%a@." Loop_check.pp_verdict (Loop_check.run plan ~bind);
  match Loop_lower.lower plan with
  | Error _ -> ()
  | Ok program -> (
      match Err.payload (Loop_interp.run program ~bind) with
      | Error e -> Fmt.pr "loop failed: %a@." Loop_interp.pp_error e
      | Ok m ->
          let t = Tensor_id.Map.find (tid 1) m in
          Fmt.pr "%s@."
            (String.concat " "
               (List.concat
                  (List.init out (fun h ->
                       List.init out (fun w ->
                           Printf.sprintf "%g"
                             (Tensor.read t
                                (Vec6.coord ~n:0 ~t:0 ~d:0 ~h ~w ~c:0))))))))

let ramp16 = Array.init 16 float_of_int

let%expect_test "an unpadded window" =
  pool_case ~input:4 ~out:2 ~kernel:2 ~stride:2 ~pad:0 ramp16;
  [%expect {|
    agree
    5 7 13 15 |}]

let%expect_test "a padded, clipped window" =
  pool_case ~input:4 ~out:2 ~kernel:3 ~stride:2 ~pad:1 ramp16;
  [%expect {|
    agree
    5 7 13 15 |}]

let%expect_test
    "the index result is the flat position of the winner, first on ties" =
  let tied = Array.make 16 1. in
  pool_case ~result:Expr.Intrinsic.Max_pool.Index ~input:4 ~out:2 ~kernel:2
    ~stride:2 ~pad:0 tied;
  pool_case ~result:Expr.Intrinsic.Max_pool.Index ~input:4 ~out:2 ~kernel:3
    ~stride:2 ~pad:1 ramp16;
  [%expect {|
    agree
    0 2 8 10
    agree
    5 7 13 15 |}]

let%expect_test "a NaN wins, and the last NaN wins with its index" =
  let data = Array.copy ramp16 in
  data.(0) <- nan;
  data.(1) <- nan;
  pool_case ~input:4 ~out:2 ~kernel:2 ~stride:2 ~pad:0 data;
  pool_case ~result:Expr.Intrinsic.Max_pool.Index ~input:4 ~out:2 ~kernel:2
    ~stride:2 ~pad:0 data;
  [%expect {|
    agree
    nan 7 13 15
    agree
    1 7 13 15 |}]
