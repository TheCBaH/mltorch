open Ssa_bridge
open Ssa_fixtures
open Loop_ir_test

(* The scalar surface beyond the first slice: unary operations, index division
   and clamping, predicates and lazy selection, maxima and argmaxes. Every case
   is checked against the reference, and where a rule is easy to get subtly
   wrong the data is chosen to show it. *)

let verdict ?(shape = Loop_fixtures.shape_w 4) ~data kernel =
  Fmt.pr "%a@." Ssa_check.pp_verdict
    (Ssa_check.run (Fusion_plan.default kernel) ~bind:(bind_data ~shape data))

let%expect_test "every unary operation, at the values that break it" =
  let data = [| nan; -0.; 1e308; -2.5 |] in
  List.iter
    (fun op ->
      Fmt.pr "%-5s " (Ssa_ir.Ssa_op.unary_name op);
      verdict ~data (Loop_programs.unary_kernel op))
    Expr.Value.[ Cos; Erf; Exp; Log; Sin; Sqrt; Trunc ];
  [%expect
    {|
    cos   agree
    erf   agree
    exp   agree
    log   agree
    sin   agree
    sqrt  agree
    trunc agree
    |}]

let position i = Expr.Index.assume_position i
let w = Expr.Index.output Expr.Axis.W
let shifted by = Expr.Index.add (Expr.Index.of_position w) (Expr.Index.const by)
let ok = function Ok x -> x | Error _ -> assert false

let%expect_test
    "floor and ceiling divide toward the right side for negative numerators" =
  (* value_of_index (floor ((w - 3) / 2)) and the ceiling: -3 -2 -1 0 / -1 -1 0 0 *)
  let index_kernel div =
    Loop_fixtures.pixel_kernel
      (Expr.Value.value_of_index (ok (div (shifted (-3)) 2)))
  in
  verdict ~data:[| 0.; 0.; 0.; 0. |] (index_kernel Expr.Index.floor_div_pos);
  verdict ~data:[| 0.; 0.; 0.; 0. |] (index_kernel Expr.Index.ceil_div_pos);
  (* min, max and clamp_low are total *)
  verdict ~data:[| 0.; 0.; 0.; 0. |]
    (Loop_fixtures.pixel_kernel
       (Expr.Value.value_of_index
          (Expr.Index.min
             (Expr.Index.max (shifted (-2)) (Expr.Index.const (-1)))
             (Expr.Index.of_position (Expr.Index.clamp_low (shifted (-1)))))));
  [%expect {|
    agree
    agree
    agree
    |}]

let%expect_test "only the selected arm runs" =
  (* t1[w] = select (w < 2.) (t0[w + 1]) 0. : the untaken arm reads out of range
     at w = 3. An eager select would fail there. *)
  let before_two =
    Expr.Bool.value_lt
      (Expr.Value.value_of_index (Expr.Index.of_position w))
      (Expr.Value.const 2.)
  in
  let load_next =
    Loop_programs.ld (Loop_programs.at Expr.Axis.W (position (shifted 1)))
  in
  let kernel =
    Loop_fixtures.pixel_kernel
      (Expr.Value.select before_two load_next (Expr.Value.const 0.))
  in
  verdict ~data:[| 5.; 6.; 7.; 8. |] kernel;
  (* and the taken arm fails when it must: w < 4 reads t0[w + 1] up to 4 *)
  let always =
    Expr.Bool.value_lt
      (Expr.Value.value_of_index (Expr.Index.of_position w))
      (Expr.Value.const 9.)
  in
  verdict ~data:[| 5.; 6.; 7.; 8. |]
    (Loop_fixtures.pixel_kernel
       (Expr.Value.select always load_next (Expr.Value.const 0.)));
  [%expect {|
    agree
    agree on failure: coord_out_of_range
    |}]

let reduce kind data =
  verdict ~shape:(Loop_programs.s1c 4) ~data (four_cell_reduction kind)

let%expect_test "maxima and argmaxes keep their NaN, tie and signed-zero rules"
    =
  List.iter
    (fun (name, kind) ->
      List.iter
        (fun data ->
          Fmt.pr "%-13s [%s]: " name
            (String.concat "; " (List.map (Fmt.str "%g") (Array.to_list data)));
          reduce kind data)
        [
          [| 1.; 3.; 3.; 2. |];
          [| nan; 1.; nan; 0. |];
          [| -0.; 0.; -0.; -1. |];
          [| neg_infinity; neg_infinity; nan; 5. |];
        ])
    [
      ("max", Expr.Reduction.Max);
      ("argmax_index", Expr.Reduction.Argmax_index);
      ("argmax_value", Expr.Reduction.Argmax_value);
    ];
  [%expect
    {|
    max           [1; 3; 3; 2]: agree
    max           [nan; 1; nan; 0]: agree
    max           [-0; 0; -0; -1]: agree
    max           [-inf; -inf; nan; 5]: agree
    argmax_index  [1; 3; 3; 2]: agree
    argmax_index  [nan; 1; nan; 0]: agree
    argmax_index  [-0; 0; -0; -1]: agree
    argmax_index  [-inf; -inf; nan; 5]: agree
    argmax_value  [1; 3; 3; 2]: agree
    argmax_value  [nan; 1; nan; 0]: agree
    argmax_value  [-0; 0; -0; -1]: agree
    argmax_value  [-inf; -inf; nan; 5]: agree
    |}]

(* ---- the max-pool intrinsic ------------------------------------------------- *)

let pool_case ?(result = Expr.Intrinsic.Max_pool.Value) ~input ~out ~kernel
    ~stride ~pad data =
  let shape = Loop_programs.hw input input in
  let plan =
    Fusion_plan.default
      (Loop_programs.pool_kernel ~input ~out ~kernel ~stride ~pad ~result)
  in
  Fmt.pr "%a@." Ssa_check.pp_verdict
    (Ssa_check.run plan ~bind:(bind_data ~shape data))

let ramp16 = Array.init 16 float_of_int

let%expect_test "max pool: windows, padding, ties, NaN and the winner's index" =
  pool_case ~input:4 ~out:2 ~kernel:2 ~stride:2 ~pad:0 ramp16;
  pool_case ~input:4 ~out:2 ~kernel:3 ~stride:2 ~pad:1 ramp16;
  pool_case ~result:Expr.Intrinsic.Max_pool.Index ~input:4 ~out:2 ~kernel:2
    ~stride:2 ~pad:0 (Array.make 16 1.);
  pool_case ~result:Expr.Intrinsic.Max_pool.Index ~input:4 ~out:2 ~kernel:3
    ~stride:2 ~pad:1 ramp16;
  let nan_data = Array.copy ramp16 in
  nan_data.(0) <- nan;
  nan_data.(1) <- nan;
  pool_case ~input:4 ~out:2 ~kernel:2 ~stride:2 ~pad:0 nan_data;
  pool_case ~result:Expr.Intrinsic.Max_pool.Index ~input:4 ~out:2 ~kernel:2
    ~stride:2 ~pad:0 nan_data;
  [%expect {|
    agree
    agree
    agree
    agree
    agree
    agree |}]

(* The window and the flat index use each axis's own geometry: a rectangular
   input tells H from W, which a square one cannot. *)
let rectangular_pool ~result =
  let open Core.Geometry in
  let hw_of h w = Hw.{ h; w } in
  let body =
    Expr.Value.intrinsic
      (Expr.Intrinsic.max_pool
         ~source:(Expr_bridge.source_of_id (Loop_fixtures.tid 0))
         ~input:(hw_of (Dim.extent 4) (Dim.extent 6))
         ~kernel:(hw_of (Dim.extent 2) (Dim.extent 3))
         ~stride:(hw_of (Op_config.Pos.of_int 2) (Op_config.Pos.of_int 1))
         ~pad:(hw_of (Op_config.Nonneg.of_int 0) (Op_config.Nonneg.of_int 1))
         ~out:Loop_programs.out_coord ~result)
  in
  Err.or_raise ~pp_error:Kernel.pp_error
    (Kernel.create
       ~inputs:
         [
           {
             Kernel.Input.id = Loop_fixtures.tid 0;
             sg = Loop_fixtures.sg 0 (Loop_programs.hw 4 6) Loop_fixtures.f32;
             binding = Kernel.Binding.Caller;
           };
         ]
       ~values:
         [
           {
             Kernel.Value.id = Loop_fixtures.tid 1;
             sg = Loop_fixtures.sg 1 (Loop_programs.hw 2 6) Loop_fixtures.f32;
             computation = Region_group.Ref.Solo (Region_program.pixel body);
             result = Kernel.Result_conversion.Round_f32;
           };
         ]
       ~outputs:[ Loop_fixtures.tid 1 ]
       ())

let%expect_test "max pool on a rectangular input" =
  let data = Array.init 24 (fun i -> float_of_int (i * 7 mod 24)) in
  List.iter
    (fun result ->
      Fmt.pr "%a@." Ssa_check.pp_verdict
        (Ssa_check.run
           (Fusion_plan.default (rectangular_pool ~result))
           ~bind:(bind_data ~shape:(Loop_programs.hw 4 6) data)))
    Expr.Intrinsic.Max_pool.[ Value; Index ];
  [%expect {|
    agree
    agree |}]
