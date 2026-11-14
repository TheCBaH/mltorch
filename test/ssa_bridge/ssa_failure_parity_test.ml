open Ssa_bridge
open Ssa_fixtures
open Loop_ir
open Loop_ir_test

(* Failure parity: a plan that fails must fail the same way through the Loop IR
   and through the structured SSA lowering, with the same kind and payload. The
   Loop interpreter is the reference; a verdict other than agreement is a gap. *)

let parity name plan ~bind =
  let program =
    Err.or_raise ~pp_error:Ssa_lower.Ssa_lower_plan.pp_error
      (Ssa_lower.Ssa_lower_plan.lower plan)
  in
  match Err.payload (Loop_lower.lower plan) with
  | Error _ -> Fmt.pr "%s: Loop refused@." name
  | Ok loop ->
      Fmt.pr "%s: %a@." name Ssa_check.pp_verdict
        (Ssa_check.compare
           ~reference:
             (Err.map_error
                (fun e -> (e : Loop_interp.error :> Kernel_eval.error))
                (Loop_interp.run loop ~bind))
           ~ssa:(Ssa_lower.Ssa_exec.run plan program ~bind))

let w = Expr.Index.output Expr.Axis.W
let position i = Expr.Index.assume_position i
let max_index = Expr.Index.const 0x7FFF_FFFF

let%expect_test "an index overflow fails the same way through both lowerings" =
  let plan =
    Fusion_plan.default
      (Loop_fixtures.pixel_kernel
         (Expr.Value.value_of_index
            (Expr.Index.add (Expr.Index.of_position w) max_index)))
  in
  parity "w + max" plan
    ~bind:(bind_data ~shape:(Loop_fixtures.shape_w 4) [| 0.; 0.; 0.; 0. |]);
  [%expect {| w + max: agree on failure: index_overflow |}]

let%expect_test "a read past the source fails the same way through both" =
  let shifted axis =
    position
      (Expr.Index.add
         (Expr.Index.of_position (Expr.Index.output axis))
         (Expr.Index.const 1))
  in
  let coord =
    Expr.Coord.set
      (Expr.Coord.set Loop_programs.out_coord Expr.Axis.H (shifted Expr.Axis.H))
      Expr.Axis.W (shifted Expr.Axis.W)
  in
  let shape = hw 2 2 in
  let plan =
    Fusion_plan.default
      (Loop_fixtures.pixel_kernel ~shape (Loop_programs.ld coord))
  in
  parity "shifted read" plan ~bind:(bind_data ~shape [| 1.; 2.; 3.; 4. |]);
  [%expect {| shifted read: agree on failure: coord_out_of_range |}]

let%expect_test "a plan that cannot fail agrees that it does not" =
  let plan = Fusion_plan.default Loop_programs.kernel in
  parity "pointwise" plan
    ~bind:(bind_data ~shape:(Loop_fixtures.shape_w 4) [| 1.; 2.; 3.; 4. |]);
  [%expect {| pointwise: agree |}]

let%expect_test "a gather index past the source fails the same way through both"
    =
  let plan = Fusion_plan.default Loop_programs.gather_kernel in
  let gather name cells =
    parity name plan
      ~bind:(Loop_programs.i64_bind ~floats:[| 10.; 20.; 30.; 40. |] ~cells)
  in
  gather "in range" [| 0L; 1L; 2L; 3L |];
  gather "extent" [| 4L; 0L; 0L; 0L |];
  gather "-extent - 1" [| -5L; 0L; 0L; 0L |];
  [%expect
    {|
    in range: agree
    extent: agree on failure: gather_index_out_of_range
    -extent - 1: agree on failure: gather_index_out_of_range |}]

let%expect_test "int64 division failures agree through both lowerings" =
  let divide name cells divisor =
    parity name
      (Fusion_plan.default
         (Loop_programs.i64_body_kernel
            (Expr.Value.i64_div Loop_programs.i64_here
               (Expr.Value.i64_const divisor))))
      ~bind:(Loop_programs.i64_bind ~floats:[| 2.9; -2.9; 0.; 0. |] ~cells)
  in
  divide "x / 2" [| -7L; 7L; -7L; 7L |] 2L;
  divide "x / 0" [| -7L; 7L; -7L; 7L |] 0L;
  divide "min_int / -1" [| Int64.min_int; 1L; 1L; 1L |] (-1L);
  divide "min_int / 0" [| Int64.min_int; 1L; 1L; 1L |] 0L;
  [%expect
    {|
    x / 2: agree
    x / 0: agree on failure: i64_division_by_zero
    min_int / -1: agree on failure: i64_division_overflow
    min_int / 0: agree on failure: i64_division_by_zero |}]

let%expect_test "an inline scan projection past its trace fails the same way" =
  let open Loop_programs in
  let bind id =
    if Tensor_id.equal id (Loop_fixtures.tid 0) then
      Some
        (Loop_fixtures.f32_tensor rows_shape (fun c ->
             float_of_int (Vec6.offset rows_shape c :> int)))
    else None
  in
  let at ~row ~lane =
    let p i = Expr.Index.assume_position (Expr.Index.const i) in
    Fusion_plan.default
      (region_kernel_of
         (Region_program.pixel
            (inline_scan_body_at ~steps:2 ~row:(p row) ~lane:(p lane))))
  in
  List.iter
    (fun (name, row, lane) -> parity name (at ~row ~lane) ~bind)
    [
      ("last cell", 2, 2);
      ("row past", 3, 0);
      ("lane past", 0, 3);
      ("both past", 3, 3);
    ];
  [%expect
    {|
    last cell: agree
    row past: agree on failure: scan_projection
    lane past: agree on failure: scan_projection
    both past: agree on failure: scan_projection |}]
