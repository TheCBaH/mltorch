open Ssa_bridge
open Ssa_fixtures
open Loop_ir_test

(* The harness must be able to fail: perturb one side of a real comparison and
   watch the verdict flip. A verdict that has never been seen to differ proves
   nothing. *)

let plan = Fusion_plan.default Loop_programs.kernel
let shape = Loop_fixtures.shape_w 4
let bind = bind_data ~shape [| 1.; 2.; 3.; 4. |]
let reference () = Kernel_eval.run_plan plan ~bind

let lowered () =
  Err.or_raise ~pp_error:Ssa_lower.Ssa_lower_plan.pp_error
    (Ssa_lower.Ssa_lower_plan.lower plan)

let ssa () = Ssa_lower.Ssa_exec.run plan (lowered ()) ~bind

let verdict ~reference ~ssa =
  Fmt.pr "%a@." Ssa_check.pp_verdict (Ssa_check.compare ~reference ~ssa)

let%expect_test "an agreeing pair agrees" =
  verdict ~reference:(reference ()) ~ssa:(ssa ());
  [%expect {| agree |}]

let%expect_test
    "a corrupted value, a lost output and an extra output are caught" =
  let out = Tensor_id.of_int 1 in
  let corrupt =
    Err.map
      (fun m ->
        Tensor_id.Map.add out (Loop_fixtures.f32_tensor shape (fun _ -> 99.)) m)
      (ssa ())
  in
  verdict ~reference:(reference ()) ~ssa:corrupt;
  (* a signed zero is a different value *)
  let zero v =
    Err.map (fun m ->
        Tensor_id.Map.add out (Loop_fixtures.f32_tensor shape (fun _ -> v)) m)
  in
  verdict ~reference:(zero 0. (reference ())) ~ssa:(zero (-0.) (ssa ()));
  verdict ~reference:(reference ())
    ~ssa:(Err.map (fun m -> Tensor_id.Map.remove out m) (ssa ()));
  verdict ~reference:(reference ())
    ~ssa:
      (Err.map
         (fun m ->
           Tensor_id.Map.add (Tensor_id.of_int 7)
             (Loop_fixtures.f32_tensor shape (fun _ -> 0.))
             m)
         (ssa ()));
  (* a NaN is a NaN whatever its payload *)
  verdict
    ~reference:(zero nan (reference ()))
    ~ssa:(zero (Int64.float_of_bits 0x7FF8_0000_0000_0001L) (ssa ()));
  [%expect
    {|
    DISAGREE: t1 differs bitwise
    DISAGREE: t1 differs bitwise
    DISAGREE: ssa produced no output for t1
    DISAGREE: ssa produced an extra output t7
    agree
    |}]

let%expect_test "failure kinds and payloads are compared, not just the fact" =
  let failing =
    let plan = Fusion_plan.default Loop_programs.shifted_kernel in
    (plan, bind_data ~shape [| 0.; 0.; 0.; 0. |])
  in
  let plan, bind = failing in
  let program =
    Err.or_raise ~pp_error:Ssa_lower.Ssa_lower_plan.pp_error
      (Ssa_lower.Ssa_lower_plan.lower plan)
  in
  let reference = Kernel_eval.run_plan plan ~bind in
  let ssa = Ssa_lower.Ssa_exec.run plan program ~bind in
  verdict ~reference ~ssa;
  (* the same kind with another value in the payload *)
  let other =
    Err.fail
      (`Coord_out_of_range
         ( Expr.Source.create 0,
           Expr.Axis.W,
           5,
           Expr.Coord.make ~n:0 ~t:0 ~d:0 ~h:0 ~w:5 ~c:0 )
        : Ssa_lower.Ssa_exec.error)
  in
  verdict ~reference ~ssa:other;
  (* another kind *)
  verdict ~reference
    ~ssa:
      (Err.fail
         (`Unbound_input (Tensor_id.of_int 0) : Ssa_lower.Ssa_exec.error));
  (* only one side failed *)
  verdict ~reference ~ssa:(Err.return Tensor_id.Map.empty);
  verdict
    ~reference:
      (Kernel_eval.run_plan
         Fusion_plan.(default Loop_programs.kernel)
         ~bind:(bind_data ~shape [| 0.; 0.; 0.; 0. |]))
    ~ssa;
  [%expect
    {|
    agree on failure: coord_out_of_range
    DISAGREE: coord_out_of_range payloads differ
    DISAGREE: failure kinds differ: reference coord_out_of_range, ssa unbound_input
    DISAGREE: only the reference failed: coord_out_of_range
    DISAGREE: only ssa failed: coord_out_of_range |}]
