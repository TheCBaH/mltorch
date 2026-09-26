open Loop_ir
open Loop_fixtures
open Loop_programs

(* The harness must be able to fail. Each case below perturbs exactly one thing
   and asserts the verdict flips, because a comparison that has never gone red is
   not evidence. *)

let reference () = Kernel_eval.run_plan plan ~bind
let loop () = Loop_interp.run doubling ~bind

let verdict ~reference ~loop =
  Fmt.str "%a" Loop_check.pp_verdict (Loop_check.compare ~reference ~loop)

let%expect_test "identical results agree, including -0. and NaN" =
  Fmt.pr "%s@." (verdict ~reference:(reference ()) ~loop:(loop ()));
  [%expect {| agree |}]

(* Perturb one cell of the loop's result by one bit. *)
let perturbed ~cell ~to_ =
  match Err.payload (loop ()) with
  | Error _ -> assert false
  | Ok m ->
      let t = Tensor_id.Map.find (tid 1) m in
      Tensor.set_float t (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:cell ~c:0) to_;
      Err.return (Tensor_id.Map.add (tid 1) t m)

let%expect_test "one flipped bit flips the verdict" =
  (* 3. * 2. = 6. becomes the next float32 up. *)
  let next = Int32.float_of_bits (Int32.add (Int32.bits_of_float 6.) 1l) in
  Fmt.pr "%s@."
    (verdict ~reference:(reference ()) ~loop:(perturbed ~cell:3 ~to_:next));
  (* -0. * 2. = -0.; a +0. there is equal under [=] and must still differ. *)
  Fmt.pr "%s@."
    (verdict ~reference:(reference ()) ~loop:(perturbed ~cell:0 ~to_:0.));
  [%expect
    {|
    DISAGREE: t1 differs bitwise
    DISAGREE: t1 differs bitwise |}]

let%expect_test "a missing or extra output is a disagreement" =
  Fmt.pr "%s@."
    (verdict ~reference:(reference ()) ~loop:(Err.return Tensor_id.Map.empty));
  let with_extra =
    match Err.payload (loop ()) with
    | Ok m ->
        Err.return (Tensor_id.Map.add (tid 9) (Tensor_id.Map.find (tid 1) m) m)
    | Error _ -> assert false
  in
  Fmt.pr "%s@." (verdict ~reference:(reference ()) ~loop:with_extra);
  [%expect
    {|
    DISAGREE: loop produced no output for t1
    DISAGREE: loop produced an extra output t9 |}]

(* ---- failures: kind AND payload ------------------------------------------- *)

let%expect_test "failures agree on kind and payload, and differ otherwise" =
  let reference =
    Kernel_eval.run_plan (Fusion_plan.default shifted_kernel) ~bind
  in
  Fmt.pr "%s@."
    (verdict ~reference ~loop:(Loop_interp.run (shifted_loop ~extent:4) ~bind));
  (* Same kind at a different position: the payload disagrees. *)
  let wrong_place =
    program ~buffers:[ input; output ]
      [
        Loop_stmt.Fail_if
          ( Loop_bool.Index_eq (Loop_index.Const 0, Loop_index.Const 0),
            Loop_failure.Load_out_of_range
              { buffer = input; coord = at_w (Loop_index.Const 7) } );
      ]
  in
  Fmt.pr "%s@." (verdict ~reference ~loop:(Loop_interp.run wrong_place ~bind));
  (* A different kind altogether. *)
  let other =
    program ~buffers:[ input; output ]
      [
        Loop_stmt.Fail_if
          ( Loop_bool.Index_eq (Loop_index.Const 0, Loop_index.Const 0),
            Loop_failure.I64_division_by_zero );
      ]
  in
  Fmt.pr "%s@." (verdict ~reference ~loop:(Loop_interp.run other ~bind));
  (* Only one side fails. *)
  Fmt.pr "%s@." (verdict ~reference ~loop:(loop ()));
  Fmt.pr "%s@."
    (verdict
       ~reference:(Kernel_eval.run_plan plan ~bind)
       ~loop:(Loop_interp.run (shifted_loop ~extent:4) ~bind));
  [%expect
    {|
    agree on failure: coord_out_of_range
    DISAGREE: coord_out_of_range payloads differ
    DISAGREE: failure kinds differ: reference coord_out_of_range, loop i64_division_by_zero
    DISAGREE: only the reference failed: coord_out_of_range
    DISAGREE: only the loop failed: coord_out_of_range |}]

let%expect_test "a refusal is a verdict of its own" =
  Fmt.pr "%a@." Loop_check.pp_verdict
    (Loop_check.run (Fusion_plan.default i64_load_of_f32_kernel) ~bind);
  [%expect {| refused: t1: load of format f32 is not lowered |}]
