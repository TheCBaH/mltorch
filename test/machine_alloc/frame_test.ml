open Loop_ir_test
open Ssa_bridge_test.Ssa_fixtures
open Machine_ir
open Machine_interp
open Machine_target_aarch64
module H = Alloc_harness
module B = Mir_builder

(* C5 for AArch64: realized frames on a stack region — prologues, epilogues on
   every exit, the link register, callee-saved state and FPCR as the caller
   left them (callee-saved patterns, flush-to-zero set) — agreeing with the
   selected route and the SSA oracle, checked by the checker; large frames
   through reserved scratch; recursion; and each frame mutation caught. *)

let data_bind data = bind_data ~shape:(Loop_fixtures.shape_w 4) data

let case_of_plan k ~bind =
  Result.get_ok
    (Machine_source_test.Mir_source.case_of_plan (Fusion_plan.default k) ~bind)

let case_of_program p =
  Result.get_ok
    (Machine_source_test.Mir_source.case_of_program p ~inputs:Alloc_test.inputs
       ())

let mm =
  case_of_plan
    (matmul_kernel ~m:5 ~k:3 ~n:3)
    ~bind:(matmul_bind ~m:5 ~k:3 ~n:3 ~a:(operand 3 15) ~b:(operand 5 9))

let%expect_test "realized frames agree" =
  let zeros = data_bind [| 0.; 0.; 0.; 0. |] in
  List.iter
    (fun c -> Fmt.pr "%s@." (H.realized_report c))
    [
      case_of_plan Loop_programs.kernel
        ~bind:(data_bind [| -0.; 1.5; nan; 3. |]);
      case_of_plan Loop_programs.shifted_kernel ~bind:zeros;
      case_of_plan Loop_programs.overflow_kernel ~bind:zeros;
      mm;
      case_of_program (Alloc_test.recurrences 5);
    ];
  (* a frame padded past every directly encodable offset: each slot reached
     through x16 *)
  Fmt.pr "large frame: %s@." (H.realized_report ~pad:40_000L mm);
  Fmt.pr "frame beyond the code model: %s@."
    (H.realized_report ~pad:0x2000_0000L mm);
  [%expect
    {|
    ok
    coord_out_of_range(t0, W)
    index_overflow(mul)
    ok
    ok
    large frame: ok
    frame beyond the code model: rejected: frame: fn0: a frame beyond the supported code model |}]

(* The calls program, realized, under the caller's state. *)
let calls_realized ?mutation ?edit (x, n) =
  let g = Result.get_ok (Err.payload (Mir_verify.generic Calls_test.program)) in
  let res = Result.get_ok (Err.payload (A64_select.program g)) in
  match H.realize ?mutation ?edit res with
  | Error e -> "rejected: " ^ e
  | Ok real ->
      let memory = Mir_memory.create () in
      let binding =
        Result.get_ok
          (Mir_interp.instantiate (H.regions_program real) memory
             ~bound:(fun _ -> None))
      in
      let sel = A64_stage.Sel.Verified.selected res.A64_select.selected in
      let record =
        (Option.get
           (Mir_program.find_view sel.A64_stage.Sel.program
              res.A64_select.record))
          .Mir_view.region
      in
      Calls_test.staged ~record memory binding
        (H.P.run ~realized:true ~seed:H.caller_state ~models:Calls_test.models
           real memory binding
           ~args:[ Mir_datum.f64 x; Mir_datum.Bits n ])
          .H.P.outcome

let%expect_test "calls under realized frames: success and failure exits" =
  List.iter
    (fun c -> print_endline (calls_realized c))
    [ (1.5, 3L); (0.1, 0L); (-2., -4L) ];
  (* a callee that writes x19: its frame saves and restores it *)
  print_endline (calls_realized ~edit:Calls_test.writes_x19 (1.5, 3L));
  [%expect
    {|
    ok 0x1.6p+2
    i64_division_by_zero
    i64_division_overflow
    ok 0x1.6p+2 |}]

(* f(n) = if n = 0 then () else f(n - 1), called from main with n *)
let recursive =
  let fn0 = Mir_id.Func.of_int 0 and fn1 = Mir_id.Func.of_int 1 in
  let signature _ =
    Some { Mir_typing.Signature.params = [ Mir_type.i64 ]; results = [] }
  in
  let f =
    let bld = B.create () in
    let e = B.new_block bld [ Mir_type.i64 ] in
    let base = B.new_block bld [] and step = B.new_block bld [] in
    let n = List.hd (B.param e) in
    let z =
      B.emit bld e
        (Mir_op.Icmp
           (Mir_op.Icmp.Eq, n, B.emit bld e (Mir_op.Const (Mir_const.i64 0L))))
    in
    B.branch e z (base, []) (step, []);
    B.return base [];
    let m =
      B.emit bld step
        (Mir_op.Iarith
           ( Mir_op.Iarith.Sub,
             n,
             B.emit bld step (Mir_op.Const (Mir_const.i64 1L)) ))
    in
    ignore
      (Result.get_ok
         (B.op bld step ~signature
            (Mir_op.Call (Mir_op.Callee.Func fn1, [ m ]))));
    B.return step [];
    B.func bld ~id:fn1 ~name:"f" ~entry:e ~results:[]
  in
  let main =
    let bld = B.create () in
    let e = B.new_block bld [ Mir_type.i64 ] in
    ignore
      (Result.get_ok
         (B.op bld e ~signature
            (Mir_op.Call (Mir_op.Callee.Func fn1, B.param e))));
    B.return e [];
    B.func bld ~id:fn0 ~name:"main" ~entry:e ~results:[]
  in
  B.program [ main; f ] ~main:fn0

let%expect_test "recursion: nested frames, bounded depth" =
  let g = Result.get_ok (Err.payload (Mir_verify.generic recursive)) in
  let res = Result.get_ok (Err.payload (A64_select.program g)) in
  let real = Result.get_ok (H.realize res) in
  List.iter
    (fun (n, max_depth) ->
      let memory = Mir_memory.create () in
      let binding =
        Result.get_ok
          (Mir_interp.instantiate (H.regions_program real) memory
             ~bound:(fun _ -> None))
      in
      Fmt.pr "n=%d: %a@." n Mir_interp.Outcome.pp
        (H.P.run ~realized:true ~seed:H.caller_state ~max_depth real memory
           binding
           ~args:[ Mir_datum.Bits (Int64.of_int n) ])
          .H.P.outcome)
    [ (0, 64); (20, 64); (100, 64) ];
  [%expect
    {|
    n=0: success [0x0]
    n=20: success [0x0]
    n=100: fuel exhausted |}]

let%expect_test "frame mutations are caught" =
  let open Machine_alloc.Mir_frame.Mutation in
  let show label s = Fmt.pr "%s: %s@." label s in
  show "misaligned frame" (H.realized_report ~mutation:Misalign mm);
  show "unexpanded large offset"
    (H.realized_report ~mutation:Unexpanded ~pad:40_000L mm);
  show "allocatable scratch"
    (H.realized_report ~mutation:Allocatable_scratch ~pad:40_000L mm);
  show "no link save" (calls_realized ~mutation:No_link_save (1.5, 3L));
  show "no control restore"
    (calls_realized ~mutation:No_control_restore (1.5, 3L));
  show "narrow callee save"
    (calls_realized ~mutation:Narrow_save ~edit:Calls_test.writes_x19 (1.5, 3L));
  List.iter
    (fun c ->
      show "epilogue on one exit only"
        (calls_realized ~mutation:Epilogue_once c))
    [ (1.5, 3L); (0.1, 0L); (-2., -4L) ];
  [%expect
    {|
    misaligned frame: rejected: physical verifier: allocated fn0: target constraint: a frame size that breaks stack alignment
    unexpanded large offset: rejected: physical verifier: allocated fn0 bb0: target constraint: a frame offset that does not encode
    allocatable scratch: rejected: physical verifier: allocated fn0 bb0: target constraint: a late form touching an allocatable register
    no link save: defect return_address at fn0 bb4
    no control restore: defect preserved_state at fn0 bb4
    narrow callee save: rejected: physical verifier: allocated fn1 bb0: target constraint: a save of a register the convention does not preserve
    epilogue on one exit only: defect preserved_state at fn1 bb2
    epilogue on one exit only: defect preserved_state at fn0 bb3
    epilogue on one exit only: i64_division_overflow |}]
