open Ssa_bridge
open Ssa_ir
open Loop_ir_test
open Loop_fixtures
open Loop_programs

(* Region programs and groups, direct: locals once per key, and the failure a
   vector read past its extent reports, against [Kernel_eval.run] with
   [Region_execution]'s own counters; a group's members each converted and stored
   by their own value. *)

let data = [| 1.; 2.; 3.; 10.; 20.; 30. |]

let bind id =
  if Tensor_id.equal id (tid 0) then
    Some
      (f32_tensor rows_shape (fun c -> data.((Vec6.offset rows_shape c :> int))))
  else None

type counts = {
  keys : int;
  locals : int;
  emitters : int;
  reductions : int;
  loads : int;
}

let reference_counts kernel =
  let counters = Region_execution.counters () in
  ignore
    (Kernel_eval.run
       ~region_counters:(Tensor_id.Map.singleton (tid 1) counters)
       kernel ~bind);
  {
    keys = counters.Region_execution.keys;
    locals = counters.Region_execution.locals;
    emitters = counters.Region_execution.emitters;
    reductions = counters.Region_execution.reductions;
    loads = counters.Region_execution.loads;
  }

let ssa_counts kernel =
  let counters = Ssa_interp.Counters.create () in
  let plan = Fusion_plan.default kernel in
  let program =
    Err.or_raise ~pp_error:Ssa_lower.Ssa_lower_plan.pp_error
      (Ssa_lower.Ssa_lower_plan.lower plan)
  in
  ignore (Ssa_lower.Ssa_exec.run ~counters plan program ~bind);
  let m = Ssa_interp.Counters.mark counters in
  {
    keys = m Ssa_mark.Key;
    locals = m Ssa_mark.Local;
    emitters = m Ssa_mark.Emitter;
    reductions = m Ssa_mark.Reduction;
    loads = Ssa_interp.Counters.loads counters;
  }

(* Counters are compared only for a run that completes. After a failure they
   count how far each executor got, which depends on evaluation order inside one
   expression; the error itself is the same. *)
let show name kernel =
  let plan = Fusion_plan.default kernel in
  let verdict = Ssa_check.run plan ~bind in
  Fmt.pr "%s: %a" name Ssa_check.pp_verdict verdict;
  (match verdict with
  | Ssa_check.Agree ->
      let c = ssa_counts kernel in
      Fmt.pr
        "; keys=%d locals=%d emitters=%d reductions=%d loads=%d; counters \
         match: %b"
        c.keys c.locals c.emitters c.reductions c.loads
        (c = reference_counts kernel)
  | _ -> ());
  Fmt.pr "@."

let%expect_test "a scalar local runs once per key, not once per output" =
  show "centered" (region_kernel_of centered_program);
  [%expect
    {| centered: agree; keys=2 locals=2 emitters=6 reductions=6 loads=12; counters match: true |}]

let%expect_test
    "a vector local: extent 1 (the SDPA case), a full row, and a pick" =
  show "extent 1" (region_kernel_of (vector_program ~extent:1 ~pick:(pick 0)));
  show "extent 3, pick 2"
    (region_kernel_of (vector_program ~extent:3 ~pick:(pick 2)));
  [%expect
    {|
    extent 1: agree; keys=2 locals=2 emitters=6 reductions=0 loads=8; counters match: true
    extent 3, pick 2: agree; keys=2 locals=6 emitters=6 reductions=0 loads=12; counters match: true
    |}]

let%expect_test "a vector read past its extent fails as an unbound local" =
  show "extent 3, pick 3"
    (region_kernel_of (vector_program ~extent:3 ~pick:(pick 3)));
  [%expect {| extent 3, pick 3: agree on failure: unbound_local |}]

let%expect_test "a partition with every axis Whole has one key" =
  show "whole only" (region_kernel_of whole_only_program);
  [%expect
    {| whole only: agree; keys=1 locals=1 emitters=6 reductions=0 loads=6; counters match: true |}]

(* ---- groups ------------------------------------------------------------------ *)

let group name kernel =
  Fmt.pr "%s: %a@." name Ssa_check.pp_verdict
    (Ssa_check.run (Fusion_plan.default kernel) ~bind:bind_none)

let%expect_test "each member is converted and stored by its own value" =
  (* [read - 2] is -1, 0, 1, 2 across W: true, false, true, true *)
  group "bool member"
    (bool_group_kernel ~second:(fun read ->
         Expr.Value.sub read (Expr.Value.const 2.)));
  (* below binary32's smallest subnormal: [Nonzero_bool] on the working value
     says true, and storing to f32 first says false *)
  group "1e-50"
    (bool_group_kernel ~second:(fun read ->
         Expr.Value.mul read (Expr.Value.const 1e-50)));
  [%expect {|
    bool member: agree
    1e-50: agree
    |}]
