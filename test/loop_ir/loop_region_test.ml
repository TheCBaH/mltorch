open Loop_ir
open Loop_fixtures
open Loop_programs

(* Region programs: locals once per key, the slot layout of the executor, and
   the failure a vector read past its extent reports. The reference is
   [Kernel_eval.run] with [Region_execution]'s own counters. *)

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

let loop_counts kernel =
  let counters = Loop_interp.counters () in
  (match Loop_lower.lower (Fusion_plan.default kernel) with
  | Ok program -> ignore (Loop_interp.run ~counters program ~bind)
  | Error _ -> ());
  {
    keys = counters.Loop_interp.keys;
    locals = counters.Loop_interp.locals;
    emitters = counters.Loop_interp.emitters;
    reductions = counters.Loop_interp.reductions;
    loads = counters.Loop_interp.loads;
  }

(* Counters are compared only for a run that completes. After a failure they
   count how far each executor got, and that depends on evaluation order inside
   one expression: [ocamlopt] evaluates a call's arguments right to left, the
   Loop IR left to right, so a load beside the failing read is counted by one and
   not the other. The error itself is the same. *)
let show name kernel =
  let plan = Fusion_plan.default kernel in
  let verdict = Loop_check.run plan ~bind in
  Fmt.pr "%s: %a" name Loop_check.pp_verdict verdict;
  (match verdict with
  | Loop_check.Agree ->
      let c = loop_counts kernel in
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
    extent 3, pick 2: agree; keys=2 locals=6 emitters=6 reductions=0 loads=12; counters match: true |}]

let%expect_test "a vector read past its extent fails as an unbound local" =
  show "extent 3, pick 3"
    (region_kernel_of (vector_program ~extent:3 ~pick:(pick 3)));
  [%expect {| extent 3, pick 3: agree on failure: unbound_local |}]

let%expect_test "a partition with every axis Whole has one key" =
  show "whole only" (region_kernel_of whole_only_program);
  [%expect
    {| whole only: agree; keys=1 locals=1 emitters=6 reductions=0 loads=6; counters match: true |}]
