(* Region-level scan construction, checking and rendering -- Stage 1's
   Region-side half (see .ai/native_scan_design.md). Execution is a later
   step: these tests only build, check and print scan-backed programs. *)

open Expr

let limits =
  Err.or_raise ~pp_error:Scan_limits.pp_error
    (Scan_limits.create ~max_state:100 ~max_updates:1000L)

let pos n = Index.clamp_low (Index.const n)
let partition = Region_partition.singleton

let pp =
  Core.Pretty.err_result ~ok:Region_program.pp ~error:Region_program.pp_error

(* trace.(0,l) = 0; trace.(s+1,l) = trace.(s,l) + 1 -- the same counter
   [test/expr/scan_test.ml] uses, reused here through
   [Region_program.Builder.scan]. *)
let counter_scan ~width ~steps continue =
  Region_program.Builder.scan ~limits ~width ~steps
    ~init:(fun ~lane:_ -> Builder.return (Value.const 0.))
    ~update:(fun ~step:_ ~lane ~previous_at ->
      Builder.return (Value.add (previous_at lane) (Value.const 1.)))
    continue

let build_counter ~width ~steps ~row ~lane =
  Region_program.Builder.run
    (counter_scan ~width ~steps (fun scan_read ->
         Region_program.Builder.finish ~max_size:64 ~max_depth:16 ~partition
           ~output:(scan_read ~row:(pos row) ~lane:(pos lane))))

let scan_local ~width ~steps =
  match
    Region_program.locals
      (Err.or_raise ~pp_error:Region_program.pp_error
         (build_counter ~width ~steps ~row:steps ~lane:0))
  with
  | [ local ] -> local
  | _ -> assert false

let%expect_test "builds, checks and renders a scan-backed Region program" =
  Fmt.pr "%a@." pp (build_counter ~width:2 ~steps:3 ~row:3 ~lane:0);
  [%expect
    {|
    region [N=singleton T=singleton D=singleton H=singleton W=singleton C=singleton]
      let l0 : scan[width=2,steps=3] = scan[w=2,s=3](init[r1]=0; update[r2,r3,p3]=(p3[r2] + 1))
      emit l0@[max(0,3),max(0,0)] |}]

let%expect_test "Builder.scan short-circuits an invalid descriptor" =
  let invoked = ref false in
  let result =
    Region_program.Builder.run
      (Region_program.Builder.scan ~limits ~width:2 ~steps:(-1)
         ~init:(fun ~lane:_ -> Builder.return (Value.const 0.))
         ~update:(fun ~step:_ ~lane ~previous_at ->
           Builder.return (Value.add (previous_at lane) (Value.const 1.)))
         (fun _scan_read ->
           invoked := true;
           Region_program.Builder.finish ~max_size:64 ~max_depth:16 ~partition
             ~output:(Value.const 0.)))
  in
  Fmt.pr "invoked=%b result=%a@." !invoked pp result;
  [%expect {| invoked=false result=invalid scan step count -1 |}]

let%expect_test "a scalar read of a declared trace local is rejected" =
  let local = scan_local ~width:2 ~steps:3 in
  Fmt.pr "%a@." pp
    (Region_program.create ~max_size:64 ~max_depth:16 ~partition
       ~locals:[ local ]
       ~output:(Value.local local.Region_local.id));
  [%expect
    {| local #3 is read as a scalar but declared scan[width=2,steps=3] |}]

let%expect_test "a vector read of a declared trace local is rejected" =
  let local = scan_local ~width:2 ~steps:3 in
  Fmt.pr "%a@." pp
    (Region_program.create ~max_size:64 ~max_depth:16 ~partition
       ~locals:[ local ]
       ~output:(Value.local_at local.Region_local.id Index.zero));
  [%expect
    {| local #3 is read as a vector but declared scan[width=2,steps=3] |}]

let%expect_test "a trace read of a declared scalar local is rejected" =
  let id = Builder.run Builder.fresh_local in
  let local = Region_local.scalar ~id ~value:(Value.const 1.) in
  Fmt.pr "%a@." pp
    (Region_program.create ~max_size:64 ~max_depth:16 ~partition
       ~locals:[ local ]
       ~output:(Value.local_scan_at id ~row:Index.zero ~lane:Index.zero));
  [%expect {| local #0 is read as a trace but declared scalar |}]

let%expect_test "a scan's update cannot forward-reference a later local" =
  let (scan_id, later_id), _state =
    Builder.run_from Builder.initial
      (let open Builder.Syntax in
       let* scan_id = Builder.fresh_local in
       let* later_id = Builder.fresh_local in
       Builder.return (scan_id, later_id))
  in
  let forward_scan =
    Err.or_raise ~pp_error:Scan.pp_error
      (Builder.run
         (Builder.scan ~limits ~width:2 ~steps:1
            ~init:(fun ~lane:_ -> Builder.return (Value.const 0.))
            ~update:(fun ~step:_ ~lane:_ ~previous_at:_ ->
              Builder.return (Value.local later_id))))
  in
  Fmt.pr "%a@." pp
    (Region_program.create ~max_size:64 ~max_depth:16 ~partition
       ~locals:
         [
           Region_local.scan ~id:scan_id ~scan:forward_scan;
           Region_local.scalar ~id:later_id ~value:(Value.const 1.);
         ]
       ~output:(Value.local_scan_at scan_id ~row:Index.zero ~lane:Index.zero));
  [%expect {| local #0 refers forward to #1 |}]
