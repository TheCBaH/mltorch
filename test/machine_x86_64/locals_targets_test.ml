module L = Machine_source_test.Mir_local_test
module Site = Machine_ir.Mir_failure.Site_entry

(* The local and meter programs through AArch64 selected and allocated, and
   x86-64 selected and realized: every run of a local site still begins with
   its bytes undefined there, and a named local's failure binds its site. *)

let sites = [| Site.Other; Site.Local_out_of_range L.local_var |]

let cases =
  [
    ("each key writes", None, L.keys ~second_writes:true);
    ( "named, outside",
      None,
      fun bld ->
        let h = Ssa_ir.Ssa_builder.local_alloc ~var:L.local_var bld ~slots:2L in
        Ssa_ir.Ssa_builder.local_write bld h (L.idx bld 0)
          (Ssa_ir.Ssa_builder.f64 bld 1.);
        L.store_at bld (L.idx bld 0)
          (Ssa_ir.Ssa_builder.local_read bld h (L.idx bld 2)) );
    ( "5 steps of 5",
      Some (L.limits ~max_state:2 ~max_updates:5L),
      L.recurrence ~steps:5 );
    ( "6 steps of 5",
      Some (L.limits ~max_state:2 ~max_updates:5L),
      L.recurrence ~steps:6 );
    ( "5 steps, state 1",
      Some (L.limits ~max_state:1 ~max_updates:5L),
      L.recurrence ~steps:5 );
  ]

let%expect_test "locals and the meter on both targets" =
  List.iter
    (fun (name, scan_limits, f) ->
      let p = L.build ?scan_limits f in
      Fmt.pr "%s: aarch64 %s, allocated %s | x86_64 %s, realized %s@." name
        (Machine_aarch64_test.A64_harness.program ~sites p ~inputs:[])
        (Machine_alloc_test.Alloc_harness.program ~sites p ~inputs:[])
        (X64_harness.program ~sites p ~inputs:[])
        (X64_harness.program ~sites ~stage:X64_harness.Realized p ~inputs:[]))
    cases;
  [%expect
    {|
    each key writes: aarch64 ok, allocated ok | x86_64 ok, realized ok
    named, outside: aarch64 unbound_local(#0), allocated unbound_local(#0) | x86_64 unbound_local(#0), realized unbound_local(#0)
    5 steps of 5: aarch64 ok, allocated ok | x86_64 ok, realized ok
    6 steps of 5: aarch64 scan_meter(updates_exhausted), allocated scan_meter(updates_exhausted) | x86_64 scan_meter(updates_exhausted), realized scan_meter(updates_exhausted)
    5 steps, state 1: aarch64 scan_meter(state_over_limit), allocated scan_meter(state_over_limit) | x86_64 scan_meter(state_over_limit), realized scan_meter(state_over_limit) |}]

(* A read of a cell this run of its site never wrote: the oracle's defect, so
   there is no oracle row to compare against. The target routes use only the
   lowered program and its bound bytes, so each is run on its own; every one
   must report the uninitialized read, not the earlier key's bytes. *)
let%expect_test "a stale cell on both targets" =
  let module Src = Machine_source_test.Mir_source in
  let module Ml = Machine_lower.Mir_lower in
  let p = L.build (L.keys ~second_writes:false) in
  let planning =
    Machine_ir.Mir_planning.make ~subject:(Ml.subject p) ~policy:"reference_f64"
      ~schedule:"scalar" ~precision:Machine_ir.Mir_planning.Precision.F64
      ~lanes:(Machine_ir.Mir_type.Lanes.of_int 1)
      ~fma:Machine_ir.Mir_planning.Fma.Forbidden ~capabilities:[]
  in
  let lowered =
    Err.or_raise ~pp_error:Ml.Refusal.pp
      (Ml.program ~planning:(Some planning) p)
  in
  let none =
    {
      Src.Route.name = "none";
      observation =
        {
          Machine_ir.Mir_observation.status =
            Machine_ir.Mir_observation.Status.Success;
          outputs = [];
          events = [];
        };
    }
  in
  let case =
    {
      Src.Case.lowered;
      bound = Src.bound_of lowered ~input:(fun _ -> None);
      oracle = none;
      generic = none;
    }
  in
  let status = function Ok o -> Src.status_name o | Error e -> e in
  Fmt.pr "aarch64 %s, allocated %s | x86_64 %s, realized %s@."
    (status (Result.map snd (Machine_aarch64_test.A64_harness.selected case)))
    (match Machine_alloc_test.Alloc_harness.physical case with
    | Machine_alloc_test.Alloc_harness.Checked o -> Src.status_name o
    | Machine_alloc_test.Alloc_harness.Rejected e -> e)
    (status (X64_harness.route X64_harness.Selected case))
    (status (X64_harness.route X64_harness.Realized case));
  [%expect
    {| aarch64 defect(uninitialized), allocated defect(uninitialized) | x86_64 defect(uninitialized), realized defect(uninitialized) |}]
