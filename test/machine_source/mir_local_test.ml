open Ssa_ir
module B = Ssa_builder

(* M4.3: scratch locals, local and scan checks, and the scan meter through the
   generic route, against the SSA interpreter. A local's freshness is checked,
   not assumed: each run of an allocation site begins with its bytes undefined,
   so a read of a cell this run never wrote is the generic interpreter's
   uninitialized defect, as it is the oracle's. *)

let local_var = Expr.Builder.run Expr.Builder.fresh_local

let out =
  {
    Ssa_buffer.id = Ssa_id.Buffer.of_int 1;
    extents = Expr.Coord.make ~n:1L ~t:1L ~d:1L ~h:1L ~w:4L ~c:1L;
    format = Ssa_format.F32;
    role = Ssa_buffer.Output;
  }

let limits ~max_state ~max_updates =
  Err.or_raise ~pp_error:Expr.Scan_limits.pp_error
    (Expr.Scan_limits.create ~max_state ~max_updates)

let idx bld k = B.index bld (Int64.of_int k)

let store_at bld at x =
  B.store_f64 bld out.Ssa_buffer.id ~encode:Ssa_op.Encode.F32_round (B.Flat at)
    x

let build ?scan_limits f =
  Err.or_raise ~pp_error:Ssa_verify.pp_error
    (B.program ?scan_limits ~buffers:[ out ] f)

let run ?mutation ?scan_limits label f =
  match build ?scan_limits f with
  | p ->
      Fmt.pr "%s: %s@." label (Mir_source.check_program ?mutation p ~inputs:[])
  | exception Err.Exn.E e -> Fmt.pr "%s: rejected: %a@." label Err.Exn.pp_kind e

(* two keys, each with its own object; only the first writes before reading *)
let keys ~second_writes bld =
  let B.Nil =
    B.for_ bld ~lo:(idx bld 0) ~hi:(idx bld 2) ~init:B.Nil (fun bld key B.Nil ->
        let h = B.local_alloc bld ~slots:1L in
        let first = B.index_compare bld Ssa_op.Compare.Eq key (idx bld 0) in
        let B.Nil =
          B.if_ bld first
            ~then_:(fun bld ->
              B.local_write bld h (idx bld 0) (B.f64 bld 5.);
              B.Nil)
            ~else_:(fun bld ->
              if second_writes then
                B.local_write bld h (idx bld 0) (B.f64 bld (-0.));
              B.Nil)
        in
        store_at bld key (B.local_read bld h (idx bld 0));
        B.Nil)
  in
  ()

let%expect_test "locals: written before read, fresh per run of a site" =
  run "write then read" (fun bld ->
      let h = B.local_alloc bld ~slots:2L in
      B.local_write bld h (idx bld 1) (B.f64 bld 7.);
      B.local_write bld h (idx bld 0) (B.f64 bld Float.nan);
      store_at bld (idx bld 0) (B.local_read bld h (idx bld 1));
      store_at bld (idx bld 1) (B.local_read bld h (idx bld 0)));
  run "never written" (fun bld ->
      let h = B.local_alloc bld ~slots:2L in
      store_at bld (idx bld 0) (B.local_read bld h (idx bld 0)));
  run "each key writes" (keys ~second_writes:true);
  run "second key reads the first's cell" (keys ~second_writes:false);
  run ~mutation:Machine_lower.Mir_lower.Mutation.Stale_local
    "second key reads the first's cell, stale"
    (keys ~second_writes:false);
  [%expect
    {|
      write then read: ok [0x1.cp+2:f32 nan:f32 0x0p+0:f32 0x0p+0:f32]
      never written: defect(uninitialized); oracle defect: Ssa_interp: a read of a local cell that was never written
      each key writes: ok [0x1.4p+2:f32 -0x0p+0:f32 0x0p+0:f32 0x0p+0:f32]
      second key reads the first's cell: defect(uninitialized); oracle defect: Ssa_interp: a read of a local cell that was never written
      second key reads the first's cell, stale: ok [0x1.4p+2:f32 0x1.4p+2:f32 0x0p+0:f32 0x0p+0:f32] DISAGREE oracle defect: Ssa_interp: a read of a local cell that was never written |}]

let%expect_test "local and scan checks" =
  run "named, outside" (fun bld ->
      let h = B.local_alloc ~var:local_var bld ~slots:2L in
      B.local_write bld h (idx bld 0) (B.f64 bld 1.);
      store_at bld (idx bld 0) (B.local_read bld h (idx bld 2)));
  run "named, below" (fun bld ->
      let h = B.local_alloc ~var:local_var bld ~slots:2L in
      store_at bld (idx bld 0) (B.local_read bld h (idx bld (-1))));
  run "anonymous, outside" (fun bld ->
      let h = B.local_alloc bld ~slots:2L in
      store_at bld (idx bld 0) (B.local_read bld h (idx bld 2)));
  List.iter
    (fun k ->
      run (Fmt.str "check_local %d of 2" k) (fun bld ->
          B.check_local bld ~var:local_var ~extent:2L (idx bld k);
          store_at bld (idx bld 0) (B.f64 bld 3.)))
    [ -1; 0; 1; 2 ];
  List.iter
    (fun (row, lane, var) ->
      run (Fmt.str "check_scan row %d lane %d" row lane) (fun bld ->
          B.check_scan bld ~var ~row:(idx bld row) ~lane:(idx bld lane)
            ~row_extent:3L ~lane_extent:2L;
          store_at bld (idx bld 0) (B.f64 bld 1.)))
    [
      (2, 1, Some local_var);
      (3, 0, Some local_var);
      (0, 2, Some local_var);
      (3, 2, None);
      (-1, 0, None);
      (0, -1, None);
    ];
  [%expect
    {|
    named, outside: unbound_local(#0)
    named, below: unbound_local(#0)
    anonymous, outside: defect(bad_access); oracle defect: Ssa_interp: a local read outside its object
    check_local -1 of 2: unbound_local(#0)
    check_local 0 of 2: ok [0x1.8p+1:f32 0x0p+0:f32 0x0p+0:f32 0x0p+0:f32]
    check_local 1 of 2: ok [0x1.8p+1:f32 0x0p+0:f32 0x0p+0:f32 0x0p+0:f32]
    check_local 2 of 2: unbound_local(#0)
    check_scan row 2 lane 1: ok [0x1p+0:f32 0x0p+0:f32 0x0p+0:f32 0x0p+0:f32]
    check_scan row 3 lane 0: scan_projection(row, #0)
    check_scan row 0 lane 2: scan_projection(lane, #0)
    check_scan row 3 lane 2: scan_projection(row, inline)
    check_scan row -1 lane 0: scan_projection(row, inline)
    check_scan row 0 lane -1: scan_projection(lane, inline) |}]

let%expect_test "the meter: updates, live state, failure prefix" =
  let charges n ~max_updates =
    run (Fmt.str "%d charges of %Ld" n max_updates)
      ~scan_limits:(limits ~max_state:100 ~max_updates) (fun bld ->
        for k = 0 to n - 1 do
          B.meter_charge bld;
          B.mark bld Ssa_mark.Scan_update;
          store_at bld (idx bld (k mod 4)) (B.f64 bld (Float.of_int k))
        done)
  in
  charges 3 ~max_updates:3L;
  charges 4 ~max_updates:3L;
  charges 1 ~max_updates:0L;
  let reserve widths ~max_state =
    run
      (Fmt.str "reserve [%s] of %d"
         (String.concat ";" (List.map string_of_int widths))
         max_state)
      ~scan_limits:(limits ~max_state ~max_updates:10L)
      (fun bld ->
        List.iter
          (fun width -> B.meter_reserve bld ~width:(Int64.of_int width))
          widths;
        store_at bld (idx bld 0) (B.f64 bld 1.))
  in
  reserve [ 3 ] ~max_state:6;
  reserve [ 3 ] ~max_state:5;
  reserve [ 2; 2 ] ~max_state:8;
  reserve [ 2; 2 ] ~max_state:7;
  let scan_limits = limits ~max_state:6 ~max_updates:10L in
  run "reserve, release, reserve" ~scan_limits (fun bld ->
      B.meter_reserve bld ~width:3L;
      B.meter_release bld ~width:3L;
      B.meter_reserve bld ~width:3L;
      store_at bld (idx bld 0) (B.f64 bld 1.));
  run "reserve, reset, reserve" ~scan_limits (fun bld ->
      B.meter_reserve bld ~width:3L;
      B.meter_reset bld;
      B.meter_reserve bld ~width:3L;
      store_at bld (idx bld 0) (B.f64 bld 1.));
  (* a reset gives back the update budget too *)
  run "charges across a reset"
    ~scan_limits:(limits ~max_state:6 ~max_updates:2L) (fun bld ->
      B.meter_charge bld;
      B.meter_charge bld;
      B.meter_reset bld;
      B.meter_charge bld;
      store_at bld (idx bld 0) (B.f64 bld 1.));
  (* the charge must fail before the body it guards: its mark and store *)
  run ~mutation:Machine_lower.Mir_lower.Mutation.Charge_after_body
    "4 charges of 3, charged after the body"
    ~scan_limits:(limits ~max_state:100 ~max_updates:3L) (fun bld ->
      for k = 0 to 3 do
        B.meter_charge bld;
        B.mark bld Ssa_mark.Scan_update;
        store_at bld (idx bld k) (B.f64 bld (Float.of_int k))
      done);
  [%expect
    {|
    3 charges of 3: ok [0x0p+0:f32 0x1p+0:f32 0x1p+1:f32 0x0p+0:f32]
    4 charges of 3: scan_meter(updates_exhausted)
    1 charges of 0: scan_meter(updates_exhausted)
    reserve [3] of 6: ok [0x1p+0:f32 0x0p+0:f32 0x0p+0:f32 0x0p+0:f32]
    reserve [3] of 5: scan_meter(state_over_limit)
    reserve [2;2] of 8: ok [0x1p+0:f32 0x0p+0:f32 0x0p+0:f32 0x0p+0:f32]
    reserve [2;2] of 7: scan_meter(state_over_limit)
    reserve, release, reserve: ok [0x1p+0:f32 0x0p+0:f32 0x0p+0:f32 0x0p+0:f32]
    reserve, reset, reserve: ok [0x1p+0:f32 0x0p+0:f32 0x0p+0:f32 0x0p+0:f32]
    charges across a reset: ok [0x1p+0:f32 0x0p+0:f32 0x0p+0:f32 0x0p+0:f32]
    4 charges of 3, charged after the body: scan_meter(updates_exhausted) DISAGREE structured vs generic: event scan_update: 3 vs 4 |}]

(* An inline scan as the emitters build one: reserve two rolling rows, charge
   each update before it runs, read the last row, release. *)
let recurrence bld ~steps =
  B.meter_reserve bld ~width:1L;
  let rows = B.local_alloc bld ~slots:2L in
  B.local_write bld rows (idx bld 0) (B.f64 bld 1.);
  let B.Nil =
    B.for_ bld ~lo:(idx bld 0) ~hi:(idx bld steps) ~init:B.Nil
      (fun bld _ B.Nil ->
        B.meter_charge bld;
        B.mark bld Ssa_mark.Scan_update;
        let prev = B.local_read bld rows (idx bld 0) in
        B.local_write bld rows (idx bld 1)
          (B.f64_binary bld Expr.Value.Mul prev (B.f64 bld 3.));
        B.local_write bld rows (idx bld 0) (B.local_read bld rows (idx bld 1));
        B.Nil)
  in
  store_at bld (idx bld 0) (B.local_read bld rows (idx bld 0));
  B.meter_release bld ~width:1L

let%expect_test "an inline scan" =
  let scan_limits = limits ~max_state:2 ~max_updates:5L in
  run "5 steps of 5" ~scan_limits (recurrence ~steps:5);
  run "6 steps of 5" ~scan_limits (recurrence ~steps:6);
  run "5 steps, state 1"
    ~scan_limits:(limits ~max_state:1 ~max_updates:5L)
    (recurrence ~steps:5);
  [%expect
    {|
    5 steps of 5: ok [0x1.e6p+7:f32 0x0p+0:f32 0x0p+0:f32 0x0p+0:f32]
    6 steps of 5: scan_meter(updates_exhausted)
    5 steps, state 1: scan_meter(state_over_limit) |}]

let%expect_test "refused: a local through a loop parameter, local storage" =
  run "carried" (fun bld ->
      let h = B.local_alloc bld ~slots:1L in
      B.local_write bld h (idx bld 0) (B.f64 bld 2.);
      let (B.Cons (h, B.Nil)) =
        B.for_ bld ~lo:(idx bld 0) ~hi:(idx bld 2)
          ~init:(B.Cons (h, B.Nil))
          (fun _ _ (B.Cons (h, B.Nil)) -> B.Cons (h, B.Nil))
      in
      store_at bld (idx bld 0) (B.local_read bld h (idx bld 0)));
  run "4 GiB of cells" (fun bld ->
      let h = B.local_alloc bld ~slots:0x2000_0000L in
      B.local_write bld h (idx bld 0) (B.f64 bld 2.);
      store_at bld (idx bld 0) (B.local_read bld h (idx bld 0)));
  [%expect
    {|
    carried: refused: local v11 is not its allocation's own result
    4 GiB of cells: refused: local v1 takes local storage past 2147483648 bytes |}]
