(* The bounded ordered scan primitive: construction, admission, inline
   evaluation and its interaction with freshening/checking. See
   .ai/native_scan_design.md for the full contract. *)

open Expr

let limits =
  Err.or_raise ~pp_error:Scan_limits.pp_error
    (Scan_limits.create ~max_state:100 ~max_updates:1000L)

let env =
  {
    Eval.Env.load = (fun _ _ -> assert false);
    load_index = (fun _ _ -> assert false);
  }

let origin = Coord.of_fn (fun _ -> 0)
let pos n = Index.clamp_low (Index.const n)

(* trace.(0, l) = 0; trace.(s+1, l) = trace.(s, l) + 1 -- so trace.(r, l) = r
   for every lane, independent of [l]. The simplest body that still exercises
   [previous_at] at the reading lane's own position. *)
let counter ~width ~steps ~limits =
  Builder.run
    (Builder.scan ~limits ~width ~steps
       ~init:(fun ~lane:_ -> Builder.return (Value.const 0.))
       ~update:(fun ~step:_ ~lane ~previous_at ->
         Builder.return (Value.add (previous_at lane) (Value.const 1.))))

let ok_counter ~width ~steps ~limits =
  Err.or_raise ~pp_error:Scan.pp_error (counter ~width ~steps ~limits)

let eval ?scan ?scan_meter e = Eval.value ?scan ?scan_meter env ~output:origin e

let show_float = function
  | Ok x -> Fmt.str "%g" x
  | Error e -> Fmt.str "%a" Core.Pretty.(error_kind Eval.pp_error) e

let%expect_test "the trace runs the recurrence over exactly `row` steps" =
  let s = ok_counter ~width:3 ~steps:5 ~limits in
  let meter = Scan_meter.create ~limits in
  List.iter
    (fun (row, lane) ->
      Fmt.pr "row=%d lane=%d: %s@." row lane
        (show_float
           (eval ~scan_meter:meter
              (Value.scan_at s ~row:(pos row) ~lane:(pos lane)))))
    [ (0, 0); (0, 2); (3, 0); (3, 2); (5, 1) ];
  [%expect
    {|
    row=0 lane=0: 0
    row=0 lane=2: 0
    row=3 lane=0: 3
    row=3 lane=2: 3
    row=5 lane=1: 5 |}]

let%expect_test "zero steps: the trace is just the initial row" =
  let s = ok_counter ~width:2 ~steps:0 ~limits in
  let meter = Scan_meter.create ~limits in
  Fmt.pr "row=0 lane=0: %s@."
    (show_float
       (eval ~scan_meter:meter
          (Value.scan_at s ~row:Index.zero ~lane:Index.zero)));
  Fmt.pr "row=1 (out of range): %s@."
    (show_float
       (eval ~scan_meter:meter (Value.scan_at s ~row:(pos 1) ~lane:Index.zero)));
  [%expect
    {|
    row=0 lane=0: 0
    row=1 (out of range): scan row 1 out of range [0,1) |}]

let%expect_test "row wins on a simultaneous row/lane failure" =
  let s = ok_counter ~width:2 ~steps:2 ~limits in
  let meter = Scan_meter.create ~limits in
  Fmt.pr "%s@."
    (show_float
       (eval ~scan_meter:meter (Value.scan_at s ~row:(pos 9) ~lane:(pos 9))));
  Fmt.pr "%s@."
    (show_float
       (eval ~scan_meter:meter (Value.scan_at s ~row:(pos 1) ~lane:(pos 9))));
  [%expect
    {|
    scan row 9 out of range [0,3)
    scan lane 9 out of range [0,2) at row 1 |}]

let%expect_test
    "an inline scan with no meter fails before evaluating either body" =
  let s = ok_counter ~width:2 ~steps:2 ~limits in
  Fmt.pr "%s@."
    (show_float (eval (Value.scan_at s ~row:Index.zero ~lane:Index.zero)));
  [%expect {| an inline scan requires a meter |}]

let%expect_test "exact-limit and next-charge behavior" =
  (* [width * steps] updates exactly clears a meter built with that many; one
     more (a wider row) exhausts it on the very next charge. *)
  let s = ok_counter ~width:2 ~steps:3 ~limits in
  let at_limit =
    Err.or_raise ~pp_error:Scan_limits.pp_error
      (Scan_limits.create ~max_state:100 ~max_updates:6L)
  in
  let meter = Scan_meter.create ~limits:at_limit in
  (* Sequenced explicitly: OCaml's argument evaluation order is unspecified,
     and [Fmt.pr]'s two arguments must not race to observe [meter] before and
     after its charges. *)
  let result =
    show_float
      (eval ~scan_meter:meter (Value.scan_at s ~row:(pos 3) ~lane:Index.zero))
  in
  Fmt.pr "row=3 (exactly 6 updates): %s remaining=%Ld@." result
    (Scan_meter.remaining meter);
  let one_short =
    Err.or_raise ~pp_error:Scan_limits.pp_error
      (Scan_limits.create ~max_state:100 ~max_updates:5L)
  in
  let meter2 = Scan_meter.create ~limits:one_short in
  Fmt.pr "row=3 (one update short): %s@."
    (show_float
       (eval ~scan_meter:meter2 (Value.scan_at s ~row:(pos 3) ~lane:Index.zero)));
  [%expect
    {|
    row=3 (exactly 6 updates): 3 remaining=0
    row=3 (one update short): scan updates exhausted at limit 5 |}]

let%expect_test "construction rejects a nonpositive step count or width" =
  let show r =
    match r with
    | Ok _ -> "ok"
    | Error e -> Fmt.str "%a" Scan.pp_error (Err.Error.kind e)
  in
  Fmt.pr "%s@." (show (counter ~width:2 ~steps:(-1) ~limits));
  Fmt.pr "%s@." (show (counter ~width:0 ~steps:2 ~limits));
  Fmt.pr "%s@." (show (counter ~width:(-3) ~steps:2 ~limits));
  [%expect
    {|
    invalid scan step count -1
    invalid scan width 0
    invalid scan width -3 |}]

let%expect_test
    "construction rejects the descriptor's worst case against limits" =
  let tight_state =
    Err.or_raise ~pp_error:Scan_limits.pp_error
      (Scan_limits.create ~max_state:3 ~max_updates:1000L)
  in
  let tight_updates =
    Err.or_raise ~pp_error:Scan_limits.pp_error
      (Scan_limits.create ~max_state:100 ~max_updates:5L)
  in
  let show r =
    match r with
    | Ok _ -> "ok"
    | Error e -> Fmt.str "%a" Scan.pp_error (Err.Error.kind e)
  in
  (* width=2 needs 2*2=4 state, over a max_state of 3. *)
  Fmt.pr "%s@." (show (counter ~width:2 ~steps:3 ~limits:tight_state));
  (* width=2,steps=3 needs 6 updates worst case, over a max_updates of 5, even
     though a row-zero projection would need none. *)
  Fmt.pr "%s@." (show (counter ~width:2 ~steps:3 ~limits:tight_updates));
  (* max_state = 0 forbids every scan; max_updates = 0 still admits a
     steps=0 descriptor, whose trace is just the initial row. *)
  let zero_state =
    Err.or_raise ~pp_error:Scan_limits.pp_error
      (Scan_limits.create ~max_state:0 ~max_updates:1000L)
  in
  let zero_updates =
    Err.or_raise ~pp_error:Scan_limits.pp_error
      (Scan_limits.create ~max_state:100 ~max_updates:0L)
  in
  Fmt.pr "%s@." (show (counter ~width:1 ~steps:0 ~limits:zero_state));
  Fmt.pr "%s@." (show (counter ~width:1 ~steps:0 ~limits:zero_updates));
  Fmt.pr "%s@." (show (counter ~width:1 ~steps:1 ~limits:zero_updates));
  [%expect
    {|
    scan state exceeds limit 3
    scan updates exceed limit 5
    scan state exceeds limit 0
    ok
    scan updates exceed limit 0 |}]

let%expect_test "freshening preserves interpretation and structure" =
  (* Two independent builder runs both mint from ordinal 0, so their raw
     [lane]/[step]/[prev] identities coincide numerically -- exactly the
     collision [freshen] exists to make harmless once composed. *)
  let s1 = ok_counter ~width:2 ~steps:4 ~limits in
  let s2 = ok_counter ~width:2 ~steps:4 ~limits in
  let e1 = Value.scan_at s1 ~row:(pos 3) ~lane:(pos 1) in
  let e2 = Value.scan_at s2 ~row:(pos 3) ~lane:(pos 1) in
  Fmt.pr "structurally equal before freshening: %b@." (Value.equal e1 e2);
  let f2 = Builder.run (Rewrite.freshen e2) in
  Fmt.pr "still equal (up to alpha) after freshening one side: %b@."
    (Value.equal e1 f2);
  let meter = Scan_meter.create ~limits in
  Fmt.pr "same evaluated value: %b@."
    (eval ~scan_meter:meter e1 = eval ~scan_meter:meter f2);
  [%expect
    {|
    structurally equal before freshening: true
    still equal (up to alpha) after freshening one side: true
    same evaluated value: true |}]

let%expect_test "freshening does not touch a captured free reference" =
  (* [update] legitimately reads an earlier Region local -- a free [Local]
     reference that freshening must leave alone, unlike [lane]/[step]/[prev]. *)
  let outer = Builder.run Builder.fresh_local in
  let s =
    Err.or_raise ~pp_error:Scan.pp_error
      (Builder.run
         (Builder.scan ~limits ~width:2 ~steps:3
            ~init:(fun ~lane:_ -> Builder.return (Value.const 0.))
            ~update:(fun ~step:_ ~lane ~previous_at ->
              Builder.return (Value.add (previous_at lane) (Value.local outer)))))
  in
  let e = Value.scan_at s ~row:(pos 2) ~lane:Index.zero in
  Fmt.pr "captured local free before freshening: %b@."
    (Local_var.Set.mem outer (Fold.locals e));
  let f = Builder.run (Rewrite.freshen e) in
  Fmt.pr "still free, same identity, after freshening: %b@."
    (Local_var.Set.mem outer (Fold.locals f));
  [%expect
    {|
    captured local free before freshening: true
    still free, same identity, after freshening: true |}]

let%expect_test "a cached Local_scan_at read delegates to the supplied reader" =
  let id = Builder.run Builder.fresh_local in
  let reader : Eval.scan_reader =
   fun v ~row ~lane ->
    if not (Local_var.equal v id) then Err.fail (Eval.Unknown_local v)
    else if row < 0 || row > 4 then
      Err.fail
        (Eval.Row_out_of_range
           {
             Eval.Scan_bounds.projection =
               { Eval.Scan_projection.local = Some v; row; lane };
             extent = 5;
           })
    else Err.return (float_of_int (row + lane))
  in
  let e = Value.local_scan_at id ~row:(pos 3) ~lane:(pos 2) in
  Fmt.pr "%s@." (show_float (eval ~scan:reader e));
  Fmt.pr "%s@."
    (show_float
       (eval ~scan:reader
          (Value.local_scan_at id ~row:(pos 9) ~lane:Index.zero)));
  Fmt.pr "%s@." (show_float (eval e));
  [%expect
    {|
    5
    scan row 9 out of range [0,5)
    unknown trace local #0 |}]

let%expect_test "Scan_admission rejects a scan under an unbounded reduction" =
  let s = ok_counter ~width:2 ~steps:2 ~limits in
  let e = Value.scan_at s ~row:Index.zero ~lane:Index.zero in
  let show r =
    match r with
    | Ok () -> "ok"
    | Error e -> Fmt.str "%a" Scan.pp_error (Err.Error.kind e)
  in
  Fmt.pr "top level: %s@." (show (Scan_admission.check ~limits e));
  (* A statically constant-extent reduction multiplies the worst case (here
     3 * (2 * 2) = 12 updates) rather than rejecting outright. *)
  let under_constant =
    Builder.run
      (Builder.reduction ~kind:Reduction.Sum ~lo:Index.zero ~hi:(Index.const 3)
         (fun _ -> Builder.return e))
  in
  Fmt.pr "under a constant reduction, within budget: %s@."
    (show (Scan_admission.check ~limits under_constant));
  let tight =
    Err.or_raise ~pp_error:Scan_limits.pp_error
      (Scan_limits.create ~max_state:100 ~max_updates:10L)
  in
  Fmt.pr "under a constant reduction, past budget (12 > 10): %s@."
    (show (Scan_admission.check ~limits:tight under_constant));
  (* [hi] depending on the output coordinate cannot be bounded statically. *)
  let under_dynamic =
    Builder.run
      (Builder.reduction ~kind:Reduction.Sum ~lo:Index.zero
         ~hi:(Index.of_position (Index.output Axis.C))
         (fun _ -> Builder.return e))
  in
  Fmt.pr "under a dynamic reduction: %s@."
    (show (Scan_admission.check ~limits under_dynamic));
  [%expect
    {|
    top level: ok
    under a constant reduction, within budget: ok
    under a constant reduction, past budget (12 > 10): scan updates exceed limit 10
    under a dynamic reduction: scan nested under a statically unbounded reduction |}]
