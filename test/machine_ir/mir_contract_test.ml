open Machine_ir

(* The comparison, provenance and failure-record contracts, by their negative
   cases: each deliberately different observation has a defined failing
   verdict, and nothing passes because "both failed". *)

module O = Mir_observation

let src = Expr.Source.create 3
let f64 x = Some (Mir_const.f64 x)

let ok ?(events = [ (Mir_event.Reduction, 4L) ]) cells =
  {
    O.status = O.Status.Success;
    outputs = [ { O.Output.source = src; cells } ];
    events;
  }

let row ?(payload = []) ?site ?invocation failure =
  { O.Row.failure; payload; invocation; site }

let failed ?(events = []) r =
  { O.status = O.Status.Failure r; outputs = []; events }

let status s = { O.status = s; outputs = []; events = [] }

let verdict ?sites e a =
  match Mir_compare.observations ?sites ~expected:e ~actual:a () with
  | Ok () -> "agree"
  | Error d -> Fmt.str "%a" Mir_compare.Difference.pp d

let var = Expr.Builder.run Expr.Builder.fresh_local

let%expect_test "comparator verdicts" =
  let show label e a = Fmt.pr "%-28s %s@." label (verdict e a) in
  show "same outputs" (ok [| f64 1.; f64 2. |]) (ok [| f64 1.; f64 2. |]);
  show "changed output" (ok [| f64 1.; f64 2. |]) (ok [| f64 1.; f64 3. |]);
  show "missing output" (ok [| f64 1. |]) { (ok [||]) with O.outputs = [] };
  show "undefined cell" (ok [| f64 1. |]) (ok [| None |]);
  show "signed zero" (ok [| f64 0. |]) (ok [| f64 (-0.) |]);
  show "NaN payloads"
    (ok [| f64 Float.nan |])
    (ok
       [| Some { Mir_const.ty = Mir_type.F64; bits = 0xFFF8_0000_0000_0001L } |]);
  show "event count"
    (ok [| f64 1. |])
    (ok ~events:[ (Mir_event.Reduction, 3L) ] [| f64 1. |]);
  let ovf = Mir_failure.Index_overflow Mir_failure.Overflow_op.Add in
  let p a b = [ Mir_const.i64 a; Mir_const.i64 b ] in
  show "same failure"
    (failed (row ~payload:(p 1L 2L) ovf))
    (failed (row ~payload:(p 1L 2L) ovf));
  show "changed payload"
    (failed (row ~payload:(p 1L 2L) ovf))
    (failed (row ~payload:(p 1L 3L) ovf));
  show "changed kind"
    (failed (row ~payload:(p 1L 2L) ovf))
    (failed
       (row ~payload:(p 1L 2L)
          (Mir_failure.Index_overflow Mir_failure.Overflow_op.Mul)));
  show "changed invocation"
    (failed (row ~invocation:0l Mir_failure.I64_division_by_zero))
    (failed (row ~invocation:1l Mir_failure.I64_division_by_zero));
  show "failure prefix events"
    (failed
       ~events:[ (Mir_event.Key, 2L) ]
       (row Mir_failure.I64_division_by_zero))
    (failed
       ~events:[ (Mir_event.Key, 1L) ]
       (row Mir_failure.I64_division_by_zero));
  show "failure vs defect"
    (failed (row Mir_failure.I64_division_by_zero))
    (status (O.Status.Defect O.Defect.Domain));
  show "defect vs defect"
    (status (O.Status.Defect O.Defect.Domain))
    (status (O.Status.Defect O.Defect.Domain));
  show "fuel vs fuel"
    (status O.Status.Fuel_exhausted)
    (status O.Status.Fuel_exhausted);
  show "unsupported"
    (status (O.Status.Unsupported "exp"))
    (status (O.Status.Unsupported "exp"));
  show "success vs failure"
    (ok [| f64 1. |])
    (failed (row Mir_failure.I64_division_by_zero));
  [%expect
    {|
    same outputs                 agree
    changed output               output t3[1]: 0x1p+1:f64 vs 0x1.8p+1:f64
    missing output               output t3 missing
    undefined cell               output t3[0]: 0x1p+0:f64 vs undefined
    signed zero                  output t3[0]: 0x0p+0:f64 vs -0x0p+0:f64
    NaN payloads                 agree
    event count                  event reduction: 4 vs 3
    same failure                 agree
    changed payload              failure row: index_overflow(add)(1:i64, 2:i64) vs index_overflow(add)(1:i64, 3:i64)
    changed kind                 failure row: index_overflow(add)(1:i64, 2:i64) vs index_overflow(mul)(1:i64, 2:i64)
    changed invocation           failure row: i64_division_by_zero() invocation 0 vs i64_division_by_zero() invocation 1
    failure prefix events        event key: 2 vs 1
    failure vs defect            inconclusive: failure(i64_division_by_zero) vs defect(domain)
    defect vs defect             inconclusive: defect(domain) vs defect(domain)
    fuel vs fuel                 inconclusive: fuel vs fuel
    unsupported                  inconclusive: unsupported(exp) vs unsupported(exp)
    success vs failure           status: success vs failure(i64_division_by_zero) |}]

let%expect_test "raw sites compare only under a shared numbering" =
  let f = Mir_failure.Unbound_local var in
  let a = failed (row ~site:(Mir_id.Site.of_int 0) f)
  and b = failed (row ~site:(Mir_id.Site.of_int 3) f) in
  Fmt.pr "decoded: %s@.normalized: %s@." (verdict a b)
    (verdict ~sites:Mir_compare.Sites.Normalized a b);
  [%expect
    {|
    decoded: agree
    normalized: failure row: unbound_local(#0)() site0 vs unbound_local(#0)() site3 |}]

let%expect_test
    "an unfused Machine IR FMA fails even where a relaxed engine passes" =
  (* a*b+c with a = 1+2^-27, b = 1-2^-27, c = -1: fused gives -2^-54, the
     separately rounded product gives 0 *)
  let a = 1. +. Float.ldexp 1. (-27) and b = 1. -. Float.ldexp 1. (-27) in
  let fused = Float.fma a b (-1.)
  and unfused = Sys.opaque_identity (a *. b) +. -1. in
  Fmt.pr "fused %h unfused %h@." fused unfused;
  let fused = ok [| f64 fused |] and unfused = ok [| f64 unfused |] in
  let planning fma =
    Mir_planning.make ~subject:"p" ~policy:"simd_fp32_relaxed"
      ~schedule:"wasm128" ~precision:Mir_planning.Precision.F32
      ~lanes:(Mir_type.Lanes.of_int 4) ~fma ~capabilities:[]
  in
  let relaxed fma external_ =
    match
      Mir_compare.relaxed_external ~planning:(planning fma) ~fused ~unfused
        ~external_
    with
    | Ok () -> "agree"
    | Error d -> Fmt.str "%a" Mir_compare.Difference.pp d
  in
  Fmt.pr "external unfused, relaxed_madd: %s@."
    (relaxed Mir_planning.Fma.Relaxed_madd unfused);
  Fmt.pr "external unfused, exact:        %s@."
    (relaxed Mir_planning.Fma.Exact unfused);
  Fmt.pr "machine IR unfused vs fused:    %s@." (verdict fused unfused);
  [%expect
    {|
    fused -0x1p-54 unfused 0x0p+0
    external unfused, relaxed_madd: agree
    external unfused, exact:        output t3[0]: -0x1p-54:f64 vs 0x0p+0:f64
    machine IR unfused vs fused:    output t3[0]: -0x1p-54:f64 vs 0x0p+0:f64 |}]

let%expect_test "planning provenance admission" =
  let s =
    Mir_planning.make ~subject:"digest-1" ~policy:"reference_f64"
      ~schedule:"scalar" ~precision:Mir_planning.Precision.F64
      ~lanes:(Mir_type.Lanes.of_int 1) ~fma:Mir_planning.Fma.Forbidden
      ~capabilities:
        [
          Mir_planning.Capability.Helper "exp";
          Mir_planning.Capability.Fused_multiply_add;
        ]
  in
  let show label r =
    Fmt.pr "%-22s %s@." label
      (match r with
      | Ok _ -> "admitted"
      | Error e -> Fmt.str "%a" Mir_planning.pp_mismatch e)
  in
  show "bound"
    (Mir_planning.admit (Some s) ~subject:"digest-1" ~contracts:false);
  show "missing" (Mir_planning.admit None ~subject:"digest-1" ~contracts:false);
  show "other program"
    (Mir_planning.admit (Some s) ~subject:"digest-2" ~contracts:false);
  show "unauthorized fma"
    (Mir_planning.admit (Some s) ~subject:"digest-1" ~contracts:true);
  let text = Mir_planning.to_string s in
  print_endline text;
  (match Mir_planning.of_string text with
  | Ok t -> Fmt.pr "round trip: %b@." (Mir_planning.equal s t)
  | Error (`Malformed_summary w) -> Fmt.pr "malformed: %s@." w);
  (match Mir_planning.of_string "subject=x\nfma=sometimes" with
  | Ok _ -> print_endline "accepted"
  | Error (`Malformed_summary w) -> Fmt.pr "malformed: %s@." w);
  [%expect
    {|
    bound                  admitted
    missing                no planning summary
    other program          planning summary is for digest-1, not digest-2
    unauthorized fma       a fused multiply-add under fma=forbidden
    subject=digest-1
    policy=reference_f64
    schedule=scalar
    precision=f64
    lanes=1
    fma=forbidden
    capabilities=fma,helper:exp
    round trip: true
    malformed: missing or unreadable field |}]

let%expect_test "failure records: sites, sentinels and kinds without sites" =
  let other = Expr.Builder.run Expr.Builder.fresh_local in
  let table =
    Mir_failure.Site_entry.
      [|
        Other;
        Local_out_of_range other;
        Scan_row_out_of_range None;
        Local_out_of_range var;
        Local_out_of_range var;
      |]
  in
  let show f =
    match Mir_failure.bind_site ~table f with
    | Some s -> Fmt.pr "%a -> %a@." Mir_failure.pp f Mir_id.Site.pp s
    | None -> Fmt.pr "%a -> no site@." Mir_failure.pp f
  in
  (* the first compatible entry; a reachable kind with none is a refusal;
     a kind with no site word needs no entry *)
  show (Mir_failure.Unbound_local var);
  show
    (Mir_failure.Scan_projection
       { Mir_failure.Scan.which = Mir_failure.Scan_axis.Row; var = None });
  show
    (Mir_failure.Scan_projection
       { Mir_failure.Scan.which = Mir_failure.Scan_axis.Lane; var = None });
  show Mir_failure.I64_division_by_zero;
  let decode kind v =
    match Mir_failure.decode ~table ~kind ~v with
    | Ok (f, payload, site) ->
        Fmt.pr "%a payload [%a] site %a@." Mir_failure.pp f
          Fmt.(list ~sep:(any ",") int64)
          payload
          Fmt.(option ~none:(any "-") Mir_id.Site.pp)
          site
    | Error e -> Fmt.pr "%a@." Mir_failure.Decode_error.pp e
  in
  let words f payload site =
    Mir_failure.words f ~payload ~site:(Option.map Mir_id.Site.of_int site)
  in
  (* equivalent raw sites 3 and 4 decode to the same identity *)
  decode 11l (words (Mir_failure.Unbound_local var) [] (Some 3));
  decode 11l (words (Mir_failure.Unbound_local var) [] (Some 4));
  (* the one-past-table sentinel is a defect *)
  decode 11l (words (Mir_failure.Unbound_local var) [] (Some 5));
  (* a site whose entry is another kind's *)
  decode 11l (words (Mir_failure.Unbound_local var) [] (Some 2));
  let coord =
    Mir_failure.Coord_out_of_range
      { Mir_failure.Coord.source = src; axis = Expr.Axis.H }
  in
  let w = words coord [ 0L; 0L; 0L; 9L; 1L; 0L ] None in
  Fmt.pr "coord words [%a]@." Fmt.(array ~sep:(any ",") int64) w;
  decode 0l w;
  decode 1l w;
  [%expect
    {|
    unbound_local(#0) -> site1
    scan_projection(row, inline) -> site2
    scan_projection(lane, inline) -> no site
    i64_division_by_zero -> no site
    unbound_local(#0) payload [] site site3
    unbound_local(#0) payload [] site site4
    the record names the sentinel site
    the record disagrees with its site
    coord words [3,3,9,0,0,0,9,1,0,0,0,0]
    coord_out_of_range(t3, H) payload [0,0,0,9,1,0] site -
    unknown failure kind 1 |}]
