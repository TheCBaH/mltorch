open Loop_ir
module F = Loop_js_failure

(* A placeholder value per field, named after the field, so the printed record
   shows which key each one is written under. *)
let placeholder f =
  Js_build.expr (Js_build.Num.var (Js_ident.v (F.Field.to_string f)))

let record kind =
  F.record kind (List.map (fun f -> (f, placeholder f)) (F.fields kind))

let%expect_test "the record each kind writes: one line per kind" =
  List.iter (fun k -> Fmt.pr "%s@." (Js_print.expr (record k))) F.Kind.all;
  [%expect
    {|
    { kind: "coord_out_of_range", buffer: buffer, axis: axis, index: index, coord: coord }
    { kind: "defect" }
    { kind: "gather_index_out_of_range", raw: raw, extent: extent }
    { kind: "i64_division_by_zero" }
    { kind: "i64_division_overflow" }
    { kind: "i64_from_float_infinite" }
    { kind: "i64_from_float_nan" }
    { kind: "i64_from_float_out_of_range", value: value }
    { kind: "index_overflow", op: op, lhs: lhs, rhs: rhs }
    { kind: "scan_meter", which: which, limit: limit }
    { kind: "scan_projection", which: which, cached: cached, row: row, lane: lane, extent: extent, site: site }
    { kind: "unbound_local", site: site }
    |}]

let%expect_test "kinds and closed string values read back what they print" =
  let ok =
    List.for_all
      (fun k -> F.Kind.of_string (F.Kind.to_string k) = Some k)
      F.Kind.all
    && List.for_all
         (fun v -> F.Overflow_op.of_string (F.Overflow_op.to_string v) = Some v)
         [ F.Overflow_op.Add; Mul ]
    && List.for_all
         (fun v -> F.Projection.of_string (F.Projection.to_string v) = Some v)
         [ F.Projection.Lane; Row ]
    && List.for_all
         (fun v -> F.Meter.of_string (F.Meter.to_string v) = Some v)
         [ F.Meter.State_over_limit; Updates_exhausted ]
    && F.Kind.of_string "nonsense" = None
  in
  Fmt.pr "%b@." ok;
  [%expect {| true |}]

let%expect_test "a record with a missing, extra or reordered field is refused" =
  let x = Js_build.expr (Js_build.Num.const 0.) in
  let attempt name kind fields =
    match F.record kind fields with
    | _ -> Fmt.pr "%s: accepted@." name
    | exception Invalid_argument m -> Fmt.pr "%s: %s@." name m
  in
  attempt "missing" F.Kind.Gather_index_out_of_range [ (F.Field.Raw, x) ];
  attempt "extra" F.Kind.Defect [ (F.Field.Extent, x) ];
  attempt "reordered" F.Kind.Gather_index_out_of_range
    [ (F.Field.Extent, x); (F.Field.Raw, x) ];
  attempt "exact" F.Kind.Gather_index_out_of_range
    [ (F.Field.Raw, x); (F.Field.Extent, x) ];
  [%expect
    {|
    missing: Loop_js_failure.record gather_index_out_of_range: fields must be [raw; extent]
    extra: Loop_js_failure.record defect: fields must be []
    reordered: Loop_js_failure.record gather_index_out_of_range: fields must be [raw; extent]
    exact: accepted
    |}]
