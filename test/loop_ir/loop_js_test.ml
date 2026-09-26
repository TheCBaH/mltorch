open Loop_ir
open Loop_fixtures

let emit = Loop_js.emit

let%expect_test "emitted JavaScript is a named function over typed arrays" =
  print_string (emit Loop_programs.doubling);
  [%expect
    {|
    "use strict";
    function loop_kernel(b0, b1) {
      for (let i0 = 0; i0 < 4; i0++) {
        b1[i0] = b0[i0] * 2;
      }
      return null;
    } |}]

let%expect_test "emitting twice is byte-identical, whatever the allocation ids"
    =
  let build ~var ~temp:t =
    let i = Loop_index.Var (v var) in
    program
      ~buffers:[ buffer 1 (shape_w 2) f32 Loop_buffer.Output ]
      [
        Loop_stmt.For
          {
            var = v var;
            lo = Loop_index.Const 0;
            hi = Loop_index.Const 2;
            body =
              [
                Loop_stmt.Assign
                  (Loop_carrier.Float, temp t, Loop_expr.Value_of_index i);
                Loop_stmt.Store
                  {
                    buffer = buffer 1 (shape_w 2) f32 Loop_buffer.Output;
                    coord = at_w i;
                    value =
                      Loop_stored.F32
                        (Loop_expr.Temp (Loop_carrier.Float, temp t));
                  };
              ];
          };
      ]
  in
  let a = emit (build ~var:0 ~temp:0) in
  Fmt.pr "%b@." (String.equal a (emit (build ~var:0 ~temp:0)));
  Fmt.pr "%b@." (String.equal a (emit (build ~var:31 ~temp:77)));
  [%expect {|
    true
    true |}]

(* ---- the failure records, one per [Loop_failure.t] constructor ------------- *)

let entry_body p =
  Js_print.stmts (Loop_js.to_ast p).Js_ast.Program.entry.Js_ast.Func.body

let i0 = Loop_index.Var (v 0)

(* One [Fail_if] inside a one-iteration loop, so the loop variable is bound. *)
let failing ?(buffers = []) pred failure =
  program ~buffers
    [
      Loop_stmt.For
        {
          var = v 0;
          lo = Loop_index.Const 0;
          hi = Loop_index.Const 1;
          body = [ Loop_stmt.Fail_if (pred, failure) ];
        };
    ]

let always = Loop_bool.Index_lt (Loop_index.Const 0, Loop_index.Const 1)
let local = Expr.Builder.run Expr.Builder.fresh_local

let%expect_test "each failure constructor's record, as emitted" =
  let b = buffer 1 (shape_w 4) f32 Loop_buffer.Input in
  let show name p = Fmt.pr "-- %s@.%s" name (entry_body p) in
  show "gather"
    (failing always
       (Loop_failure.Gather_out_of_range
          { raw = Loop_expr.I64_const (-5L); extent = 3 }));
  show "i64 division by zero" (failing always Loop_failure.I64_division_by_zero);
  show "i64 division overflow"
    (failing always Loop_failure.I64_division_overflow);
  show "i64 from float"
    (failing always
       (Loop_failure.I64_from_float { value = Loop_expr.Const 1. }));
  let overflowing =
    Loop_index.Add (Loop_index.Scale (3, i0), Loop_index.Const 5)
  in
  show "index overflow"
    (failing (Loop_bool.Index_overflows overflowing)
       (Loop_failure.Index_overflow { index = overflowing }));
  show "load out of range"
    (failing ~buffers:[ b ] always
       (Loop_failure.Load_out_of_range { buffer = b; coord = at_w i0 }));
  show "local out of range"
    (failing always
       (Loop_failure.Local_out_of_range { local; index = i0; extent = 2 }));
  show "scan lane"
    (failing always
       (Loop_failure.Scan_lane_out_of_range
          {
            local = Some local;
            row = i0;
            lane = Loop_index.Const 1;
            extent = 2;
          }));
  show "scan row"
    (failing always
       (Loop_failure.Scan_row_out_of_range
          { local = None; row = i0; lane = Loop_index.Const 1; extent = 2 }));
  [%expect
    {|
    -- gather
    for (let i0 = 0; i0 < 1; i0++) {
      if (0 < 1) {
        return { kind: "gather_index_out_of_range", raw: (-5n).toString(), extent: 3 };
      }
    }
    return null;
    -- i64 division by zero
    for (let i0 = 0; i0 < 1; i0++) {
      if (0 < 1) {
        return { kind: "i64_division_by_zero" };
      }
    }
    return null;
    -- i64 division overflow
    for (let i0 = 0; i0 < 1; i0++) {
      if (0 < 1) {
        return { kind: "i64_division_overflow" };
      }
    }
    return null;
    -- i64 from float
    for (let i0 = 0; i0 < 1; i0++) {
      if (0 < 1) {
        return i64_from_float_failure(1);
      }
    }
    return null;
    -- index overflow
    for (let i0 = 0; i0 < 1; i0++) {
      if (3 * i0 < -2147483648 || 3 * i0 >= 2147483648) {
        return { kind: "index_overflow", op: "mul", lhs: 3, rhs: i0 };
      }
      if (3 * i0 + 5 < -2147483648 || 3 * i0 + 5 >= 2147483648) {
        return { kind: "index_overflow", op: "add", lhs: 3 * i0, rhs: 5 };
      }
    }
    return null;
    -- load out of range
    for (let i0 = 0; i0 < 1; i0++) {
      if (0 < 1) {
        return coord_failure(1, [1, 1, 1, 1, 4, 1], [0, 0, 0, 0, i0, 0]);
      }
    }
    return null;
    -- local out of range
    for (let i0 = 0; i0 < 1; i0++) {
      if (0 < 1) {
        return { kind: "unbound_local", site: 0 };
      }
    }
    return null;
    -- scan lane
    for (let i0 = 0; i0 < 1; i0++) {
      if (0 < 1) {
        return { kind: "scan_projection", which: "lane", cached: true, row: i0, lane: 1, extent: 2, site: 0 };
      }
    }
    return null;
    -- scan row
    for (let i0 = 0; i0 < 1; i0++) {
      if (0 < 1) {
        return { kind: "scan_projection", which: "row", cached: false, row: i0, lane: 1, extent: 2, site: 0 };
      }
    }
    return null; |}]

(* ---- the prelude ---------------------------------------------------------- *)

(* The functions the script declares, in order: what the prelude chose. A text
   search for a helper's name would also find [f16_to_float] inside
   [bf16_to_float], which is the mechanism this replaced. *)
let declared_functions script =
  String.split_on_char '\n' script
  |> List.filter_map (fun line ->
      match String.index_opt line '(' with
      | Some i when String.length line > 9 && String.sub line 0 9 = "function "
        ->
          Some (String.sub line 9 (i - 9))
      | _ -> None)

let format_program fmt =
  match
    Err.payload
      (Loop_lower.lower
         (Fusion_plan.default
            (Loop_programs.format_kernel ~fmt:(Payload.Fmt fmt) 4)))
  with
  | Ok p -> p
  | Error _ -> failwith "format_program: refused"

let%expect_test "the prelude holds the helpers a program uses, and only those" =
  List.iter
    (fun (name, p) ->
      Fmt.pr "%s: %s@." name (String.concat ", " (declared_functions (emit p))))
    [
      ("doubling", Loop_programs.doubling);
      ("bf16 load", format_program Payload.BF16);
      ("f16 load", format_program Payload.F16);
    ];
  [%expect
    {|
    doubling: loop_kernel
    bf16 load: bf16_to_float, loop_kernel
    f16 load: f16_to_float, loop_kernel
    |}]
