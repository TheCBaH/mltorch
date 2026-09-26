(* The generated-code gate's generator. For each hand-built case it writes the
   emitted JavaScript plus a driver into run.js, and the Loop interpreter's
   rendering of the very same run into expected.txt. `node run.js` must print
   expected.txt byte for byte: floats as the hex of their binary64 bits (NaN as
   "nan"), so signed zeros and the f32 round are compared, not merely printed. *)

open Loop_ir
open Loop_ir_test
open Loop_fixtures

(* What a buffer is bound to. [Cells] are raw storage integers for a format
   that is not a float ([F16] and [BF16] bit patterns, [Bool] bytes, quantized and
   integer cells); [F32] and [F64] carry floats. *)
type input =
  | Cells of {
      fmt : Payload.packed_fmt;
      quant : Quant.t option;
      cells : int64 array;
    }
  | F32 of float array
  | F64 of float array

type case = {
  name : string;
  program : Loop_program.t;
  inputs : (int * input) list;
}

let hex x =
  if Float.is_nan x then "nan"
  else Printf.sprintf "%016Lx" (Int64.bits_of_float x)

let constant =
  let out = buffer 1 (shape_w 4) f32 Loop_buffer.Output in
  let w = Loop_index.Var (v 0) in
  program ~buffers:[ out ]
    [
      Loop_stmt.For
        {
          var = v 0;
          lo = Loop_index.Const 0;
          hi = Loop_index.Const 4;
          body =
            [
              Loop_stmt.Store
                {
                  buffer = out;
                  coord = at_w w;
                  (* 2^24 + 1 needs 25 bits: the f32 store must round it, and the
                     literal must survive emission in full to be rounded. *)
                  value =
                    Loop_stored.F32
                      (Loop_expr.Binary
                         ( Expr.Value.Add,
                           Loop_expr.Const 16777217.,
                           Loop_expr.Value_of_index w ));
                };
            ];
        };
    ]

let lowered name plan inputs =
  match Err.payload (Loop_lower.lower plan) with
  | Ok program -> { name; program; inputs }
  | Error (`Unsupported u) ->
      Fmt.failwith "loop_js_gen: %s refused: %a" name Loop_unsupported.pp u

let kernel name k inputs = lowered name (Fusion_plan.default k) inputs
let i64_fmt = Payload.Fmt Payload.I64

(* One int64 value over the two inputs [Loop_programs.i64_body_kernel] declares:
   t0 the floats, t2 the int64 cells. *)
let int64_case name ?(floats = [| 2.9; -2.9; 0.; 0. |]) ~cells body =
  kernel name
    (Loop_programs.i64_body_kernel body)
    [ (0, F32 floats); (2, Cells { fmt = i64_fmt; quant = None; cells }) ]

let big = Int64.add (Int64.shift_left 1L 53) 1L
let here = Loop_programs.i64_here

let float_to_i64_case name x =
  int64_case name ~floats:[| x; x; x; x |] ~cells:[| 0L; 0L; 0L; 0L |]
    (Expr.Value.float_to_i64
       (Expr.Value.load
          (Expr_bridge.source_of_id (tid 0))
          Loop_programs.out_coord))

let gather_case name cells =
  kernel name Loop_programs.gather_kernel
    [
      (0, F32 [| 10.; 20.; 30.; 40. |]);
      (2, Cells { fmt = i64_fmt; quant = None; cells });
    ]

let format_case name ~fmt ?quant cells =
  kernel name
    (Loop_programs.format_kernel ~fmt ?quant (Array.length cells))
    [ (0, Cells { fmt; quant; cells }) ]

let fmt_of_tag f = Payload.Fmt f

let int64_cases =
  let cells = [| Int64.max_int; Int64.min_int; 3L; -5L |] in
  let ties = [| 5L; -9L; 5L; 2L |] in
  let one = Expr.Value.i64_const 1L in
  let reduce kind = Loop_programs.reduce_w kind in
  let empty = (Expr.Index.zero, Expr.Index.const 0) in
  [
    int64_case "i64_add_big"
      ~cells:[| big; Int64.max_int; Int64.min_int; 0L |]
      (Expr.Value.i64_add here one);
    int64_case "i64_add_wrap" ~cells (Expr.Value.i64_add here one);
    int64_case "i64_sub_wrap" ~cells (Expr.Value.i64_sub here one);
    int64_case "i64_mul_wrap" ~cells (Expr.Value.i64_mul here here);
    (* The wrap must happen BEFORE the next operation reads the value: stored
       directly, an unwrapped sum would be wrapped by the typed array anyway. *)
    int64_case "i64_wrap_then_div"
      ~cells:[| Int64.max_int; 1L; 1L; 1L |]
      (Expr.Value.i64_div
         (Expr.Value.i64_add here one)
         (Expr.Value.i64_const 2L));
    int64_case "i64_wrap_then_float"
      ~cells:[| Int64.max_int; 1L; 1L; 1L |]
      (Expr.Value.float_to_i64
         (Expr.Value.i64_to_float (Expr.Value.i64_add here one)));
    int64_case "i64_div_2" ~cells:[| -7L; 7L; -7L; 7L |]
      (Expr.Value.i64_div here (Expr.Value.i64_const 2L));
    int64_case "i64_div_neg2" ~cells:[| -7L; 7L; -7L; 7L |]
      (Expr.Value.i64_div here (Expr.Value.i64_const (-2L)));
    int64_case "i64_div_zero" ~cells:[| -7L; 7L; -7L; 7L |]
      (Expr.Value.i64_div here (Expr.Value.i64_const 0L));
    int64_case "i64_div_overflow"
      ~cells:[| Int64.min_int; 1L; 1L; 1L |]
      (Expr.Value.i64_div here (Expr.Value.i64_const (-1L)));
    int64_case "i64_div_min_by_one"
      ~cells:[| Int64.min_int; 1L; 1L; 1L |]
      (Expr.Value.i64_div here one);
    float_to_i64_case "f2i_2_9" 2.9;
    float_to_i64_case "f2i_neg_2_9" (-2.9);
    float_to_i64_case "f2i_neg_zero" (-0.);
    float_to_i64_case "f2i_min" (-.Float.pow 2. 63.);
    float_to_i64_case "f2i_2_63" (Float.pow 2. 63.);
    float_to_i64_case "f2i_nan" nan;
    float_to_i64_case "f2i_inf" infinity;
    float_to_i64_case "f2i_neg_inf" neg_infinity;
    float_to_i64_case "f2i_1e30" 1e30;
    int64_case "i64_through_float" ~cells:[| big; 3L; -3L; 0L |]
      (Expr.Value.float_to_i64 (Expr.Value.i64_to_float here));
    int64_case "i64_sum" ~cells:ties (reduce Expr.Reduction.Sum ());
    int64_case "i64_max" ~cells:ties (reduce Expr.Reduction.Max ());
    int64_case "i64_argmax_value" ~cells:ties
      (reduce Expr.Reduction.Argmax_value ());
    int64_case "i64_argmax_index" ~cells:ties
      (reduce Expr.Reduction.Argmax_index ());
    int64_case "i64_sum_empty" ~cells:ties
      (reduce Expr.Reduction.Sum ~lo:(fst empty) ~hi:(snd empty) ());
    int64_case "i64_max_empty" ~cells:ties
      (reduce Expr.Reduction.Max ~lo:(fst empty) ~hi:(snd empty) ());
    int64_case "i64_argmax_empty" ~cells:ties
      (reduce Expr.Reduction.Argmax_index ~lo:(fst empty) ~hi:(snd empty) ());
    int64_case "i64_sum_modular"
      ~cells:[| Int64.max_int; 1L; 1L; 1L |]
      (reduce Expr.Reduction.Sum ());
    gather_case "gather_in_range" [| 0L; 1L; 2L; 3L |];
    gather_case "gather_negative" [| -1L; -4L; -2L; -3L |];
    gather_case "gather_extent" [| 4L; 0L; 0L; 0L |];
    gather_case "gather_below_negative_extent" [| -5L; 0L; 0L; 0L |];
    gather_case "gather_min_int" [| Int64.min_int; 0L; 0L; 0L |];
    gather_case "gather_2_32" [| Int64.shift_left 1L 32; 0L; 0L; 0L |];
    kernel "mixed_order" Loop_programs.mixed_kernel
      [
        (0, F32 [| 1.5; 2.5; -1.5; 1e10 |]);
        (2, Cells { fmt = i64_fmt; quant = None; cells = [| 1L; 2L; 3L; 4L |] });
      ];
  ]

(* A load's decode, checked in binary64 rather than through an f32 store that
   would hide a one-ulp difference: each cell of a Bool output is 1 iff the
   decoded value EQUALS the double the OCaml codec gives for that cell. *)
let exact_decode name ~fmt ?quant cells =
  let n = Array.length cells in
  let shape = Loop_programs.shape_c n in
  let sg id fmt =
    Tensor_sig.create ~id:(tid id) ~name:"" ~shape ~fmt
      ?quant:(if id = 0 then quant else None)
      ()
  in
  let input =
    { Loop_buffer.id = tid 0; sg = sg 0 fmt; role = Loop_buffer.Input }
  in
  let output =
    {
      Loop_buffer.id = tid 1;
      sg = sg 1 (Payload.Fmt Payload.Bool);
      role = Loop_buffer.Output;
    }
  in
  let tensor = Loop_programs.raw_tensor fmt ?quant n cells in
  let at i =
    Expr.Coord.make ~n:(Loop_index.Const 0) ~t:(Loop_index.Const 0)
      ~d:(Loop_index.Const 0) ~h:(Loop_index.Const 0) ~w:(Loop_index.Const 0)
      ~c:(Loop_index.Const i)
  in
  let program =
    Loop_fixtures.program ~buffers:[ input; output ]
      (List.init n (fun i ->
           let expected =
             Tensor.read tensor (Vec6.coord ~n:0 ~t:0 ~d:0 ~h:0 ~w:0 ~c:i)
           in
           Loop_stmt.Store
             {
               buffer = output;
               coord = at i;
               value =
                 Loop_stored.Bool
                   (Loop_expr.Select
                      ( Loop_bool.Value_eq
                          ( Loop_expr.Load (input, at i),
                            Loop_expr.Const expected ),
                        Loop_expr.Const 1.,
                        Loop_expr.Const 0. ));
             }))
  in
  { name; program; inputs = [ (0, Cells { fmt; quant; cells }) ] }

let bool_and_format_cases =
  let per_tensor scale zero_point = Quant.per_tensor ~scale ~zero_point in
  let per_channel scale zero_point =
    Err.or_raise ~pp_error:Quant.pp_error (Quant.per_channel ~scale ~zero_point)
  in
  [
    kernel "bool_f32"
      (Loop_programs.bool_kernel ~input_fmt:(fmt_of_tag Payload.F32))
      [ (0, F32 [| nan; -0.; 0.; 1e-40 |]) ];
    kernel "bool_f64_below_f32_range"
      (Loop_programs.bool_kernel ~input_fmt:(fmt_of_tag Payload.F64))
      [ (0, F64 [| 1e-50; 1e-300; -0.; 5e-324 |]) ];
    kernel "bool_fill_zero" (Loop_programs.bool_fill_kernel 0.) [];
    kernel "bool_fill_neg_zero" (Loop_programs.bool_fill_kernel (-0.)) [];
    kernel "bool_fill_half" (Loop_programs.bool_fill_kernel 0.5) [];
    kernel "bool_fill_nan" (Loop_programs.bool_fill_kernel nan) [];
    kernel "bool_fill_tiny" (Loop_programs.bool_fill_kernel 1e-50) [];
    format_case "format_bool_bytes" ~fmt:(fmt_of_tag Payload.Bool)
      [| 0L; 1L; 2L; 255L |];
    format_case "format_f16" ~fmt:(fmt_of_tag Payload.F16)
      [| 0x3c00L; 0x0001L; 0x7c00L; 0xfe00L |];
    format_case "format_f16_edges" ~fmt:(fmt_of_tag Payload.F16)
      [| 0x8000L; 0x7bffL; 0x03ffL; 0x0400L |];
    format_case "format_bf16" ~fmt:(fmt_of_tag Payload.BF16)
      [| 0x3f80L; 0x0001L; 0x7f80L; 0xffc0L |];
    format_case "format_i8_per_tensor" ~fmt:(fmt_of_tag Payload.I8)
      ~quant:(per_tensor 0.1 (-3)) [| -128L; 127L; 0L; -3L |];
    format_case "format_i8_per_channel" ~fmt:(fmt_of_tag Payload.I8)
      ~quant:(per_channel [| 0.5; 0.25; 2.; 0.1 |] [| 0; 1; -2; 5 |])
      [| -128L; 127L; 3L; -4L |];
    format_case "format_i16_per_tensor" ~fmt:(fmt_of_tag Payload.I16)
      ~quant:(per_tensor 0.001 7)
      [| -32768L; 32767L; 0L; 1L |];
    format_case "format_i16_per_channel" ~fmt:(fmt_of_tag Payload.I16)
      ~quant:(per_channel [| 1e-3; 2e-3; 3e-3; 4e-3 |] [| 0; -7; 7; 32767 |])
      [| -32768L; 32767L; 5L; -5L |];
    exact_decode "exact_i8_per_tensor" ~fmt:(fmt_of_tag Payload.I8)
      ~quant:(per_tensor 0.1 3) [| -128L; 127L; 4L; 5L |];
    exact_decode "exact_i16_per_channel" ~fmt:(fmt_of_tag Payload.I16)
      ~quant:(per_channel [| 0.1; 0.3; 0.7; 1e-3 |] [| 3; -7; 32767; -32768 |])
      [| 32767L; -32768L; 12345L; 5L |];
    exact_decode "exact_f16" ~fmt:(fmt_of_tag Payload.F16)
      [| 0x0001L; 0x03ffL; 0x3555L; 0x7bffL |];
    format_case "format_i32" ~fmt:(fmt_of_tag Payload.I32)
      [| Int64.of_int32 Int32.min_int; Int64.of_int32 Int32.max_int; -1L; 0L |];
    kernel "format_f64"
      (Loop_programs.format_kernel ~fmt:(fmt_of_tag Payload.F64) 4)
      [ (0, F64 [| 1e300; 5e-324; -0.; nan |]) ];
    format_case "format_i64_as_float" ~fmt:(fmt_of_tag Payload.I64)
      [| big; Int64.min_int; Int64.max_int; -1L |];
  ]

let rows_data = [| 1.; 2.; 3.; 10.; 20.; 30. |]

let trace_at ~row ~lane =
  Loop_programs.trace_program_at ~steps:2 ~row:(Loop_programs.pick row)
    ~lane:(Loop_programs.pick lane)

let inline name ~row ~lane =
  kernel name
    (Loop_programs.region_kernel_of
       (Region_program.pixel
          (Loop_programs.inline_scan_body_at ~steps:2
             ~row:(Loop_programs.pick row) ~lane:(Loop_programs.pick lane))))
    [ (0, F32 rows_data) ]

(* The lowered inline-scan program under scan limits of its own: a budget below a
   scan's cost cannot be built into a kernel, so the meter's failures are reached
   by replacing the limits the program carries. *)
let metered name ~max_state ~max_updates =
  let c =
    kernel name
      (Loop_programs.inline_scan_kernel ~steps:2)
      [ (0, F32 rows_data) ]
  in
  let limits =
    Err.or_raise ~pp_error:Expr.Scan_limits.pp_error
      (Expr.Scan_limits.create ~max_state ~max_updates)
  in
  { c with program = { c.program with Loop_program.scan_limits = limits } }

let region name program =
  kernel name (Loop_programs.region_kernel_of program) [ (0, F32 rows_data) ]

let unary name op data =
  kernel name (Loop_programs.unary_kernel op) [ (0, F32 data) ]

let reductions =
  let case name kind ?bounds data =
    let lo, hi = Option.value bounds ~default:Loop_programs.full in
    kernel name (Loop_programs.reduction_kernel kind ~lo ~hi) [ (0, F32 data) ]
  in
  let empty = (Expr.Index.zero, Expr.Index.const 0) in
  [
    case "sum_negative_zeros" Expr.Reduction.Sum [| -0.; -0.; -0. |];
    case "sum_nan" Expr.Reduction.Sum [| 1.; nan; 2. |];
    case "max_signed_zeros" Expr.Reduction.Max [| -0.; 0.; -0. |];
    case "max_negative" Expr.Reduction.Max [| -5.; -7.; -6. |];
    case "argmax_ties" Expr.Reduction.Argmax_index [| 1.; 3.; 3. |];
    case "argmax_nan" Expr.Reduction.Argmax_index [| nan; 5.; nan |];
    case "sum_empty" Expr.Reduction.Sum ~bounds:empty [| 1.; 2.; 3. |];
    case "max_empty" Expr.Reduction.Max ~bounds:empty [| 1.; 2.; 3. |];
    case "argmax_empty" Expr.Reduction.Argmax_index ~bounds:empty
      [| 1.; 2.; 3. |];
  ]

let pools =
  let ramp = Array.init 16 float_of_int in
  let with_nan = Array.copy ramp in
  with_nan.(0) <- nan;
  with_nan.(1) <- nan;
  let case name ?(result = Expr.Intrinsic.Max_pool.Value) ~kernel:window ~stride
      ~pad data =
    kernel name
      (Loop_programs.pool_kernel ~input:4 ~out:2 ~kernel:window ~stride ~pad
         ~result)
      [ (0, F32 data) ]
  in
  [
    case "pool_plain" ~kernel:2 ~stride:2 ~pad:0 ramp;
    case "pool_padded" ~kernel:3 ~stride:2 ~pad:1 ramp;
    case "pool_index" ~result:Expr.Intrinsic.Max_pool.Index ~kernel:3 ~stride:2
      ~pad:1 ramp;
    case "pool_nan" ~kernel:2 ~stride:2 ~pad:0 with_nan;
    case "pool_nan_index" ~result:Expr.Intrinsic.Max_pool.Index ~kernel:2
      ~stride:2 ~pad:0 with_nan;
  ]

let cases =
  [
    { name = "constant"; program = constant; inputs = [] };
    {
      name = "doubling";
      program = Loop_programs.doubling;
      inputs = [ (0, F32 Loop_programs.data) ];
    };
    {
      name = "failure";
      program = Loop_programs.shifted_loop ~extent:4;
      inputs = [ (0, F32 Loop_programs.data) ];
    };
    kernel "shifted_load" Loop_programs.shifted_kernel
      [ (0, F32 [| 1.; 2.; 3.; 4. |]) ];
    kernel "index_overflow" Loop_programs.overflow_kernel
      [ (0, F32 [| 0.; 0.; 0.; 0. |]) ];
    kernel "filled"
      (Loop_programs.filled_kernel 16777217.)
      [ (0, F32 [| 1.; 1.; 1.; 1. |]) ];
  ]
  @ reductions @ pools @ int64_cases @ bool_and_format_cases
  @ [
      (let k = Loop_programs.chain ~outputs:[ tid 2 ] in
       lowered "chain_default" (Fusion_plan.default k)
         [ (0, F32 [| Loop_programs.two24; Loop_programs.two24 |]) ]);
      (let k = Loop_programs.chain ~outputs:[ tid 2 ] in
       lowered "chain_fused"
         (fst (Fusion_plan.plan k))
         [ (0, F32 [| Loop_programs.two24; Loop_programs.two24 |]) ]);
      region "region_centered" Loop_programs.centered_program;
      region "region_vector_1"
        (Loop_programs.vector_program ~extent:1 ~pick:(Loop_programs.pick 0));
      region "region_vector_3"
        (Loop_programs.vector_program ~extent:3 ~pick:(Loop_programs.pick 2));
      region "region_vector_past_extent"
        (Loop_programs.vector_program ~extent:3 ~pick:(Loop_programs.pick 3));
      region "scan_trace_2" (Loop_programs.trace_program ~steps:2);
      region "scan_trace_0" (Loop_programs.trace_program ~steps:0);
      region "scan_cached_row_past" (trace_at ~row:3 ~lane:0);
      region "scan_cached_lane_past" (trace_at ~row:0 ~lane:3);
      region "scan_cached_both_past" (trace_at ~row:3 ~lane:3);
      region "scan_cached_negative_row" (trace_at ~row:(-1) ~lane:0);
      kernel "scan_inline"
        (Loop_programs.inline_scan_kernel ~steps:2)
        [ (0, F32 rows_data) ];
      inline "scan_inline_row_past" ~row:3 ~lane:0;
      inline "scan_inline_lane_past" ~row:0 ~lane:3;
      inline "scan_inline_both_past" ~row:3 ~lane:3;
      metered "meter_exact" ~max_state:8192 ~max_updates:6L;
      metered "meter_updates_exhausted" ~max_state:8192 ~max_updates:5L;
      metered "meter_state_over_limit" ~max_state:5 ~max_updates:8192L;
      kernel "group_bool_negative"
        (Loop_programs.bool_group_kernel ~second:(fun read ->
             Expr.Value.sub read (Expr.Value.const 2.)))
        [];
      kernel "group_bool_1e-50"
        (Loop_programs.bool_group_kernel ~second:(fun read ->
             Expr.Value.mul read (Expr.Value.const 1e-50)))
        [];
      unary "exp" Expr.Value.Exp [| 0.5; 1.; 2.5; -1.5 |];
      unary "log" Expr.Value.Log [| 0.5; 1.; 2.5; 10. |];
      unary "sin" Expr.Value.Sin [| 0.5; 1.; 2.5; -1.5 |];
      unary "cos" Expr.Value.Cos [| 0.5; 1.; 2.5; -1.5 |];
      unary "sqrt" Expr.Value.Sqrt [| 0.5; 2.; 3.; 10. |];
      unary "trunc" Expr.Value.Trunc [| 0.5; -1.5; 2.9; -2.9 |];
      unary "erf" Expr.Value.Erf [| 0.5; 1.; -1.5; 3. |];
    ]

(* The tensor for input [id]: its data laid out densely in the shape the
   program's own buffer declares. *)
let bind_of program inputs id =
  match
    ( List.assoc_opt (Tensor_id.to_int id) inputs,
      List.find_opt
        (fun (b : Loop_buffer.t) -> Tensor_id.equal b.Loop_buffer.id id)
        program.Loop_program.buffers )
  with
  | Some (F32 data), Some b ->
      let shape = b.Loop_buffer.sg.Tensor_sig.shape in
      Some (f32_tensor shape (fun c -> data.((Vec6.offset shape c :> int))))
  | Some (F64 data), Some b ->
      let shape = b.Loop_buffer.sg.Tensor_sig.shape in
      Some
        (Tensor.materialize_fmt (Payload.Fmt Payload.F64) shape (fun c ->
             data.((Vec6.offset shape c :> int))))
  | Some (Cells { fmt; quant; cells }), Some b ->
      let shape = b.Loop_buffer.sg.Tensor_sig.shape in
      Some
        (Loop_programs.raw_tensor fmt ?quant ~shape (Array.length cells) cells)
  | _ -> None

let projection which (b : Expr.Eval.Scan_bounds.t) =
  let p = b.Expr.Eval.Scan_bounds.projection in
  Fmt.str "%s cached=%b row=%d lane=%d extent=%d" which
    (Option.is_some p.Expr.Eval.Scan_projection.local)
    p.Expr.Eval.Scan_projection.row p.Expr.Eval.Scan_projection.lane
    b.Expr.Eval.Scan_bounds.extent

let ints l = String.concat "," (List.map string_of_int l)

(* One line per output buffer, cells in dense order, then nothing; or the single
   failure line. The vocabulary is the driver's, so both sides print it alike. *)
let expected c =
  Fmt.str "case %s\n" c.name
  ^
  match
    Err.payload (Loop_interp.run c.program ~bind:(bind_of c.program c.inputs))
  with
  | Error (`Coord_out_of_range (source, axis, index, coord)) ->
      Fmt.str "failure coord_out_of_range buffer=%d axis=%d index=%d coord=%s\n"
        (Expr.Source.to_int source)
        (Expr.Axis.to_int axis) index
        (ints (List.map (Expr.Coord.get coord) Expr.Axis.all))
  | Error (`Index_overflow _) -> "failure index_overflow\n"
  | Error (`Unbound_local _) -> "failure unbound_local\n"
  | Error (`Scan_projection (Expr.Eval.Row_out_of_range b)) ->
      Fmt.str "failure scan_projection %s\n" (projection "row" b)
  | Error (`Scan_projection (Expr.Eval.Lane_out_of_range b)) ->
      Fmt.str "failure scan_projection %s\n" (projection "lane" b)
  | Error (`Scan_meter (Expr.Scan_meter.Updates_exhausted { limit })) ->
      Fmt.str "failure scan_meter updates_exhausted limit=%Ld\n" limit
  | Error (`Scan_meter (Expr.Scan_meter.State_over_limit { limit })) ->
      Fmt.str "failure scan_meter state_over_limit limit=%d\n" limit
  | Error `I64_division_by_zero -> "failure i64_division_by_zero\n"
  | Error `I64_division_overflow -> "failure i64_division_overflow\n"
  | Error `I64_from_float_nan -> "failure i64_from_float_nan\n"
  | Error `I64_from_float_infinite -> "failure i64_from_float_infinite\n"
  | Error (`I64_from_float_out_of_range x) ->
      Fmt.str "failure i64_from_float_out_of_range value=%s\n" (hex x)
  | Error
      (`Gather_index_out_of_range
         { Expr.Eval.Gather_index_out_of_range.raw; extent }) ->
      Fmt.str "failure gather_index_out_of_range raw=%Ld extent=%d\n" raw extent
  | Error _ -> failwith "loop_js_gen: an unexpected failure kind"
  | Ok outputs ->
      String.concat ""
        (List.filter_map
           (fun (b : Loop_buffer.t) ->
             match b.Loop_buffer.role with
             | Loop_buffer.Output ->
                 let (Tensor.Tensor tt as t) =
                   Tensor_id.Map.find b.Loop_buffer.id outputs
                 in
                 let cells = ref [] in
                 Vec6.iter b.Loop_buffer.sg.Tensor_sig.shape (fun coord ->
                     let cell =
                       match tt.Tensor.payload with
                       | { Payload.fmt = Payload.I64; data; _ } ->
                           Int64.to_string
                             data.{(Vec6.offset tt.Tensor.shape coord :> int)}
                       | _ -> hex (Tensor.read t coord)
                     in
                     cells := cell :: !cells);
                 Some
                   (Fmt.str "t%d: %s\n"
                      (Tensor_id.to_int b.Loop_buffer.id)
                      (String.concat " " (List.rev !cells)))
             | Loop_buffer.Input | Loop_buffer.Scratch -> None)
           c.program.Loop_program.buffers)

(* How a failure record is printed: its kind, then the fields the interpreter's
   row carries for that kind, in the same words the OCaml side uses. *)
let report_failure =
  {js|  let line = "failure " + r.kind;
  if (r.kind === "coord_out_of_range") line += " buffer=" + r.buffer + " axis=" + r.axis + " index=" + r.index + " coord=" + r.coord.join(",");
  else if (r.kind === "scan_projection") line += " " + r.which + " cached=" + r.cached + " row=" + r.row + " lane=" + r.lane + " extent=" + r.extent;
  else if (r.kind === "scan_meter") line += " " + r.which + " limit=" + r.limit;
  else if (r.kind === "i64_from_float_out_of_range") line += " value=" + hex(r.value);
  else if (r.kind === "gather_index_out_of_range") line += " raw=" + r.raw + " extent=" + r.extent;
  console.log(line);
|js}

let fmt_of (b : Loop_buffer.t) =
  let (Payload.Fmt f) = b.Loop_buffer.sg.Tensor_sig.fmt in
  Payload.fmt_name f

let driver c =
  let js = Loop_js.emit c.program in
  let args =
    List.map
      (fun (b : Loop_buffer.t) ->
        let n = (Vec6.numel b.Loop_buffer.sg.Tensor_sig.shape :> int) in
        match List.assoc_opt (Tensor_id.to_int b.Loop_buffer.id) c.inputs with
        | Some (F32 data | F64 data) ->
            Fmt.str "new %s(bits([%s]))" (Loop_js.typed_array b)
              (String.concat ", "
                 (Array.to_list
                    (Array.map
                       (fun x ->
                         Printf.sprintf "\"%016Lx\"" (Int64.bits_of_float x))
                       data)))
        | Some (Cells { cells; _ }) ->
            let cell =
              match fmt_of b with
              | "i64" -> fun c -> Int64.to_string c ^ "n"
              | _ -> Int64.to_string
            in
            Fmt.str "new %s([%s])" (Loop_js.typed_array b)
              (String.concat ", " (List.map cell (Array.to_list cells)))
        | None -> Fmt.str "new %s(%d)" (Loop_js.typed_array b) n)
      c.program.Loop_program.buffers
  in
  let outputs =
    List.mapi (fun i b -> (i, b)) c.program.Loop_program.buffers
    |> List.filter_map (fun (i, (b : Loop_buffer.t)) ->
        match b.Loop_buffer.role with
        | Loop_buffer.Output ->
            Some (i, Tensor_id.to_int b.Loop_buffer.id, fmt_of b = "i64")
        | Loop_buffer.Input | Loop_buffer.Scratch -> None)
  in
  String.concat ""
    [
      "(function () {\nconsole.log(\"case " ^ c.name ^ "\");\n";
      js;
      "\nconst args = [" ^ String.concat ", " args ^ "];\n";
      "const r = loop_kernel(...args);\nif (r !== null) {\n";
      report_failure;
      "} else {\n";
      String.concat ""
        (List.map
           (fun (i, id, exact) ->
             Fmt.str
               "  console.log(\"t%d: \" + Array.from(args[%d], %s).join(\" \"));\n"
               id i
               (if exact then "(x) => x.toString()" else "hex"))
           outputs);
      "}\n})();\n";
    ]

let prelude =
  {js|function hex(x) {
  if (Number.isNaN(x)) return "nan";
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, x);
  return view.getBigUint64(0).toString(16).padStart(16, "0");
}
function bits(words) {
  return Float64Array.from(words, (h) => {
    const view = new DataView(new ArrayBuffer(8));
    view.setBigUint64(0, BigInt("0x" + h));
    return view.getFloat64(0);
  });
}
|js}

(* ---- the runtime helpers against their OCaml counterparts ------------------

   [Math.exp] and friends are not correctly rounded by the ECMAScript spec, so
   each carries a recorded tolerance in ulps of binary64: the largest gap
   between the JavaScript result and the OCaml one over the sampled inputs. The
   operations that are exact everywhere ([sqrt], [trunc], [float_max],
   [pool_better]) allow none. [erf] inherits only its inner [exp]'s tolerance.
   The measured maximum is printed to stderr when MEASURE is set, so the recorded
   figure is a measurement; the gate itself prints only within/exceeds, because a
   platform's libm is not a golden. *)

let sample n ~lo ~hi =
  (* A fixed generator: the same inputs on every run and machine. *)
  let state = ref 0x2545F4914F6CDD1DL in
  List.init n (fun _ ->
      (state :=
         Int64.(add (mul !state 6364136223846793005L) 1442695040888963407L));
      let u =
        Int64.to_float (Int64.shift_right_logical !state 11)
        /. 9007199254740992.
      in
      lo +. ((hi -. lo) *. u))

let signed_zeros_and_nan = [ 0.; -0.; nan; infinity; neg_infinity; 1.; -1. ]

type helper = {
  helper : string;
  tolerance : int;
      (** ulps of binary64 for every helper but [erf]; for [erf], multiples of
          [epsilon] (an ulp of 1.0), because it is [1 - poly * exp(..)]: near
          zero the subtraction cancels, so the inner [exp]'s error is a huge
          number of ulps of a tiny result yet still under an epsilon in absolute
          terms, which is the tolerance it actually inherits. *)
  arity : [ `Unary of float -> float | `Binary of float -> float -> float ];
  inputs : (float * float) list;
}

let pairs xs = List.concat_map (fun a -> List.map (fun b -> (a, b)) xs) xs

let helpers =
  let unary name tolerance f xs =
    {
      helper = name;
      tolerance;
      arity = `Unary f;
      inputs = List.map (fun x -> (x, 0.)) (xs @ signed_zeros_and_nan);
    }
  in
  [
    unary "exp" 1 Stdlib.exp (sample 400 ~lo:(-30.) ~hi:30.);
    unary "log" 1 Stdlib.log (sample 400 ~lo:1e-6 ~hi:1e6);
    unary "sin" 1 Stdlib.sin (sample 400 ~lo:(-20.) ~hi:20.);
    unary "cos" 1 Stdlib.cos (sample 400 ~lo:(-20.) ~hi:20.);
    unary "sqrt" 0 Stdlib.sqrt (sample 400 ~lo:0. ~hi:1e6);
    unary "trunc" 0 Float.trunc (sample 400 ~lo:(-1e6) ~hi:1e6);
    unary "erf" 2
      (Expr.Value.apply_unary Expr.Value.Erf)
      (sample 400 ~lo:(-6.) ~hi:6.);
    {
      helper = "float_max";
      tolerance = 0;
      arity = `Binary (Expr.Max_op.apply Expr.Max_op.Float_max);
      inputs = pairs signed_zeros_and_nan;
    };
  ]

let bits x = Printf.sprintf "%016Lx" (Int64.bits_of_float x)

let helper_js h =
  let expected =
    List.map
      (fun (a, b) -> match h.arity with `Unary f -> f a | `Binary f -> f a b)
      h.inputs
  in
  let words l =
    String.concat "," (List.map (fun x -> "\"" ^ bits x ^ "\"") l)
  in
  Fmt.str "checkHelper(%S, %d, %s, [%s], [%s], [%s]);\n" h.helper h.tolerance
    (match h.arity with `Unary _ -> "false" | `Binary _ -> "true")
    (words (List.map fst h.inputs))
    (words (List.map snd h.inputs))
    (words expected)

(* [pool_better] returns a boolean, so it is compared on its truth table. *)
let pool_better_js =
  let cases =
    pairs signed_zeros_and_nan
    |> List.map (fun (best, value) ->
        let truth = Expr.Max_op.pool_better ~best ~value in
        Fmt.str "[\"%s\",\"%s\",%b]" (bits best) (bits value) truth)
  in
  Fmt.str "checkPoolBetter([%s]);\n" (String.concat "," cases)

(* All 65536 patterns of each 16-bit float format, bitwise against the OCaml codec.
   A decode is exactly a binary32 (half widens exactly, bfloat16 IS the high half
   of one), so each result is compared as its binary32 pattern, NaN as the
   canonical quiet NaN: 8 hex digits a pattern instead of 16. *)
let f32_pattern x =
  if Float.is_nan x then 0x7fc00000l else Int32.bits_of_float x

let decode_table decode =
  String.concat ""
    (List.init 65536 (fun h -> Printf.sprintf "%08lx" (f32_pattern (decode h))))

let decode_script =
  let check name decode =
    Fmt.str "checkDecode(%S, %s_to_float, \"%s\");\n" name name
      (decode_table decode)
  in
  check "f16" Half.Half.to_float ^ check "bf16" Half.Bf16.to_float

let decode_prelude =
  {js|function checkDecode(name, decode, expected) {
  const u32 = new Uint32Array(1);
  const f32 = new Float32Array(u32.buffer);
  let wrong = 0;
  for (let h = 0; h < 65536; h++) {
    const x = decode(h);
    let got;
    if (Number.isNaN(x)) got = 0x7fc00000;
    else { f32[0] = x; got = u32[0]; }
    if (got !== parseInt(expected.substr(h * 8, 8), 16)) wrong++;
  }
  console.log("helper " + name + " decode: " + (wrong === 0 ? "all 65536 patterns exact" : wrong + " patterns differ"));
}
|js}

let helper_prelude =
  {js|function ulpsApart(a, b) {
  if (Number.isNaN(a) || Number.isNaN(b)) return Number.isNaN(a) && Number.isNaN(b) ? 0n : 1n << 62n;
  const key = (x) => {
    const view = new DataView(new ArrayBuffer(8));
    view.setFloat64(0, x);
    const bits = view.getBigUint64(0);
    return bits >> 63n ? -(bits & ((1n << 63n) - 1n)) : bits;
  };
  const d = key(a) - key(b);
  return d < 0n ? -d : d;
}
function epsilonsApart(a, b) {
  if (Object.is(a, b) || (Number.isNaN(a) && Number.isNaN(b))) return 0n;
  if (!Number.isFinite(a) || !Number.isFinite(b)) return 1n << 62n;
  return BigInt(Math.ceil(Math.abs(a - b) / Number.EPSILON));
}
function checkHelper(name, tolerance, binary, xs, ys, expected) {
  const fn = { exp: Math.exp, log: Math.log, sin: Math.sin, cos: Math.cos, sqrt: Math.sqrt, trunc: Math.trunc, erf: erf, float_max: float_max }[name];
  const xa = bits(xs), ya = bits(ys), ea = bits(expected);
  let worst = 0n;
  for (let i = 0; i < xa.length; i++) {
    const got = binary ? fn(xa[i], ya[i]) : fn(xa[i]);
    const gap = name === "erf" ? epsilonsApart(got, ea[i]) : ulpsApart(got, ea[i]);
    if (gap > worst) worst = gap;
  }
  const unit = name === "erf" ? "epsilon" : "ulps";
  if (process.env.MEASURE) console.error("helper " + name + ": " + worst + " " + unit);
  console.log("helper " + name + ": " + (worst <= BigInt(tolerance) ? "within " : "EXCEEDS ") + tolerance + " " + unit);
}
function checkPoolBetter(rows) {
  let bad = 0;
  for (const [b, v, truth] of rows) {
    if (pool_better(bits([b])[0], bits([v])[0]) !== truth) bad++;
  }
  console.log("helper pool_better: " + (bad === 0 ? "within 0 ulps" : "EXCEEDS 0 ulps"));
}
|js}

let helper_expected =
  String.concat ""
    (List.map
       (fun h ->
         Fmt.str "helper %s: within %d %s\n" h.helper h.tolerance
           (if h.helper = "erf" then "epsilon" else "ulps"))
       helpers)
  ^ "helper pool_better: within 0 ulps\n"
  ^ "helper f16 decode: all 65536 patterns exact\n"
  ^ "helper bf16 decode: all 65536 patterns exact\n"

let helper_script () =
  (* The helper sources by name, so the script calls the very code the emitter
     prepends. *)
  String.concat "\n" (List.map snd Loop_js_runtime.helpers)
  ^ "\n" ^ helper_prelude ^ decode_prelude
  ^ String.concat "" (List.map helper_js helpers)
  ^ pool_better_js ^ decode_script

let write path s =
  let oc = open_out_bin path in
  output_string oc s;
  close_out oc

let () =
  write "run.js"
    (prelude ^ helper_script () ^ String.concat "" (List.map driver cases));
  write "expected.txt"
    (helper_expected ^ String.concat "" (List.map expected cases))
