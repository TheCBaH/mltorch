open Ssa_bridge
open Ssa_ir
module B = Ssa_builder
module Fx = Ssa_ir_test.Ssa_fixtures
module Lf = Loop_ir_test.Loop_fixtures

(* Hand-built programs through generated C against the structured interpreter:
   control flow, every failing operation and the row it reports, rounding, the
   scan meter and scratch locals. The outcome text, every output cell and the
   failure row must be the interpreter's. *)

let idx bld n = B.index bld (Int64.of_int n)

let bufs =
  [
    Fx.buffer 0 ~h:1L ~w:4L Ssa_format.F32 Ssa_buffer.Input;
    Fx.buffer 1 ~h:1L ~w:8L Ssa_format.F32 Ssa_buffer.Output;
    Fx.buffer 2 ~h:1L ~w:4L Ssa_format.Bool Ssa_buffer.Output;
  ]

let load_at bld id i =
  B.load_f64 bld (Fx.buf id) ~decode:Ssa_op.Decode.F32_to_f64
    (Fx.at bld ~h:(idx bld 0) ~w:i)

let store_at bld id i x =
  B.store_f64 bld (Fx.buf id) ~encode:Ssa_op.Encode.F32_round
    (Fx.at bld ~h:(idx bld 0) ~w:i)
    x

let build ?scan_limits f =
  Err.or_raise ~pp_error:Ssa_verify.pp_error
    (B.program ?scan_limits ~buffers:bufs f)

let input = [| 1.5; -2.; 0.25; 8. |]

let structured p =
  let out = Array.make 8 0. and bits = Array.make 4 0. in
  let memory =
    List.fold_left
      (fun m (b : Ssa_buffer.t) ->
        let cells =
          match (b.Ssa_buffer.id :> int) with
          | 0 -> input
          | 1 -> out
          | _ -> bits
        in
        Ssa_id.Buffer.Map.add b.Ssa_buffer.id (Ssa_memory.Floats cells) m)
      Ssa_id.Buffer.Map.empty p.Ssa_program.buffers
  in
  let row =
    match Err.payload (Ssa_interp.run p ~memory) with
    | Ok () -> None
    | Error (#Ssa_interp.failure as f) -> Some (f :> Kernel_eval.error)
    | Error (`Invalid_program _) -> failwith "the program does not verify"
  in
  (row, Array.to_list out @ Array.to_list bits)

let c p =
  let kernel, sites =
    match Ssa_c.kernel ~name:Loop_c_exec.kernel_name p with
    | Ok k -> k
    | Error e -> Fmt.failwith "%a" Ssa_c.pp_error e
  in
  let loop =
    Lf.program
      ~buffers:(List.map Loop_of_ssa.loop_buffer (Ssa_c.arguments p))
      []
  in
  let shape = Lf.shape_w 4 in
  let bind id =
    if Tensor_id.equal id (Lf.tid 0) then
      Some (Lf.f32_tensor shape (fun c -> input.((Vec6.offset shape c :> int))))
    else None
  in
  match Err.payload (Loop_c_exec.exec_kernel ~kernel ~sites loop ~bind) with
  | Error (#Loop_ir.Loop_interp.error as e) ->
      (Some (e :> Kernel_eval.error), [])
  | Error e -> Fmt.failwith "%a" Loop_c_exec.pp_error e
  | Ok outputs ->
      let cells id n =
        match Tensor_id.Map.find_opt (Lf.tid id) outputs with
        | Some t -> Lf.cells t n
        | None -> List.init n (fun _ -> 0.)
      in
      (None, cells 1 8 @ cells 2 4)

(* A NaN's sign is the host's: x86 makes -nan where arm64 makes nan. *)
let pp_cell ppf x =
  if Float.is_nan x then Fmt.string ppf "nan" else Fmt.float ppf x

let attempt ?scan_limits ?(prepare = Fun.id) f =
  let p = prepare (build ?scan_limits f) in
  let s_row, s_cells = structured p in
  let c_row, c_cells = c p in
  let text =
    match c_row with
    | None -> "ok"
    | Some e -> Fmt.str "%a" Kernel_eval.pp_error e
  in
  Fmt.pr "%s | %a | same row and cells as the interpreter: %b@." text
    Fmt.(list ~sep:(any " ") pp_cell)
    (if c_row = None then c_cells else [])
    (s_row = c_row
    && (c_row <> None
       || List.for_all2 Core.Float_bits.equal_portable s_cells c_cells))

let%expect_test "control flow: recurrences, sums, nests, branches, zero trips" =
  attempt (fun bld ->
      let (B.Cons (a, B.Cons (b, B.Nil))) =
        B.for_ bld ~lo:(idx bld 0) ~hi:(idx bld 10)
          ~init:(B.Cons (B.f64 bld 0., B.Cons (B.f64 bld 1., B.Nil)))
          (fun bld _ (B.Cons (a, B.Cons (b, B.Nil))) ->
            B.Cons (b, B.Cons (B.f64_binary bld Expr.Value.Add a b, B.Nil)))
      in
      store_at bld 1 (idx bld 0) a;
      store_at bld 1 (idx bld 1) b);
  attempt (fun bld ->
      let (B.Cons (a, B.Cons (b, B.Nil))) =
        B.for_ bld ~lo:(idx bld 0) ~hi:(idx bld 5)
          ~init:(B.Cons (B.f64 bld 1., B.Cons (B.f64 bld 2., B.Nil)))
          (fun _ _ (B.Cons (a, B.Cons (b, B.Nil))) ->
            B.Cons (b, B.Cons (a, B.Nil)))
      in
      store_at bld 1 (idx bld 0) a;
      store_at bld 1 (idx bld 1) b);
  attempt (fun bld ->
      let sum =
        B.ordered_sum bld ~lo:(idx bld 0) ~hi:(idx bld 4) ~seed:(B.f64 bld 0.5)
          (fun bld k ->
            B.mark bld Ssa_mark.Reduction;
            load_at bld 0 k)
      in
      store_at bld 1 (idx bld 0) sum);
  attempt (fun bld ->
      let (B.Cons (x, B.Nil)) =
        B.for_ bld ~lo:(idx bld 0) ~hi:(idx bld 3)
          ~init:(B.Cons (B.f64 bld 0., B.Nil))
          (fun bld _ (B.Cons (x, B.Nil)) ->
            B.for_ bld ~lo:(idx bld 0) ~hi:(idx bld 4)
              ~init:(B.Cons (x, B.Nil))
              (fun bld k (B.Cons (y, B.Nil)) ->
                B.Cons
                  (B.f64_binary bld Expr.Value.Add y (load_at bld 0 k), B.Nil)))
      in
      store_at bld 1 (idx bld 0) x);
  attempt (fun bld ->
      let (B.Cons (x, B.Nil)) =
        B.for_ bld ~lo:(idx bld 0) ~hi:(idx bld 4)
          ~init:(B.Cons (B.f64 bld 0., B.Nil))
          (fun bld k (B.Cons (acc, B.Nil)) ->
            let c = B.index_compare bld Ssa_op.Compare.Lt (idx bld 1) k in
            B.if_ bld c
              ~then_:(fun bld ->
                B.Cons
                  (B.f64_binary bld Expr.Value.Add acc (load_at bld 0 k), B.Nil))
              ~else_:(fun _ -> B.Cons (acc, B.Nil)))
      in
      store_at bld 1 (idx bld 0) x);
  (* a body that would fail never runs when the range is empty *)
  List.iter
    (fun (lo, hi) ->
      attempt (fun bld ->
          let sum =
            B.ordered_sum bld ~lo:(idx bld lo) ~hi:(idx bld hi)
              ~seed:(B.f64 bld 7.) (fun bld _ -> load_at bld 0 (idx bld 1000))
          in
          let (B.Cons (carried, B.Nil)) =
            B.for_ bld ~lo:(idx bld lo) ~hi:(idx bld hi)
              ~init:(B.Cons (B.f64 bld 3., B.Nil))
              (fun bld _ (B.Cons (x, B.Nil)) ->
                B.Cons
                  ( B.f64_binary bld Expr.Value.Add x
                      (load_at bld 0 (idx bld 1000)),
                    B.Nil ))
          in
          store_at bld 1 (idx bld 0) sum;
          store_at bld 1 (idx bld 1) carried))
    [ (0, 0); (5, 2); (0, 1) ];
  [%expect
    {|
    ok | 55 89 0 0 0 0 0 0 0 0 0 0 | same row and cells as the interpreter: true
    ok | 2 1 0 0 0 0 0 0 0 0 0 0 | same row and cells as the interpreter: true
    ok | 8.25 0 0 0 0 0 0 0 0 0 0 0 | same row and cells as the interpreter: true
    ok | 23.25 0 0 0 0 0 0 0 0 0 0 0 | same row and cells as the interpreter: true
    ok | 8.25 0 0 0 0 0 0 0 0 0 0 0 | same row and cells as the interpreter: true
    ok | 7 3 0 0 0 0 0 0 0 0 0 0 | same row and cells as the interpreter: true
    ok | 7 3 0 0 0 0 0 0 0 0 0 0 | same row and cells as the interpreter: true
    t0[0,0,0,0,1000,0] out of range on axis W: 1000 |  | same row and cells as the interpreter: true |}]

let%expect_test "failing operations report the interpreter's rows" =
  (* both axes outside: the first in axis order is reported *)
  attempt (fun bld ->
      store_at bld 1 (idx bld 0)
        (B.load_f64 bld (Fx.buf 0) ~decode:Ssa_op.Decode.F32_to_f64
           (Fx.at bld ~h:(idx bld 3) ~w:(idx bld 9))));
  attempt (fun bld ->
      store_at bld 1 (idx bld 0)
        (B.load_f64 bld (Fx.buf 0) ~decode:Ssa_op.Decode.F32_to_f64
           (Fx.at bld ~h:(idx bld 0) ~w:(B.index bld (-1L)))));
  attempt (fun bld ->
      B.check_access bld (Fx.buf 0) (Fx.at bld ~h:(idx bld 0) ~w:(idx bld 4));
      store_at bld 1 (idx bld 0) (B.f64 bld 1.));
  (* the first operation to leave the domain fails, though a later one returns *)
  attempt (fun bld ->
      let top = B.index bld 0x7FFF_FFFFL in
      let over = B.index_add bld top (idx bld 1) in
      let back = B.index_add bld over (B.index bld (-1L)) in
      store_at bld 1 back (B.f64 bld 1.));
  attempt (fun bld ->
      store_at bld 1
        (B.index_scale bld 2L (B.index bld 0x4000_0000L))
        (B.f64 bld 1.));
  attempt (fun bld ->
      let low =
        B.index_add bld (B.index bld (-0x8000_0000L)) (B.index bld (-1L))
      in
      store_at bld 1 low (B.f64 bld 1.));
  (* integer division and the float to integer conversion *)
  attempt (fun bld ->
      let q = B.i64_div bld (B.i64 bld 7L) (B.i64 bld 0L) in
      store_at bld 1 (idx bld 0) (B.i64_to_f64 bld q));
  attempt (fun bld ->
      let q = B.i64_div bld (B.i64 bld Int64.min_int) (B.i64 bld (-1L)) in
      store_at bld 1 (idx bld 0) (B.i64_to_f64 bld q));
  attempt (fun bld ->
      let q = B.i64_div bld (B.i64 bld (-7L)) (B.i64 bld 2L) in
      store_at bld 1 (idx bld 0) (B.i64_to_f64 bld q));
  List.iter
    (fun x ->
      attempt (fun bld ->
          let v = B.float_to_i64 bld (B.f64 bld x) in
          store_at bld 1 (idx bld 0) (B.i64_to_f64 bld v)))
    [
      nan;
      infinity;
      neg_infinity;
      9.3e18;
      -9.3e18;
      -2.5;
      1e15;
      -9223372036854775808.;
      9223372036854775808.;
      9223372036854774784.;
    ];
  attempt (fun bld ->
      B.check_gather bld (B.i64 bld 4L) ~extent:4L;
      store_at bld 1 (idx bld 0) (B.f64 bld 1.));
  attempt (fun bld ->
      B.check_gather bld (B.i64 bld (-4L)) ~extent:4L;
      store_at bld 1 (idx bld 0) (B.f64 bld 1.));
  attempt (fun bld ->
      B.check_gather bld (B.i64 bld (-5L)) ~extent:4L;
      store_at bld 1 (idx bld 0) (B.f64 bld 1.));
  [%expect
    {|
    t0[0,0,0,3,9,0] out of range on axis H: 3 |  | same row and cells as the interpreter: true
    t0[0,0,0,0,-1,0] out of range on axis W: -1 |  | same row and cells as the interpreter: true
    t0[0,0,0,0,4,0] out of range on axis W: 4 |  | same row and cells as the interpreter: true
    index overflow: add 2147483647 1 exceeds the 63-bit int domain |  | same row and cells as the interpreter: true
    index overflow: mul 2 1073741824 exceeds the 63-bit int domain |  | same row and cells as the interpreter: true
    index overflow: add -2147483648 -1 exceeds the 63-bit int domain |  | same row and cells as the interpreter: true
    I64 division by zero |  | same row and cells as the interpreter: true
    I64 division overflow: -2^63 / -1 does not fit |  | same row and cells as the interpreter: true
    ok | -3 0 0 0 0 0 0 0 0 0 0 0 | same row and cells as the interpreter: true
    Float-to-I64 cast of NaN |  | same row and cells as the interpreter: true
    Float-to-I64 cast of an infinite value |  | same row and cells as the interpreter: true
    Float-to-I64 cast of an infinite value |  | same row and cells as the interpreter: true
    Float-to-I64 cast of 0x1.02207973f644p+63, outside [-2^63, 2^63) |  | same row and cells as the interpreter: true
    Float-to-I64 cast of -0x1.02207973f644p+63, outside [-2^63, 2^63) |  | same row and cells as the interpreter: true
    ok | -2 0 0 0 0 0 0 0 0 0 0 0 | same row and cells as the interpreter: true
    ok | 1e+15 0 0 0 0 0 0 0 0 0 0 0 | same row and cells as the interpreter: true
    ok | -9.22337e+18 0 0 0 0 0 0 0 0 0 0 0 | same row and cells as the interpreter: true
    Float-to-I64 cast of 0x1p+63, outside [-2^63, 2^63) |  | same row and cells as the interpreter: true
    ok | 9.22337e+18 0 0 0 0 0 0 0 0 0 0 0 | same row and cells as the interpreter: true
    gather index 4 out of range [-4, 3] |  | same row and cells as the interpreter: true
    ok | 1 0 0 0 0 0 0 0 0 0 0 0 | same row and cells as the interpreter: true
    gather index -5 out of range [-4, 3] |  | same row and cells as the interpreter: true |}]

let local_var = Expr.Builder.run Expr.Builder.fresh_local

let%expect_test "scratch locals, scan projections and the meter" =
  attempt (fun bld ->
      let h = B.local_alloc bld ~slots:2L in
      B.local_write bld h (idx bld 1) (B.f64 bld 7.);
      store_at bld 1 (idx bld 0) (B.local_read bld h (idx bld 1)));
  attempt (fun bld ->
      B.check_local bld ~var:local_var ~extent:2L (idx bld 1);
      store_at bld 1 (idx bld 0) (B.f64 bld 3.));
  attempt (fun bld ->
      B.check_local bld ~var:local_var ~extent:2L (B.index bld (-1L));
      store_at bld 1 (idx bld 0) (B.f64 bld 3.));
  attempt (fun bld ->
      let h = B.local_alloc ~var:local_var bld ~slots:2L in
      B.local_write bld h (idx bld 0) (B.f64 bld 1.);
      store_at bld 1 (idx bld 0) (B.local_read bld h (idx bld 2)));
  List.iter
    (fun (row, lane) ->
      List.iter
        (fun var ->
          attempt (fun bld ->
              B.check_scan bld ~var
                ~row:(B.index bld (Int64.of_int row))
                ~lane:(B.index bld (Int64.of_int lane))
                ~row_extent:3L ~lane_extent:2L;
              store_at bld 1 (idx bld 0) (B.f64 bld 1.)))
        [ Some local_var; None ])
    [ (2, 1); (3, 0); (0, 2); (3, 2); (-1, 0) ];
  let limits ~max_state ~max_updates =
    Err.or_raise ~pp_error:Expr.Scan_limits.pp_error
      (Expr.Scan_limits.create ~max_state ~max_updates)
  in
  List.iter
    (fun (n, max_updates) ->
      attempt ~scan_limits:(limits ~max_state:100 ~max_updates) (fun bld ->
          for _ = 1 to n do
            B.meter_charge bld
          done;
          store_at bld 1 (idx bld 0) (B.f64 bld 1.)))
    [ (3, 3L); (4, 3L); (1, 0L) ];
  List.iter
    (fun (widths, max_state) ->
      attempt ~scan_limits:(limits ~max_state ~max_updates:10L) (fun bld ->
          List.iter
            (fun w -> B.meter_reserve bld ~width:(Int64.of_int w))
            widths;
          store_at bld 1 (idx bld 0) (B.f64 bld 1.)))
    [ ([ 3 ], 6); ([ 3 ], 5); ([ 2; 2 ], 8); ([ 2; 2 ], 7) ];
  attempt ~scan_limits:(limits ~max_state:6 ~max_updates:10L) (fun bld ->
      B.meter_reserve bld ~width:3L;
      B.meter_release bld ~width:3L;
      B.meter_reserve bld ~width:3L;
      store_at bld 1 (idx bld 0) (B.f64 bld 1.));
  attempt ~scan_limits:(limits ~max_state:6 ~max_updates:10L) (fun bld ->
      B.meter_reserve bld ~width:3L;
      B.meter_reset bld;
      B.meter_reserve bld ~width:3L;
      store_at bld 1 (idx bld 0) (B.f64 bld 1.));
  [%expect
    {|
    ok | 7 0 0 0 0 0 0 0 0 0 0 0 | same row and cells as the interpreter: true
    ok | 3 0 0 0 0 0 0 0 0 0 0 0 | same row and cells as the interpreter: true
    unbound local #0 |  | same row and cells as the interpreter: true
    unbound local #0 |  | same row and cells as the interpreter: true
    ok | 1 0 0 0 0 0 0 0 0 0 0 0 | same row and cells as the interpreter: true
    ok | 1 0 0 0 0 0 0 0 0 0 0 0 | same row and cells as the interpreter: true
    scan row 3 out of range [0,3) |  | same row and cells as the interpreter: true
    scan row 3 out of range [0,3) |  | same row and cells as the interpreter: true
    scan lane 2 out of range [0,2) at row 0 |  | same row and cells as the interpreter: true
    scan lane 2 out of range [0,2) at row 0 |  | same row and cells as the interpreter: true
    scan row 3 out of range [0,3) |  | same row and cells as the interpreter: true
    scan row 3 out of range [0,3) |  | same row and cells as the interpreter: true
    scan row -1 out of range [0,3) |  | same row and cells as the interpreter: true
    scan row -1 out of range [0,3) |  | same row and cells as the interpreter: true
    ok | 1 0 0 0 0 0 0 0 0 0 0 0 | same row and cells as the interpreter: true
    scan updates exhausted at limit 3 |  | same row and cells as the interpreter: true
    scan updates exhausted at limit 0 |  | same row and cells as the interpreter: true
    ok | 1 0 0 0 0 0 0 0 0 0 0 0 | same row and cells as the interpreter: true
    scan state exceeds limit 5 |  | same row and cells as the interpreter: true
    ok | 1 0 0 0 0 0 0 0 0 0 0 0 | same row and cells as the interpreter: true
    scan state exceeds limit 7 |  | same row and cells as the interpreter: true
    ok | 1 0 0 0 0 0 0 0 0 0 0 0 | same row and cells as the interpreter: true
    ok | 1 0 0 0 0 0 0 0 0 0 0 0 | same row and cells as the interpreter: true |}]

let%expect_test "rounding, conversion and order are part of the value" =
  attempt (fun bld ->
      store_at bld 1 (idx bld 0) (B.f64 bld 0.1);
      let sum =
        B.ordered_sum bld ~lo:(idx bld 0) ~hi:(idx bld 3) ~seed:(B.f64 bld 0.)
          (fun bld i ->
            B.f64_binary bld Expr.Value.Mul (load_at bld 0 i) (B.f64 bld 1e8))
      in
      store_at bld 1 (idx bld 1) sum;
      store_at bld 1 (idx bld 2)
        (B.f32_to_f64 bld (B.f64_to_f32 bld (B.f64 bld 16777217.)));
      List.iteri
        (fun k x ->
          B.store_f64 bld (Fx.buf 2) ~encode:Ssa_op.Encode.Bool_nonzero
            (Fx.at bld ~h:(idx bld 0) ~w:(idx bld k))
            (B.f64 bld x))
        [ nan; -0.; 2.5; 0. ]);
  List.iteri
    (fun k x ->
      attempt (fun bld ->
          store_at bld 1 (idx bld k)
            (B.f32_to_f64 bld (B.i64_to_f32 bld (B.i64 bld x)))))
    [
      16777217L;
      0x0020000020000001L;
      0x7FFFFF7FFFFFFFFFL;
      Int64.max_int;
      Int64.min_int;
      -16777217L;
    ];
  (* unary functions, the maximum with its NaN and zero rules, the pool rule *)
  List.iter
    (fun op ->
      attempt (fun bld ->
          store_at bld 1 (idx bld 0)
            (B.f64_unary bld op (load_at bld 0 (idx bld 0)));
          store_at bld 1 (idx bld 1)
            (B.f64_unary bld op (load_at bld 0 (idx bld 1)))))
    [
      Expr.Value.Cos;
      Expr.Value.Erf;
      Expr.Value.Exp;
      Expr.Value.Log;
      Expr.Value.Sin;
      Expr.Value.Sqrt;
      Expr.Value.Trunc;
    ];
  attempt (fun bld ->
      List.iteri
        (fun k (x, y) ->
          store_at bld 1 (idx bld k) (B.f64_max bld (B.f64 bld x) (B.f64 bld y)))
        [ (nan, 1.); (0., -0.); (-0., 0.); (2., 3.); (3., 2.) ]);
  [%expect
    {|
    ok | 0.1 -2.5e+07 1.67772e+07 0 0 0 0 0 1 0 1 0 | same row and cells as the interpreter: true
    ok | 1.67772e+07 0 0 0 0 0 0 0 0 0 0 0 | same row and cells as the interpreter: true
    ok | 0 9.0072e+15 0 0 0 0 0 0 0 0 0 0 | same row and cells as the interpreter: true
    ok | 0 0 9.22337e+18 0 0 0 0 0 0 0 0 0 | same row and cells as the interpreter: true
    ok | 0 0 0 9.22337e+18 0 0 0 0 0 0 0 0 | same row and cells as the interpreter: true
    ok | 0 0 0 0 -9.22337e+18 0 0 0 0 0 0 0 | same row and cells as the interpreter: true
    ok | 0 0 0 0 0 -1.67772e+07 0 0 0 0 0 0 | same row and cells as the interpreter: true
    ok | 0.0707372 -0.416147 0 0 0 0 0 0 0 0 0 0 | same row and cells as the interpreter: true
    ok | 0.966105 -0.995322 0 0 0 0 0 0 0 0 0 0 | same row and cells as the interpreter: true
    ok | 4.48169 0.135335 0 0 0 0 0 0 0 0 0 0 | same row and cells as the interpreter: true
    ok | 0.405465 nan 0 0 0 0 0 0 0 0 0 0 | same row and cells as the interpreter: true
    ok | 0.997495 -0.909297 0 0 0 0 0 0 0 0 0 0 | same row and cells as the interpreter: true
    ok | 1.22474 nan 0 0 0 0 0 0 0 0 0 0 | same row and cells as the interpreter: true
    ok | 1 -2 0 0 0 0 0 0 0 0 0 0 | same row and cells as the interpreter: true
    ok | nan 0 0 3 3 0 0 0 0 0 0 0 | same row and cells as the interpreter: true |}]

(* a fused multiply-add rounds once: the product's low bits survive into the sum *)
let%expect_test "a contracted multiply-add is one rounding in C too" =
  let contract p = fst (Ssa_opt_contract.pass ~scalar:true p) in
  attempt ~prepare:contract (fun bld ->
      let x = B.f64 bld 0.1 and y = B.f64 bld 0.1 and z = B.f64 bld (-0.01) in
      store_at bld 1 (idx bld 0)
        (B.f64_binary bld Expr.Value.Add z
           (B.f64_binary bld Expr.Value.Mul x y)));
  [%expect
    {| ok | 9.02056e-19 0 0 0 0 0 0 0 0 0 0 0 | same row and cells as the interpreter: true |}]

(* what the narrow cases of each representation would get wrong *)
let%expect_test "integer wrap, negative zero, binary32 rounding and strides" =
  (* int64 arithmetic wraps *)
  List.iter
    (fun (op, a, b) ->
      attempt (fun bld ->
          store_at bld 1 (idx bld 0)
            (B.i64_to_f64 bld (B.i64_arith bld op (B.i64 bld a) (B.i64 bld b)))))
    [
      (Ssa_op.I64_op.Add, Int64.max_int, 1L);
      (Ssa_op.I64_op.Sub, Int64.min_int, 1L);
      (Ssa_op.I64_op.Mul, 0x4000_0000_0000_0000L, 4L);
      (Ssa_op.I64_op.Mul, Int64.max_int, 3L);
    ];
  (* an index that is -0 in a Number becomes +0 as a float *)
  attempt (fun bld ->
      let z = B.index_ceil_div bld 3L (B.index bld (-1L)) in
      store_at bld 1 (idx bld 0) (B.index_to_f64 bld z);
      let f = B.index_floor_div bld 3L (B.index bld 0L) in
      store_at bld 1 (idx bld 1) (B.index_to_f64 bld f));
  (* a binary32 value differs from the binary64 it came from *)
  attempt (fun bld ->
      let wide = B.f64 bld 16777217. in
      let narrow = B.f32_to_f64 bld (B.f64_to_f32 bld wide) in
      let same = B.float_compare bld Ssa_op.Compare.Eq narrow wide in
      store_at bld 1 (idx bld 0)
        (B.select bld same (B.f64 bld 1.) (B.f64 bld 2.)));
  (* a loop whose range is not a multiple of its stride: 0, 2, 4, 6 *)
  let strided p =
    let rule t (s : Ssa_region.t Ssa_stmt.t) =
      match s with
      | Ssa_stmt.For f when Int64.equal f.step 1L ->
          Ssa_rewrite.mark_changed t;
          [ Ssa_stmt.For { f with step = 2L } ]
      | _ -> [ s ]
    in
    fst (Ssa_rewrite.program rule p)
  in
  attempt ~prepare:strided (fun bld ->
      let (B.Cons (x, B.Nil)) =
        B.for_ bld ~lo:(idx bld 0) ~hi:(idx bld 7)
          ~init:(B.Cons (B.f64 bld 0., B.Nil))
          (fun bld i (B.Cons (acc, B.Nil)) ->
            B.Cons
              (B.f64_binary bld Expr.Value.Add acc (B.index_to_f64 bld i), B.Nil))
      in
      store_at bld 1 (idx bld 0) x);
  [%expect
    {|
    ok | -9.22337e+18 0 0 0 0 0 0 0 0 0 0 0 | same row and cells as the interpreter: true
    ok | 9.22337e+18 0 0 0 0 0 0 0 0 0 0 0 | same row and cells as the interpreter: true
    ok | 0 0 0 0 0 0 0 0 0 0 0 0 | same row and cells as the interpreter: true
    ok | 9.22337e+18 0 0 0 0 0 0 0 0 0 0 0 | same row and cells as the interpreter: true
    ok | 0 0 0 0 0 0 0 0 0 0 0 0 | same row and cells as the interpreter: true
    ok | 2 0 0 0 0 0 0 0 0 0 0 0 | same row and cells as the interpreter: true
    ok | 12 0 0 0 0 0 0 0 0 0 0 0 | same row and cells as the interpreter: true |}]

(* A bundle decodes a record against its own table of failure sites, so an
   emitted site is the index of an entry of that table naming the same failure
   and variable, not a number of the emitter's own. *)
let%expect_test "a failure site is the index of the caller's matching entry" =
  let local_var, other =
    Expr.Builder.run
      Expr.Builder.Syntax.(
        let* a = Expr.Builder.fresh_local in
        let* b = Expr.Builder.fresh_local in
        Expr.Builder.return (a, b))
  in
  let entry local =
    Loop_ir.Loop_failure.Local_out_of_range
      { local; index = Loop_ir.Loop_index.Const 0; extent = 2 }
  in
  (* the wanted variable's entry is third *)
  let table = [| entry other; entry other; entry local_var |] in
  let p =
    build (fun bld ->
        B.check_local bld ~var:local_var ~extent:2L (B.index bld 5L);
        store_at bld 1 (idx bld 0) (B.f64 bld 1.))
  in
  (match Ssa_c.kernel ~sites:table ~name:Loop_c_exec.kernel_name p with
  | Error e -> Fmt.pr "refused: %a@." Ssa_c.pp_error e
  | Ok (kernel, sites) -> (
      Fmt.pr "table returned as given: %b@." (sites == table);
      let loop =
        Lf.program
          ~buffers:(List.map Loop_of_ssa.loop_buffer (Ssa_c.arguments p))
          []
      in
      match
        Err.payload
          (Loop_c_exec.exec_kernel ~kernel ~sites loop ~bind:(fun _ -> None))
      with
      | Error (`Unbound_local v) ->
          Fmt.pr "decoded to the wanted variable: %b@."
            (Expr.Local_var.equal v local_var)
      | _ -> Fmt.pr "unexpected outcome@."));
  (* a table that does not name the failure: the record decodes as a defect *)
  (match
     Ssa_c.kernel ~sites:[| entry other |] ~name:Loop_c_exec.kernel_name p
   with
  | Error e -> Fmt.pr "refused: %a@." Ssa_c.pp_error e
  | Ok (kernel, sites) -> (
      let loop =
        Lf.program
          ~buffers:(List.map Loop_of_ssa.loop_buffer (Ssa_c.arguments p))
          []
      in
      match
        Err.payload
          (Loop_c_exec.exec_kernel ~kernel ~sites loop ~bind:(fun _ -> None))
      with
      | Error (`C_host m) -> Fmt.pr "reported as a defect: %s@." m
      | _ -> Fmt.pr "unexpected outcome@."));
  [%expect
    {|
    table returned as given: true
    decoded to the wanted variable: true
    reported as a defect: failure site out of range |}]
