open Loop_ir
open Loop_fixtures

let i0 = Loop_index.Var (v 0)
let c n = Loop_index.Const n

let fail_of p ~bind =
  match Err.payload (Loop_interp.run p ~bind) with
  | Ok _ -> Fmt.pr "no failure@."
  | Error (`Index_overflow { Expr.Index_overflow.op; lhs; rhs }) ->
      (* Not [pp_error]: its text names the host's int width, 63 bits natively
         and 32 under js_of_ocaml, and the row is what is being pinned. *)
      Fmt.pr "index_overflow: %s %d %d@."
        (match op with `Add -> "add" | `Mul -> "mul" | `Sub -> "sub")
        lhs rhs
  | Error e ->
      Fmt.pr "%s: %a@."
        (Loop_check.kind (e :> Kernel_eval.error))
        Loop_interp.pp_error e

let loop ?(lo = 0) hi body =
  Loop_stmt.For { var = v 0; lo = c lo; hi = c hi; body }

(* ---- statement forms ------------------------------------------------------ *)

let%expect_test "store, encode and the f32 round on write" =
  let out = buffer 1 (shape_w 3) f32 Loop_buffer.Output in
  let p =
    program ~buffers:[ out ]
      [
        loop 3
          [
            Loop_stmt.Store
              {
                buffer = out;
                coord = at_w i0;
                value =
                  Loop_stored.F32
                    (Loop_expr.Binary
                       ( Expr.Value.Mul,
                         Loop_expr.Value_of_index i0,
                         Loop_expr.Const 0.1 ));
              };
          ];
      ]
  in
  let r = Tensor_id.Map.find (tid 1) (run_ok p ~bind:bind_none) in
  (* 0.1 is not representable in binary32: the stored cell is the rounded one. *)
  Fmt.pr "%a@." Fmt.(list ~sep:(any " ") (fmt "%h")) (cells r 3);
  Fmt.pr "%h@." (Int32.float_of_bits (Int32.bits_of_float 0.2));
  [%expect {|
    0x0p+0 0x1.99999ap-4 0x1.99999ap-3
    0x1.99999ap-3 |}]

let%expect_test "temporaries, index temporaries, arrays, if and select" =
  let out = buffer 1 (shape_w 4) f32 Loop_buffer.Output in
  let a = Loop_array.of_int 3 in
  let x n = Loop_expr.Temp (Loop_carrier.Float, temp n) in
  let p =
    program ~buffers:[ out ]
      [
        Loop_stmt.Alloc (a, Slot.(count_of_extent (extent 4)));
        loop 4
          [
            Loop_stmt.Assign_index (temp 9, Loop_index.Scale (3, i0));
            Loop_stmt.If
              ( Loop_bool.Index_lt (i0, c 2),
                [
                  Loop_stmt.Array_set
                    (a, i0, Loop_expr.Value_of_index (Loop_index.Temp (temp 9)));
                ],
                [ Loop_stmt.Array_set (a, i0, Loop_expr.Const (-1.)) ] );
            Loop_stmt.Assign
              (Loop_carrier.Float, temp 1, Loop_expr.Array_get (a, i0));
            Loop_stmt.Store
              {
                buffer = out;
                coord = at_w i0;
                value =
                  Loop_stored.F32
                    (Loop_expr.Select
                       ( Loop_bool.Value_lt (x 1, Loop_expr.Const 0.),
                         Loop_expr.Const 100.,
                         x 1 ));
              };
          ];
      ]
  in
  let r = Tensor_id.Map.find (tid 1) (run_ok p ~bind:bind_none) in
  Fmt.pr "%a@." Fmt.(list ~sep:(any " ") float) (cells r 4);
  [%expect {| 0 3 100 100 |}]

let%expect_test "a load decodes, a bool store canonicalises" =
  let input = buffer 0 (shape_w 4) f32 Loop_buffer.Input in
  let out =
    buffer 1 (shape_w 4) (Payload.Fmt Payload.Bool) Loop_buffer.Output
  in
  let p =
    program ~buffers:[ input; out ]
      [
        loop 4
          [
            Loop_stmt.Store
              {
                buffer = out;
                coord = at_w i0;
                value = Loop_stored.Bool (Loop_expr.Load (input, at_w i0));
              };
          ];
      ]
  in
  let data = [| 0.; -0.; nan; 1e-40 |] in
  let bind id =
    if Tensor_id.equal id (tid 0) then
      Some
        (f32_tensor (shape_w 4) (fun c -> data.(Dim.to_int (Vec6.get c Axis.W))))
    else None
  in
  let r = Tensor_id.Map.find (tid 1) (run_ok p ~bind) in
  Fmt.pr "%a@." Fmt.(list ~sep:(any " ") float) (cells r 4);
  [%expect {| 0 0 1 1 |}]

let%expect_test "the int64 carrier: modular arithmetic and exact store" =
  let out = buffer 1 (shape_w 2) (Payload.Fmt Payload.I64) Loop_buffer.Output in
  let big = Int64.add (Int64.shift_left 1L 53) 1L in
  let p =
    program ~buffers:[ out ]
      [
        loop 2
          [
            Loop_stmt.Store
              {
                buffer = out;
                coord = at_w i0;
                value =
                  Loop_stored.I64
                    (Loop_expr.I64_binary
                       ( Expr.Value.I64_add,
                         Loop_expr.I64_const big,
                         Loop_expr.I64_of_index i0 ));
              };
          ];
      ]
  in
  let (Tensor.Tensor t) =
    Tensor_id.Map.find (tid 1) (run_ok p ~bind:bind_none)
  in
  (match t.Tensor.payload with
  | { Payload.fmt = Payload.I64; data; _ } ->
      Fmt.pr "%Ld %Ld@." data.{0} data.{1}
  | _ -> Fmt.pr "not an int64 tensor@.");
  [%expect {| 9007199254740993 9007199254740994 |}]

let%expect_test "marks are counted, and a load is counted once per evaluation" =
  let input = buffer 0 (shape_w 2) f32 Loop_buffer.Input in
  let p =
    program ~buffers:[ input ]
      [
        loop 2
          [
            Loop_stmt.Mark Loop_mark.Key;
            Loop_stmt.Mark Loop_mark.Local;
            Loop_stmt.Assign
              (Loop_carrier.Float, temp 1, Loop_expr.Load (input, at_w i0));
          ];
        Loop_stmt.Mark Loop_mark.Emitter;
      ]
  in
  let counters = Loop_interp.counters () in
  let bind id =
    if Tensor_id.equal id (tid 0) then
      Some (f32_tensor (shape_w 2) (fun _ -> 1.))
    else None
  in
  ignore (run_ok ~counters p ~bind);
  Fmt.pr "keys=%d locals=%d emitters=%d loads=%d@." counters.keys
    counters.locals counters.emitters counters.loads;
  [%expect {| keys=2 locals=2 emitters=1 loads=2 |}]

(* ---- every Fail_if -------------------------------------------------------- *)

let always = Loop_bool.Index_eq (c 0, c 0)

let%expect_test "load_out_of_range names the first failing axis" =
  let input = buffer 0 (shape_w 4) f32 Loop_buffer.Input in
  let coord =
    Expr.Coord.make ~n:(c 0) ~t:(c 0) ~d:(c 0) ~h:(c 5) ~w:(c 9) ~c:(c 0)
  in
  let p =
    program ~buffers:[ input ]
      [
        Loop_stmt.Fail_if
          (always, Loop_failure.Load_out_of_range { buffer = input; coord });
      ]
  in
  fail_of p ~bind:(fun _ -> Some (f32_tensor (shape_w 4) (fun _ -> 0.)));
  [%expect {| coord_out_of_range: t0[0,0,0,5,9,0] out of range on axis H: 5 |}]

let%expect_test "gather, division and float-to-int64 failures" =
  let site f = program [ Loop_stmt.Fail_if (always, f) ] in
  fail_of ~bind:bind_none
    (site
       (Loop_failure.Gather_out_of_range
          { raw = Loop_expr.I64_const Int64.min_int; extent = 4 }));
  fail_of ~bind:bind_none (site Loop_failure.I64_division_by_zero);
  fail_of ~bind:bind_none (site Loop_failure.I64_division_overflow);
  List.iter
    (fun x ->
      fail_of ~bind:bind_none
        (site (Loop_failure.I64_from_float { value = Loop_expr.Const x })))
    [ nan; infinity; 1e30 ];
  [%expect
    {|
    gather_index_out_of_range: gather index -9223372036854775808 out of range [-4, 3]
    i64_division_by_zero: I64 division by zero
    i64_division_overflow: I64 division overflow: -2^63 / -1 does not fit
    i64_from_float_nan: Float-to-I64 cast of NaN
    i64_from_float_infinite: Float-to-I64 cast of an infinite value
    i64_from_float_out_of_range: Float-to-I64 cast of 0x1.93e5939a08ceap+99, outside [-2^63, 2^63) |}]

(* 2^30 * 6 leaves the index domain; the reported row is the language's own. *)
let big = Loop_index.Scale (1 lsl 30, Loop_index.Scale (2, c 3))

let%expect_test "an out-of-domain index is reported where the program guards it"
    =
  let p =
    program
      [
        Loop_stmt.Fail_if (always, Loop_failure.Index_overflow { index = big });
      ]
  in
  fail_of ~bind:bind_none p;
  (* The predicate form: false in the domain, true outside it. *)
  let guard i =
    program
      [
        Loop_stmt.Fail_if
          ( Loop_bool.Index_overflows i,
            Loop_failure.Index_overflow { index = i } );
      ]
  in
  fail_of ~bind:bind_none (guard (Loop_index.Scale (1 lsl 30, c 1)));
  fail_of ~bind:bind_none (guard big);
  [%expect
    {|
    index_overflow: mul 1073741824 6
    no failure
    index_overflow: mul 1073741824 6 |}]

let%expect_test "an unguarded out-of-domain index is a defect, not a wrap" =
  let out = buffer 1 (shape_w 4) f32 Loop_buffer.Output in
  let p =
    program ~buffers:[ out ]
      [
        Loop_stmt.Store
          {
            buffer = out;
            coord = at_w big;
            value = Loop_stored.F32 (Loop_expr.Const 0.);
          };
      ]
  in
  (match Loop_interp.run p ~bind:bind_none with
  | _ -> Fmt.pr "returned@."
  | exception Invalid_argument m -> Fmt.pr "Invalid_argument: %s@." m);
  [%expect
    {| Invalid_argument: Loop_interp: unguarded index outside the index domain |}]

let%expect_test "a local out of range reports the unbound local" =
  let local = Expr.Builder.run Expr.Builder.fresh_local in
  let p =
    program
      [
        Loop_stmt.Fail_if
          ( always,
            Loop_failure.Local_out_of_range { local; index = c 7; extent = 4 }
          );
      ]
  in
  fail_of ~bind:bind_none p;
  [%expect {| unbound_local: unbound local #0 |}]

(* ---- the binding boundary and defects ------------------------------------- *)

let%expect_test "inputs are validated before anything runs" =
  let input = buffer 0 (shape_w 4) f32 Loop_buffer.Input in
  let p = program ~buffers:[ input ] [ Loop_stmt.Mark Loop_mark.Key ] in
  fail_of ~bind:bind_none p;
  fail_of p ~bind:(fun _ -> Some (f32_tensor (shape_w 3) (fun _ -> 0.)));
  [%expect
    {|
    unbound_input: no binding for input t0
    binding_mismatch: t0: bound tensor has the wrong shape |}]

let%expect_test "an unchecked access out of range is a defect, not a failure" =
  let input = buffer 0 (shape_w 2) f32 Loop_buffer.Input in
  let p =
    program ~buffers:[ input ]
      [
        Loop_stmt.Assign
          (Loop_carrier.Float, temp 1, Loop_expr.Load (input, at_w (c 5)));
      ]
  in
  (match
     Loop_interp.run p ~bind:(fun _ ->
         Some (f32_tensor (shape_w 2) (fun _ -> 0.)))
   with
  | _ -> Fmt.pr "returned@."
  | exception Invalid_argument m -> Fmt.pr "Invalid_argument: %s@." m);
  [%expect {| Invalid_argument: Loop_interp: unchecked access out of range |}]
