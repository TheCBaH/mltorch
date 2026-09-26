open Js_of_ocaml
open Loop_ir
open Loop_fixtures

let () = assert Js_exec_install.installed

(* ---- the int64 alias ------------------------------------------------------ *)

let big_of_string s =
  Js.Unsafe.fun_call
    (Js.Unsafe.js_expr "BigInt")
    [| Js.Unsafe.inject (Js.string s) |]

let read_big view i =
  Int64.of_string
    (Js.to_string
       (Js.Unsafe.coerce
          (Js.Unsafe.meth_call
             (Js.Unsafe.get view (string_of_int i))
             "toString" [||])))

let write_big view i n =
  Js.Unsafe.set view (string_of_int i) (big_of_string (Int64.to_string n))

let extremes = [| Int64.min_int; -1L; 0L; 9007199254740993L; Int64.max_int |]

(* Written through the Bigarray, read through the [BigInt64Array] alias, and the
   reverse: a jsoo change to the int64 layout turns this red. *)
let%expect_test "the BigInt64Array alias reads and writes an int64 Bigarray" =
  Fmt.pr "little endian: %b@." Loop_js_exec.little_endian;
  let ba = Bigarray.Array1.create Bigarray.int64 Bigarray.c_layout 5 in
  Array.iteri (fun i n -> ba.{i} <- n) extremes;
  let view = Loop_js_exec.int64_view ba in
  Fmt.pr "through the alias: %a@."
    Fmt.(array ~sep:(any " ") int64)
    (Array.init 5 (read_big view));
  let back = Bigarray.Array1.create Bigarray.int64 Bigarray.c_layout 5 in
  let view = Loop_js_exec.int64_view back in
  Array.iteri (fun i n -> write_big view i n) extremes;
  Fmt.pr "through the Bigarray: %a@."
    Fmt.(array ~sep:(any " ") int64)
    (Array.init 5 (fun i -> back.{i}));
  [%expect
    {|
    little endian: true
    through the alias: -9223372036854775808 -1 0 9007199254740993 9223372036854775807
    through the Bigarray: -9223372036854775808 -1 0 9007199254740993 9223372036854775807
    |}]

(* ---- running a program ---------------------------------------------------- *)

let show_result = function
  | Ok outputs ->
      Tensor_id.Map.iter
        (fun id t ->
          Fmt.pr "%a: %a@." Tensor_id.pp id
            Fmt.(list ~sep:(any " ") float)
            (cells t 4))
        outputs
  | Error e -> Fmt.pr "error: %a@." Loop_js_exec.pp_error e

let%expect_test "a program runs over the bound storage and fills its output" =
  show_result
    (Err.payload
       (Loop_js_exec.exec Loop_programs.doubling ~bind:Loop_programs.bind));
  [%expect {| t1: -0 3 nan 6 |}]

(* ---- every failure constructor, decoded with its full payload -------------- *)

let always = Loop_bool.Index_lt (Loop_index.Const 0, Loop_index.Const 1)
let fresh_local () = Expr.Builder.run Expr.Builder.fresh_local

let failing ?(buffers = []) ?(scan_limits = Expr.Scan_limits.default) body =
  { (program ~buffers body) with Loop_program.scan_limits }

let fail_if failure = [ Loop_stmt.Fail_if (always, failure) ]

let interp p ~bind =
  match Err.payload (Loop_interp.run p ~bind) with
  | Ok _ -> Error "the interpreter did not fail"
  | Error e -> Ok (e :> Loop_js_exec.error)

let exec p ~bind =
  match Err.payload (Loop_js_exec.exec p ~bind) with
  | Ok _ -> Error "the executor did not fail"
  | Error e -> Ok e

(* The executor's row against the interpreter's: the same tag and payload, by
   structural equality, which is the harness's own rule. *)
let same name ?(bind = bind_none) p =
  match (interp p ~bind, exec p ~bind) with
  | Ok a, Ok b ->
      Fmt.pr "%s: %s@." name
        (if a = b then Fmt.str "%a" Loop_js_exec.pp_error b
         else
           Fmt.str "DIFFER: interpreter %a, executor %a" Loop_js_exec.pp_error a
             Loop_js_exec.pp_error b)
  | Error m, _ | _, Error m -> Fmt.pr "%s: %s@." name m

let out_of_range_input = buffer 0 (shape_w 4) f32 Loop_buffer.Input

let%expect_test
    "each Loop_failure constructor is decoded to the interpreter's row" =
  let local = fresh_local () in
  let bind_in id =
    if Tensor_id.equal id (tid 0) then
      Some (f32_tensor (shape_w 4) (fun _ -> 1.))
    else None
  in
  same "gather"
    (failing
       (fail_if
          (Loop_failure.Gather_out_of_range
             { raw = Loop_expr.I64_const (-5L); extent = 3 })));
  same "i64 division by zero"
    (failing (fail_if Loop_failure.I64_division_by_zero));
  same "i64 division overflow"
    (failing (fail_if Loop_failure.I64_division_overflow));
  List.iter
    (fun (name, x) ->
      same name
        (failing
           (fail_if (Loop_failure.I64_from_float { value = Loop_expr.Const x }))))
    [
      ("i64 from float nan", Float.nan);
      ("i64 from float infinite", Float.infinity);
      ("i64 from float out of range", 1e30);
    ];
  let add = Loop_index.Add (Loop_index.Const 2147483647, Loop_index.Const 1) in
  let mul = Loop_index.Scale (65536, Loop_index.Const 65536) in
  List.iter
    (fun (name, index) ->
      same name
        (failing
           [
             Loop_stmt.Fail_if
               ( Loop_bool.Index_overflows index,
                 Loop_failure.Index_overflow { index } );
           ]))
    [ ("index overflow (add)", add); ("index overflow (mul)", mul) ];
  same "load out of range" ~bind:bind_in
    (failing ~buffers:[ out_of_range_input ]
       (fail_if
          (Loop_failure.Load_out_of_range
             { buffer = out_of_range_input; coord = at_w (Loop_index.Const 7) })));
  same "unbound local"
    (failing
       (fail_if
          (Loop_failure.Local_out_of_range
             { local; index = Loop_index.Const 3; extent = 2 })));
  same "scan lane"
    (failing
       (fail_if
          (Loop_failure.Scan_lane_out_of_range
             {
               local = Some local;
               row = Loop_index.Const 4;
               lane = Loop_index.Const 5;
               extent = 3;
             })));
  same "scan row"
    (failing
       (fail_if
          (Loop_failure.Scan_row_out_of_range
             {
               local = None;
               row = Loop_index.Const 6;
               lane = Loop_index.Const 1;
               extent = 3;
             })));
  let limits ~max_state ~max_updates =
    Err.or_raise
      ~pp_error:Fmt.(any "limits")
      (Expr.Scan_limits.create ~max_state ~max_updates)
  in
  same "scan meter: updates exhausted"
    (failing
       ~scan_limits:(limits ~max_state:100 ~max_updates:2L)
       [
         Loop_stmt.Reset_meter;
         Loop_stmt.Charge_scan_update;
         Loop_stmt.Charge_scan_update;
         Loop_stmt.Charge_scan_update;
       ]);
  same "scan meter: state over limit"
    (failing
       ~scan_limits:(limits ~max_state:4 ~max_updates:100L)
       [ Loop_stmt.Reset_meter; Loop_stmt.Reserve_scan_state 3 ]);
  [%expect
    {|
    gather: gather index -5 out of range [-3, 2]
    i64 division by zero: I64 division by zero
    i64 division overflow: I64 division overflow: -2^63 / -1 does not fit
    i64 from float nan: Float-to-I64 cast of NaN
    i64 from float infinite: Float-to-I64 cast of an infinite value
    i64 from float out of range: Float-to-I64 cast of 0x1.93e5939a08ceap+99, outside [-2^63, 2^63)
    index overflow (add): index overflow: add 2147483647 1 exceeds the 32-bit int domain
    index overflow (mul): index overflow: mul 65536 65536 exceeds the 32-bit int domain
    load out of range: t0[0,0,0,0,7,0] out of range on axis W: 7
    unbound local: unbound local #0
    scan lane: scan lane 5 out of range [0,3) at row 4
    scan row: scan row 6 out of range [0,3)
    scan meter: updates exhausted: scan updates exhausted at limit 2
    scan meter: state over limit: scan state exceeds limit 4 |}]

(* ---- refusals -------------------------------------------------------------- *)

let%expect_test "a binding the program did not declare is refused" =
  (match
     Err.payload (Loop_js_exec.exec Loop_programs.doubling ~bind:bind_none)
   with
  | Ok _ -> Fmt.pr "ran@."
  | Error e -> Fmt.pr "%a@." Loop_js_exec.pp_error e);
  [%expect {| no binding for input t0 |}]

(* ---- the helpers, in-process ------------------------------------------------ *)

(* One store per case: 1 when the predicate holds, 0 otherwise. *)
let predicate_program pred =
  let out = buffer 1 (shape_w 1) f32 Loop_buffer.Output in
  program ~buffers:[ out ]
    [
      Loop_stmt.If
        ( pred,
          [
            Loop_stmt.Store
              {
                buffer = out;
                coord = at_w (Loop_index.Const 0);
                value = Loop_stored.F32 (Loop_expr.Const 1.);
              };
          ],
          [
            Loop_stmt.Store
              {
                buffer = out;
                coord = at_w (Loop_index.Const 0);
                value = Loop_stored.F32 (Loop_expr.Const 0.);
              };
          ] );
    ]

let holds run p =
  match Err.payload (run p) with
  | Ok m -> (
      match Tensor_id.Map.find_opt (tid 1) m with
      | Some t -> Some (List.hd (cells t 1) = 1.)
      | None -> None)
  | Error _ -> None

let%expect_test "pool_better agrees with the interpreter on ties, zeros and NaN"
    =
  let c x = Loop_expr.Const x in
  let interp_run p = Loop_interp.run p ~bind:bind_none in
  let js_run p = Loop_js_exec.exec p ~bind:bind_none in
  List.iter
    (fun (best, value) ->
      let p = predicate_program (Loop_bool.Pool_better (c best, c value)) in
      Fmt.pr "pool_better(%g, %g): interpreter %a, generated %a@." best value
        Fmt.(option ~none:(any "?") bool)
        (holds interp_run p)
        Fmt.(option ~none:(any "?") bool)
        (holds js_run p))
    [
      (1., 2.);
      (2., 1.);
      (1., 1.);
      (0., -0.);
      (-0., 0.);
      (1., nan);
      (nan, 1.);
      (nan, nan);
    ];
  [%expect
    {|
    pool_better(1, 2): interpreter true, generated true
    pool_better(2, 1): interpreter false, generated false
    pool_better(1, 1): interpreter false, generated false
    pool_better(0, -0): interpreter false, generated false
    pool_better(-0, 0): interpreter false, generated false
    pool_better(1, nan): interpreter true, generated true
    pool_better(nan, 1): interpreter false, generated false
    pool_better(nan, nan): interpreter true, generated true |}]

let%expect_test "source the engine cannot parse is a Js_compile refusal" =
  (match Err.payload (Loop_js_exec.compile_source "return (") with
  | Ok _ -> Fmt.pr "compiled@."
  | Error (`Js_compile m) ->
      (* The message is the engine's; only its class is ours to pin. *)
      Fmt.pr "Js_compile: %s@." (List.hd (String.split_on_char ':' m)));
  (match Err.payload (Loop_js_exec.compile_source "return 1;") with
  | Ok _ -> Fmt.pr "compiled@."
  | Error (`Js_compile m) -> Fmt.pr "Js_compile: %s@." m);
  [%expect {|
    Js_compile: SyntaxError
    compiled |}]

let%expect_test "a tensor of the wrong format is a binding mismatch" =
  let wrong id =
    if Tensor_id.equal id (tid 0) then
      Some (Tensor.materialize_bool (shape_w 4) (fun _ -> true))
    else None
  in
  (match Err.payload (Loop_js_exec.exec Loop_programs.doubling ~bind:wrong) with
  | Ok _ -> Fmt.pr "ran@."
  | Error e -> Fmt.pr "%a@." Loop_js_exec.pp_error e);
  [%expect {| t0: bound tensor has the wrong format |}]

let%expect_test "a host exception escaping the kernel is a Js_exception" =
  let throwing =
    Loop_js_exec.compile_as Loop_programs.doubling
      "return function () { return 1n + 1; };"
  in
  (match Err.payload throwing with
  | Error _ -> Fmt.pr "did not compile@."
  | Ok compiled -> (
      match
        Err.payload (Loop_js_exec.run compiled ~bind:Loop_programs.bind)
      with
      | Ok _ -> Fmt.pr "ran@."
      | Error (`Js_exception m) ->
          Fmt.pr "Js_exception: %s@." (List.hd (String.split_on_char ':' m))
      | Error _ -> Fmt.pr "another error@."));
  [%expect {| Js_exception: TypeError |}]
