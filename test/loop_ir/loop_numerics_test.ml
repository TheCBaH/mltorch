open Loop_ir
open Loop_fixtures

let%expect_test "policies have distinct identities" =
  List.iter
    (fun p -> Fmt.pr "%s@." (Loop_numerics.identity p))
    Loop_numerics.all;
  [%expect
    {|
    numerics=fp32_ordered vectorized=f32 other=f32 fma=none reassociation=none
    numerics=reference_f64 vectorized=f64 other=f64 fma=none reassociation=none
    numerics=simd_fp32_ordered vectorized=f32 other=f64 fma=none reassociation=none
    numerics=simd_fp32_relaxed vectorized=f32 other=f64 fma=permitted reassociation=permitted |}];
  assert (
    List.length
      (List.sort_uniq compare
         (List.map Loop_numerics.identity Loop_numerics.all))
    = List.length Loop_numerics.all);
  List.iter
    (fun p -> assert (Loop_numerics.of_name (Loop_numerics.name p) = Some p))
    Loop_numerics.all

let%expect_test "only C, Machine IR and Wasm run a fp32 policy" =
  List.iter
    (fun backend ->
      List.iter
        (fun p ->
          match Err.payload (Loop_numerics.check ~backend p) with
          | Ok p ->
              Fmt.pr "%s %s: ok@."
                (Loop_numerics.Backend.name backend)
                (Loop_numerics.name p)
          | Error e -> Fmt.pr "%a@." Loop_numerics.pp_unsupported e)
        Loop_numerics.all)
    Loop_numerics.Backend.[ C; Interpreter; Javascript; Machine; Wasm ];
  [%expect
    {|
    c fp32_ordered: ok
    c reference_f64: ok
    c simd_fp32_ordered: ok
    c simd_fp32_relaxed: ok
    the interpreter backend does not run the fp32_ordered policy
    interpreter reference_f64: ok
    the interpreter backend does not run the simd_fp32_ordered policy
    the interpreter backend does not run the simd_fp32_relaxed policy
    the javascript backend does not run the fp32_ordered policy
    javascript reference_f64: ok
    the javascript backend does not run the simd_fp32_ordered policy
    the javascript backend does not run the simd_fp32_relaxed policy
    machine fp32_ordered: ok
    machine reference_f64: ok
    machine simd_fp32_ordered: ok
    machine simd_fp32_relaxed: ok
    wasm fp32_ordered: ok
    wasm reference_f64: ok
    wasm simd_fp32_ordered: ok
    wasm simd_fp32_relaxed: ok |}]

(* 2^60 + 2^36 + 1 lies just above the midpoint of two adjacent binary32 values.
   Through binary64 the +1 is lost, the value lands on the midpoint, and
   ties-to-even rounds it down: the double-rounding the one-step conversion must
   not make. *)
let%expect_test "int64 to binary32 rounds once" =
  let n = Int64.(add (shift_left 1L 60) (add (shift_left 1L 36) 1L)) in
  let once = Loop_numerics.round32_of_i64 n in
  let twice = Loop_numerics.round32 (Int64.to_float n) in
  Fmt.pr "once  %h@.twice %h@." once twice;
  [%expect {|
    once  0x1.000002p+60
    twice 0x1p+60 |}];
  List.iter
    (fun n ->
      assert (
        Loop_numerics.round32_of_i64 n
        = Loop_numerics.round32 (Int64.to_float n)))
    [ 0L; 1L; -1L; 16777217L; -16777217L; 9007199254740991L; Int64.max_int ];
  Fmt.pr "min_int %h@." (Loop_numerics.round32_of_i64 Int64.min_int);
  [%expect {| min_int -0x1p+63 |}]

let%expect_test "binary32 constants read back exactly" =
  List.iter
    (fun x -> Fmt.pr "%s@." (Loop_numerics.f32_literal x))
    [ 0.; -0.; 1.; 0.1; 16777217.; nan; infinity; neg_infinity; 3e38; 1e39 ];
  [%expect
    {|
    (0x0p+0f)
    (-0x0p+0f)
    (0x1p+0f)
    (0x1.99999ap-4f)
    (0x1p+24f)
    NAN
    INFINITY
    (-INFINITY)
    (0x1.c363ccp+127f)
    INFINITY |}]

(* ---- the oracle ------------------------------------------------------------ *)

let input = buffer 0 (shape_w 4) f32 Loop_buffer.Input
let output = buffer 1 (shape_w 4) f32 Loop_buffer.Output

let run_with precision body =
  let p =
    program ~buffers:[ input; output ]
      [
        Loop_stmt.For
          {
            var = v 0;
            lo = Loop_index.Const 0;
            hi = Loop_index.Const 4;
            body =
              [
                Loop_stmt.Store_flat
                  {
                    buffer = output;
                    offset = Loop_index.Var (v 0);
                    value = Loop_stored.F32 body;
                  };
              ];
          };
      ]
  in
  let bind id =
    if Tensor_id.equal id (tid 0) then
      Some
        (f32_tensor (shape_w 4) (fun c ->
             [| 1.; 16777217.; 0.1; 1e20 |].((Vec6.offset (shape_w 4) c :> int))))
    else None
  in
  match Err.payload (Loop_interp.run ~precision p ~bind) with
  | Ok m -> cells (Tensor_id.Map.find (tid 1) m) 4
  | Error _ -> failwith "run"

let x = Loop_expr.Load_flat (input, Loop_index.Var (v 0))

(* Without the per-operation rounding the oracle is the binary64 interpreter:
   the two precisions must disagree here, or the F32 mode checks nothing. *)
let%expect_test "the fp32 oracle differs from binary64 where rounding matters" =
  let c = Loop_expr.Const 1.0000001 in
  let body =
    Loop_expr.Binary
      ( Expr.Value.Div,
        Loop_expr.Binary
          (Expr.Value.Mul, Loop_expr.Binary (Expr.Value.Mul, x, c), c),
        Loop_expr.Const 3. )
  in
  let f32c = run_with Loop_numerics.Precision.F32 body in
  let f64c = run_with Loop_numerics.Precision.F64 body in
  let hex = Fmt.(list ~sep:sp (fun ppf x -> pf ppf "%h" x)) in
  Fmt.pr "f32 %a@.f64 %a@.same: %b@." hex f32c hex f64c (f32c = f64c);
  [%expect
    {|
    f32 0x1.55555ap-2 0x1.55555ap+22 0x1.111116p-5
    0x1.ce97dp+64
    f64 0x1.55555ap-2 0x1.55555ap+22 0x1.111114p-5
    0x1.ce97dp+64
    same: false |}]

(* ---- Fp32_ordered: every kernel binary32 ----------------------------------- *)

module P = Loop_vector_programs

let program_of_body ~input body =
  let output = buffer 1 (shape_w 4) f32 Loop_buffer.Output in
  program ~buffers:[ input; output ]
    [
      Loop_stmt.For
        {
          var = v 0;
          lo = Loop_index.Const 0;
          hi = Loop_index.Const 4;
          body =
            [
              Loop_stmt.Store_flat
                {
                  buffer = output;
                  offset = Loop_index.Var (v 0);
                  value = Loop_stored.F32 body;
                };
            ];
        };
    ]

let%expect_test "Fp32_ordered resolves every admitted kernel to binary32" =
  let targets =
    [
      ("no target", None);
      ("neon128", Some Loop_target.neon128);
      ("wasm128", Some Loop_target.wasm128);
    ]
  in
  let programs = P.all @ Loop_fp32_programs.extra in
  List.iter
    (fun (label, target) ->
      let f32 numerics =
        List.length
          (List.filter
             (fun (_, p) ->
               (Loop_plan.resolve ?target ~numerics p).Loop_plan.precision
               = Loop_numerics.Precision.F32)
             programs)
      in
      Fmt.pr "%-10s of %d programs: fp32_ordered %d, simd_fp32_ordered %d@."
        label (List.length programs)
        (f32 Loop_numerics.Fp32_ordered)
        (f32 Loop_numerics.Simd_fp32_ordered))
    targets;
  [%expect
    {|
    no target  of 40 programs: fp32_ordered 40, simd_fp32_ordered 0
    neon128    of 40 programs: fp32_ordered 40, simd_fp32_ordered 21
    wasm128    of 40 programs: fp32_ordered 40, simd_fp32_ordered 21 |}]

(* A dequantizing read was the one refusal: its decode was defined in binary64
   only. With binary32 parameters it is explicit binary32 operations, so every
   policy admits it and [Fp32_ordered] plans it binary32. *)
let%expect_test "Fp32_ordered admits a dequantizing read" =
  let input =
    {
      Loop_buffer.id = tid 0;
      sg =
        Tensor_sig.create ~id:(tid 0) ~name:"" ~shape:(shape_w 4)
          ~fmt:(Payload.Fmt Payload.I8)
          ~quant:(Quant.per_tensor ~scale:0.25 ~zero_point:2.)
          ();
      role = Loop_buffer.Input;
    }
  in
  let p =
    program_of_body ~input (Loop_expr.Load_flat (input, Loop_index.Var (v 0)))
  in
  List.iter
    (fun numerics ->
      let plan = Loop_plan.resolve ~numerics p in
      Fmt.pr "%s: %s, %s@."
        (Loop_numerics.name numerics)
        (Loop_numerics.Precision.name plan.Loop_plan.precision)
        (match Loop_plan.admitted plan with
        | Ok _ -> "admitted"
        | Error r -> Fmt.str "error: %a" Loop_numerics.Refusal.pp r))
    Loop_numerics.all;
  [%expect
    {|
    fp32_ordered: f32, admitted
    reference_f64: f64, admitted
    simd_fp32_ordered: f64, admitted
    simd_fp32_relaxed: f64, admitted |}]
