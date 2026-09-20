(* [Eval_symbolic4]'s own mixed-dtype rejection, the Native4D twin of
   `test/native/eval_symbolic_mixed_dtype_test.ml`. Native4D's Symbolic route
   had no format check at all before this fix (Direct already rejects this
   pair), so a mismatched I64/F32 [Add]/[Sub]/[Mul] pair
   silently built a pixel that promotes the I64 operand through
   [Payload.get_float] in DOUBLE precision before a single F32 rounding.
   This does not add I64,I64 dispatch to Native4D's Symbolic route (still
   Direct-only) -- the matching-format cases below stay observably
   unchanged. *)

open Native4d

let shape3 = Shape4.of_ints ~n:1 ~h:1 ~w:1 ~c:3

let catch f =
  try
    ignore (f ());
    "no exception"
  with Err.Exn.E e -> Format.asprintf "raised: %a" Err.Exn.pp_kind e

let build ~x_fmt ~y_fmt op_of =
  Builder.build
    ~outputs:(fun o -> [ o ])
    (let open Builder in
     let* x = input ~shape:shape3 ~fmt:x_fmt () in
     let* y = input ~shape:shape3 ~fmt:y_fmt () in
     op_of x y)
  |> Err.or_raise ~pp_error:Builder.pp_error

let f32 = Payload.Fmt Payload.F32
let i64 = Payload.Fmt Payload.I64

let%expect_test "Symbolic4 graph: mixed I64/F32 add/sub/mul are rejected" =
  let run op_of () = Eval_symbolic4.run (build ~x_fmt:i64 ~y_fmt:f32 op_of) in
  Fmt.pr "%s@." (catch (run Builder.add));
  Fmt.pr "%s@." (catch (run Builder.sub));
  Fmt.pr "%s@." (catch (run Builder.mul));
  [%expect
    {|
    raised: add: unsupported mixed dtype, a=i64 b=f32
    raised: sub: unsupported mixed dtype, a=i64 b=f32
    raised: mul: unsupported mixed dtype, a=i64 b=f32
    |}]

let%expect_test
    "Symbolic4 graph: matching F32/F32 and I64/I64 add/sub/mul are unaffected" =
  let run x_fmt op_of () =
    Eval_symbolic4.run (build ~x_fmt ~y_fmt:x_fmt op_of)
  in
  Fmt.pr "%s@." (catch (run f32 Builder.add));
  Fmt.pr "%s@." (catch (run f32 Builder.sub));
  Fmt.pr "%s@." (catch (run f32 Builder.mul));
  Fmt.pr "%s@." (catch (run i64 Builder.add));
  Fmt.pr "%s@." (catch (run i64 Builder.sub));
  Fmt.pr "%s@." (catch (run i64 Builder.mul));
  [%expect
    {|
    no exception
    no exception
    no exception
    no exception
    no exception
    no exception
    |}]

(* The Stage/Kernel-level twin of the first fixture above, mirroring
   `test/native/eval_symbolic_mixed_dtype_test.ml`'s own addition:
   [Eval_symbolic4.run] produces the same (Native, not Native4D-specific)
   [Stage_program.t] `Kernel_adapt.of_stage_program` consumes (confirmed by
   `compute_test.ml`'s own existing usage), so this proves the rejection
   survives that composition too, not merely [Eval_symbolic4.run] alone. *)
let%expect_test
    "Stage/Kernel: Native4D mixed I64/F32 add/sub/mul are rejected before a \
     kernel is built" =
  let run op_of () =
    Kernel_adapt.of_stage_program
      (Eval_symbolic4.run (build ~x_fmt:i64 ~y_fmt:f32 op_of))
  in
  Fmt.pr "%s@." (catch (run Builder.add));
  Fmt.pr "%s@." (catch (run Builder.sub));
  Fmt.pr "%s@." (catch (run Builder.mul));
  [%expect
    {|
    raised: add: unsupported mixed dtype, a=i64 b=f32
    raised: sub: unsupported mixed dtype, a=i64 b=f32
    raised: mul: unsupported mixed dtype, a=i64 b=f32
    |}]
