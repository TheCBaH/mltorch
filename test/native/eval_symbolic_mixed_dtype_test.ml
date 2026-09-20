(* [Eval_symbolic]'s own mixed-dtype rejection, the Symbolic twin of
   [pointwise_binary_i64_test.ml]'s Direct-route fixture. Before this fix,
   Symbolic's [process_node] dispatched every op through [E.pixel] uniformly
   with no format check at all, so a mismatched I64/F32 [Add]/[Sub]/[Mul]
   pair silently built a pixel that promotes the I64 operand through
   [Payload.get_float] in DOUBLE precision before a single F32 rounding --
   the exact defect class Direct's own fix closed there. This does
   NOT add I64,I64 dispatch to Symbolic (still Direct-only) -- the
   matching-format cases below
   stay observably unchanged (still route through the ordinary float pixel,
   still lossy above 2^53 for I64,I64), confirming this fix is scoped to the
   mixed pair alone. *)

open Graph_symbolic_fixtures

(* [Eval_symbolic.run] raises rather than returning [Err.t] -- the same
   convention its own [Region_computation] error arms already use -- so this
   mirrors [to_copy_long_i64_test.ml]'s own [catch], not [pointwise_binary_
   i64_test.ml]'s [Err.t]-returning [run]. *)
let catch f =
  try
    ignore (f ());
    "no exception"
  with Err.Exn.E e -> Format.asprintf "raised: %a" Err.Exn.pp_kind e

let build ~x_fmt ~y_fmt op_of =
  Err.or_raise ~pp_error:Graph_builder.pp_error
    Graph_builder.(
      build ~name:"mixed_dtype" ~outputs:(fun r -> [ r ])
      @@
      let* x = input ~shape:(s1c 3) ~name:"x" ~fmt:x_fmt () in
      let* y = input ~shape:(s1c 3) ~name:"y" ~fmt:y_fmt () in
      op_of x y)

let f32 = Payload.Fmt Payload.F32
let i64 = Payload.Fmt Payload.I64

let%expect_test "Symbolic graph: mixed I64/F32 add/sub/mul are rejected" =
  let run op_of () = Eval_symbolic.run (build ~x_fmt:i64 ~y_fmt:f32 op_of) in
  Fmt.pr "%s@." (catch (run Graph_builder.add));
  Fmt.pr "%s@." (catch (run Graph_builder.sub));
  Fmt.pr "%s@." (catch (run Graph_builder.mul));
  [%expect
    {|
    raised: add: unsupported mixed dtype, a=i64 b=f32
    raised: sub: unsupported mixed dtype, a=i64 b=f32
    raised: mul: unsupported mixed dtype, a=i64 b=f32
    |}]

let%expect_test
    "Symbolic graph: matching F32/F32 and I64/I64 add/sub/mul are unaffected" =
  let run x_fmt op_of () =
    Eval_symbolic.run (build ~x_fmt ~y_fmt:x_fmt op_of)
  in
  Fmt.pr "%s@." (catch (run f32 Graph_builder.add));
  Fmt.pr "%s@." (catch (run f32 Graph_builder.sub));
  Fmt.pr "%s@." (catch (run f32 Graph_builder.mul));
  Fmt.pr "%s@." (catch (run i64 Graph_builder.add));
  Fmt.pr "%s@." (catch (run i64 Graph_builder.sub));
  Fmt.pr "%s@." (catch (run i64 Graph_builder.mul));
  [%expect
    {|
    no exception
    no exception
    no exception
    no exception
    no exception
    no exception
    |}]

(* The Stage/Kernel-level twin of the first fixture above: proves the
   rejection is visible to a caller that goes through the full
   [Eval_symbolic.run] -> [Kernel_adapt.of_stage_program] pipeline a real
   Symbolic/Kernel consumer uses, not merely to [Eval_symbolic.run] in
   isolation. [Eval_symbolic.run] already raises before a [Stage_program.t]
   exists, so [Kernel_adapt.of_stage_program] can never actually see a mixed
   pair -- this fixture is the evidence for that, not a second, independent
   check. *)
let%expect_test
    "Stage/Kernel: mixed I64/F32 add/sub/mul are rejected before a kernel is \
     built" =
  let run op_of () =
    Kernel_adapt.of_stage_program
      (Eval_symbolic.run (build ~x_fmt:i64 ~y_fmt:f32 op_of))
  in
  Fmt.pr "%s@." (catch (run Graph_builder.add));
  Fmt.pr "%s@." (catch (run Graph_builder.sub));
  Fmt.pr "%s@." (catch (run Graph_builder.mul));
  [%expect
    {|
    raised: add: unsupported mixed dtype, a=i64 b=f32
    raised: sub: unsupported mixed dtype, a=i64 b=f32
    raised: mul: unsupported mixed dtype, a=i64 b=f32
    |}]
