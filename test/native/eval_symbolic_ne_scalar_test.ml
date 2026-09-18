(* [Ne_scalar]'s Symbolic/Kernel route (P6.4): unlike the Direct-route
   dispatch arm ([Eval_direct]'s own [Ne_scalar] case, exercised end to end
   by [ne_scalar_bool_test.ml]), Symbolic has no
   [Compute_i64]-style Bool-aware arm and [Eval_symbolic.check_mixed_dtype]
   does not list [Ne_scalar] at all, so [Eval_symbolic.run] builds an
   ordinary float [Stage.t] for it -- the same [SEMANTICS]-generic 0./1.
   formula [pointwise_test.ml]'s own "Direct: ne_scalar" fixture exercises
   directly. [Graph_builder.ne_scalar] still declares the output edge [Bool]
   unconditionally, so this is a real declared/actual format mismatch, not
   merely an unexercised path -- this fixture proves the same
   [Kernel.materializable] F32-only gate [Eq_scalar]'s own analogous fixture
   proves catches it at [Kernel_adapt.of_stage_program]. *)

let shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:3

let build =
  Graph_builder.(
    build ~name:"ne_scalar_symbolic" ~outputs:(fun r -> [ r ])
    @@
    let* x = input ~shape ~name:"x" () in
    ne_scalar ~name:"out" 2. x)
  |> Err.or_raise ~pp_error:Graph_builder.pp_error

let%expect_test
    "Symbolic -> Kernel: Ne_scalar's Bool-declared output is rejected at \
     kernel construction, not silently computed as float" =
  let stage_program = Eval_symbolic.run build in
  let pp_ok fmt _ = Format.pp_print_string fmt "ok" in
  Format.printf "%a@."
    (Core.Pretty.err_result ~ok:pp_ok ~error:Kernel_adapt.pp_error)
    (Kernel_adapt.of_stage_program stage_program);
  [%expect {| t1: a stored value must be f32 and unquantized, got bool |}]
