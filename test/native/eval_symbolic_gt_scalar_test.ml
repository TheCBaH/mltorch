(* [Gt_scalar]'s Symbolic/Kernel route: unlike the Direct-route dispatch arm
   ([Eval_direct]'s own [Gt_scalar] case, exercised by [dispatch_test.ml] --
   see the implementation tracker), Symbolic has no [Compute_i64]-style
   Bool-aware arm and [Eval_symbolic.check_mixed_dtype] does not list
   [Gt_scalar] at all, so [Eval_symbolic.run] builds an ordinary float
   [Stage.t] for it -- the same [SEMANTICS]-generic 0./1. formula
   [pointwise_test.ml]'s own "Direct: gt_scalar" fixture exercises directly.
   [Graph_builder.gt_scalar] still declares the output edge [Bool]
   unconditionally, so this is a real declared/actual format mismatch, not
   merely an unexercised path -- this fixture proves [Kernel.materializable]'s
   existing F32-only gate (already relied on by the [To_copy(Bool)]/
   [Bitwise_not] Direct-route notes in the tracker) actually catches it at
   [Kernel_adapt.of_stage_program], rather than silently landing a kernel
   that would write float 0./1. bit patterns into storage the graph declares
   as canonical 0/1-byte [Bool]. *)

let shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:1 ~w:1 ~c:3

let build =
  Graph_builder.(
    build ~name:"gt_scalar_symbolic" ~outputs:(fun r -> [ r ])
    @@
    let* x = input ~shape ~name:"x" () in
    gt_scalar ~name:"out" 2. x)
  |> Err.or_raise ~pp_error:Graph_builder.pp_error

let%expect_test
    "Symbolic -> Kernel: Gt_scalar's Bool-declared output is rejected at \
     kernel construction, not silently computed as float" =
  let stage_program = Eval_symbolic.run build in
  let pp_ok fmt _ = Format.pp_print_string fmt "ok" in
  Format.printf "%a@."
    (Core.Pretty.err_result ~ok:pp_ok ~error:Kernel_adapt.pp_error)
    (Kernel_adapt.of_stage_program stage_program);
  [%expect {| t1: a stored value must be f32 and unquantized, got bool |}]
