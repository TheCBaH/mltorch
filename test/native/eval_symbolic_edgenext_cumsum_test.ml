(* Gate 7 item 4 (Symbolic/Kernel half): the same EdgeNeXt Bool/cumsum
   subgraph [edgenext_cumsum_test.ml] proves on Direct, run through
   Symbolic/Kernel instead. [inv] (the [Bitwise_not] output) is shared by
   BOTH cumsum consumers, so [Eval_symbolic] cannot fuse it into either
   consumer's own expression -- it must become a genuine stored
   [Kernel.Value.t], declared [Bool] by [Graph_builder.bitwise_not]. Every
   stored value goes through [Kernel.materializable] (see [kernel.ml]'s own
   [of_stage_program], which checks EVERY [Value.t], not just the graph's
   requested outputs) at [Kernel_adapt.of_stage_program] construction time --
   the same F32-only gate [eval_symbolic_gt_scalar_test.ml] already proved
   for a Bool-declared OUTPUT. This fixture proves the identical gate fires
   for a Bool-declared shared INTERMEDIATE feeding two all-Float32-output
   consumers, confirming the "expected to hit the same
   [Kernel.materializable] ... not independently confirmed" note the Gate 7
   tracker entry left open. *)

let shape = Vec6.shape ~n:1 ~t:1 ~d:1 ~h:2 ~w:4 ~c:1

let build =
  Graph_builder.(
    build ~name:"edgenext_cumsum_symbolic" ~outputs:(fun (h, w) -> [ h; w ])
    @@
    let* x = input ~shape ~name:"x" () in
    let* mask = to_copy ~name:"mask" Pointwise.To_copy.Bool x in
    let* inv = bitwise_not ~name:"inv" mask in
    let* cum_h = cumsum ~name:"cum_h" { Reduce.Cumsum.axis = Axis.H } inv in
    let* cum_w = cumsum ~name:"cum_w" { Reduce.Cumsum.axis = Axis.W } inv in
    return (cum_h, cum_w))
  |> Err.or_raise ~pp_error:Graph_builder.pp_error

let%expect_test
    "Symbolic -> Kernel: EdgeNeXt's shared Bool [inv] intermediate is rejected \
     at kernel construction, even though both consumers and both graph outputs \
     are Float32" =
  let stage_program = Eval_symbolic.run build in
  let pp_ok fmt _ = Format.pp_print_string fmt "ok" in
  Format.printf "%a@."
    (Core.Pretty.err_result ~ok:pp_ok ~error:Kernel_adapt.pp_error)
    (Kernel_adapt.of_stage_program stage_program);
  [%expect {| t1: a stored value must be f32 and unquantized, got bool |}]
