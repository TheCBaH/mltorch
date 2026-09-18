(* Gate 7 item 4 (Native4D Symbolic half): the Native4D twin of
   [eval_symbolic_edgenext_cumsum_test.ml] -- [Eval_symbolic4.run] produces
   the SAME [Stage_program.t] type Native's own [Eval_symbolic.run] does (see
   that module's own top comment: "into the SAME [Stage_program.t] a Native
   graph produces"), so Native's [Kernel_adapt.of_stage_program] applies to a
   Native4D-built program unchanged -- no Native4D-specific Kernel_adapt
   exists, and none is needed. This closes the P6.3 tracker note that
   Native4D's "own Symbolic safety was reasoned from the shared
   [Kernel.materializable] gate, not independently re-proven with its own
   fixture" for this subgraph: [inv] (the [Bitwise_not] output) is shared by
   both [cumsum4] consumers, so it must become a genuine stored
   [Kernel.Value.t], declared [Bool], and is rejected before either
   Float32 cumsum could be computed. *)

open Native4d

let shape4 = Shape4.of_ints ~n:1 ~h:2 ~w:4 ~c:1

let build =
  Builder.build
    ~outputs:(fun (h, w) -> [ h; w ])
    (let open Builder in
     let* x = input ~shape:shape4 () in
     let* mask = to_copy Pointwise.To_copy.Bool x in
     let* inv = bitwise_not mask in
     let* cum_h = cumsum4 { Ops4_cumsum.Cumsum4.axis = Axis4.H } inv in
     let* cum_w = cumsum4 { Ops4_cumsum.Cumsum4.axis = Axis4.W } inv in
     return (cum_h, cum_w))
  |> Err.or_raise ~pp_error:Builder.pp_error

let%expect_test
    "Native4D Symbolic -> Native's Kernel: EdgeNeXt's shared Bool [inv] \
     intermediate is rejected at kernel construction, even though both cumsum4 \
     consumers and both graph outputs are Float32" =
  let stage_program = Eval_symbolic4.run build in
  let pp_ok fmt _ = Format.pp_print_string fmt "ok" in
  Format.printf "%a@."
    (Core.Pretty.err_result ~ok:pp_ok ~error:Kernel_adapt.pp_error)
    (Kernel_adapt.of_stage_program stage_program);
  [%expect {| t1: a stored value must be f32 and unquantized, got bool |}]
