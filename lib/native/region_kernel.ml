let of_graph ?(limits = Kernel.Limits.default) graph =
  Kernel_adapt.of_stage_program ~limits (Eval_symbolic.run ~limits graph)
