let of_graph ?limits graph =
  Kernel_adapt.of_stage_program ?limits (Eval_symbolic.run graph)
