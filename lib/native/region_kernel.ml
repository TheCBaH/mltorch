let of_graph ?limits graph =
  let regionized = Eval_symbolic.run_regionized ?limits graph in
  Kernel_adapt.of_stage_program ?limits
    ~region_candidates:regionized.Eval_symbolic.candidates regionized.program
