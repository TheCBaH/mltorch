val of_graph :
  ?limits:Kernel.Limits.t ->
  Graph_ir.graph ->
  (Kernel.t, Kernel_adapt.error) Err.t
