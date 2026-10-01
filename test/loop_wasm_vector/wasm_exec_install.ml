(* Every [Loop_check.run] in this library, the copied fixtures and the op sweep
   included, also runs the SIMD-lowered Wasm under node and compares it with the
   reference. [installed] exists so a test module can name this one and keep it
   linked. *)
let () = Loop_ir.Loop_check.install (Some Loop_wasm_exec.executor_simd)
let installed = true
