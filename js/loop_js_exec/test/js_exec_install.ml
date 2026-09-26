(* Every [Loop_check.run] in this library, the copied fixtures and the op sweep
   included, also runs the generated JavaScript and compares it with the
   reference. Inline tests run after every module has initialised, so installing
   here reaches all of them; [installed] exists so a test module can name this
   one and keep it linked. *)
let () = Loop_ir.Loop_check.install (Some Loop_js_exec.exec)
let installed = true
