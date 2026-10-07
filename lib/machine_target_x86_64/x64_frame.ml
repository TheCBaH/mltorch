(* What frame realization uses on x86-64: r11 for a large offset's address
   (LEA), r10 reserved for control state in transit. MXCSR moves only through
   memory (STMXCSR/LDMXCSR), which the late-form contract does not admit, so
   the entry function's FP controls are established by the native entry
   wrapper instead and the realized frame leaves MXCSR alone. *)

open X64_op

type op = X64_op.t

let scratch = X64_reg.q 11
let control_scratch = X64_reg.q 10
let add_imm base k = Lea (base, k)
let fp_control = None
