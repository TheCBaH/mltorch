(* The few C library functions a freestanding RV32 process needs, as one typed
   module in the profile's calling convention. There is no libc in the process,
   so a function not here stays undefined and the image is refused when a unit
   references it. The binary64 functions exist only under the binary64
   profile. *)

open Rv32_asm

let names : Rv32_profile.t -> string list = function
  | Rv32imafd_ilp32d ->
      [ "fabs"; "fabsf"; "fma"; "fmaf"; "memcpy"; "memset"; "sqrt"; "sqrtf" ]
  | Rv32imf_ilp32f -> [ "fabsf"; "fmaf"; "memcpy"; "memset"; "sqrtf" ]

let fn name body =
  [ align 4; global name; label name ] @ body @ [ insn "ret" [] ]

let f n = reg (Printf.sprintf "fa%d" n)

(* [memcpy(dst, src, n)] and [memset(dst, byte, n)] return [dst]. *)
let memcpy =
  fn "memcpy"
    [
      insn "mv" [ reg "t0"; reg "a0" ];
      insn "beqz" [ reg "a2"; sym "memcpy_done" ];
      label "memcpy_loop";
      insn "lbu" [ reg "t1"; mem "a1" 0 ];
      insn "sb" [ reg "t1"; mem "t0" 0 ];
      insn "addi" [ reg "a1"; reg "a1"; imm 1 ];
      insn "addi" [ reg "t0"; reg "t0"; imm 1 ];
      insn "addi" [ reg "a2"; reg "a2"; imm (-1) ];
      insn "bnez" [ reg "a2"; sym "memcpy_loop" ];
      label "memcpy_done";
    ]

let memset =
  fn "memset"
    [
      insn "mv" [ reg "t0"; reg "a0" ];
      insn "beqz" [ reg "a2"; sym "memset_done" ];
      label "memset_loop";
      insn "sb" [ reg "a1"; mem "t0" 0 ];
      insn "addi" [ reg "t0"; reg "t0"; imm 1 ];
      insn "addi" [ reg "a2"; reg "a2"; imm (-1) ];
      insn "bnez" [ reg "a2"; sym "memset_loop" ];
      label "memset_done";
    ]

let binary32 () =
  fn "fabsf" [ insn "fabs.s" [ f 0; f 0 ] ]
  @ fn "fmaf" [ insn "fmadd.s" [ f 0; f 0; f 1; f 2 ] ]
  @ fn "sqrtf" [ insn "fsqrt.s" [ f 0; f 0 ] ]

let binary64 () =
  fn "fabs" [ insn "fabs.d" [ f 0; f 0 ] ]
  @ fn "fma" [ insn "fmadd.d" [ f 0; f 0; f 1; f 2 ] ]
  @ fn "sqrt" [ insn "fsqrt.d" [ f 0; f 0 ] ]

let module_ ?(profile = Rv32_profile.default) () =
  module_ "libc" (fun () ->
      [ text ] @ memcpy @ memset @ binary32 ()
      @
      match profile with
      | Rv32_profile.Rv32imafd_ilp32d -> binary64 ()
      | Rv32imf_ilp32f -> [])
