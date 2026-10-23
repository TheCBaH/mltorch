(* One line per DMA-suite kernel: its text segment's size and MD5. A kernel's
   bytes do not depend on its cases, which it reads at run time. *)

module K = Peak_kernels
module C = Dma_copy
module O = Dma_overlap
module Rv32_image = Machine_rivet_riscv32.Rv32_image
module Insn = Machine_rivet_riscv32.Rv32_kernel_insn
module Item = Machine_rivet_riscv32.Rv32_kernel_item
module Gpr = Machine_rivet_riscv32.Rv32_kernel_reg.Gpr

let path = { C.Path.src = Sram; dst = Dtcm }

let copies =
  List.map
    (fun engine -> { C.path; engine; cases = [ C.Case.whole 4096 ] })
    C.Engine.
      [
        Dma B16;
        Dma B4;
        Rvv_copy 1;
        Rvv_copy 8;
        Rvv_read 8;
        Rvv_write 8;
        Scalar_copy;
      ]

let overlaps =
  List.map
    (fun work ->
      { O.work; length = 8192; cases = [ { dma = true; trips = 1 } ] })
    O.Work.[ Alu; Dtcm_read; Dtcm_write; Sram_read ]

module G = Dma_gemm

let gemms =
  List.map
    (fun (engine, width, depth) -> { G.blocks = 4; depth; engine; width })
    G.Engine.[ (Rvv_i8, 64, 256); (Zvt_f32, 128, 256); (Zvt_i8, 128, 512) ]

let kernels =
  List.map (fun s -> (C.name s, C.items s, false)) copies
  @ List.map (fun s -> (O.name s, O.items s, false)) overlaps
  @ List.map (fun g -> (G.name g, G.items g, G.Engine.matrix g.engine)) gemms

let items f =
  match Err.payload (f ()) with
  | Ok items -> items
  | Error e -> Fmt.failwith "%a" K.pp_error e

let module_ ~matrix f =
  let extensions =
    if matrix then [ Result.fold ~ok:Fun.id ~error:failwith K.zvt ] else []
  in
  match
    K.to_cli (K.typed_module ~name:"dma" ~extensions (items f) K.rv32_profile)
  with
  | Ok m -> m
  | Error e -> failwith e

let line (name, f, matrix) =
  let laid =
    Err.payload
      (Rv32_image.plan ~profile:K.rv32_profile ~entry:K.entry
         [ module_ ~matrix f ])
  in
  match laid with
  | Error e -> Fmt.failwith "plan: %a" Rv32_image.Error.pp e
  | Ok laid -> (
      match Err.payload (Rv32_image.bind ~base:0x10000L laid) with
      | Error e -> Fmt.failwith "bind: %a" Rv32_image.Error.pp e
      | Ok image ->
          Fmt.str "%s: %s" name
            (String.concat "; "
               (List.map
                  (fun (s : Image.segment) ->
                    Fmt.str "%s:%d:%s" s.Image.name
                      (String.length s.Image.bytes)
                      (Digest.to_hex (Digest.string s.Image.bytes)))
                  image.Image.segments)))

let%expect_test "DMA kernels encode to the pinned bytes" =
  List.iter (fun k -> print_endline (line k)) kernels;
  [%expect
    {|
    sram->dtcm dma/16: .text:112:cbe5e47b7a4040db87521155edc7b030
    sram->dtcm dma/4: .text:112:cbe5e47b7a4040db87521155edc7b030
    sram->dtcm rvv-copy/m1: .text:140:abf802e2cac7d2045d85d1ee956397f0
    sram->dtcm rvv-copy/m8: .text:140:1da3d9569b986922988c6420c6278dfb
    sram->dtcm rvv-read/m8: .text:124:9ce7f097df26ef00cd0ded52bd192191
    sram->dtcm rvv-write/m8: .text:132:3d5ff8f16272431acf07eb99cf5070e1
    sram->dtcm scalar-copy: .text:144:2aa8bfd231fd5a6e11d0e1447f4aa474
    overlap alu 8192: .text:176:a68ecaf64e143e655d2cafb64e336c19
    overlap dtcm-read 8192: .text:168:a7a6a95e02dfdcdf28beeea65928d94a
    overlap dtcm-write 8192: .text:168:04f1f8e12b37ae7cbb3ab79cfa6f126c
    overlap sram-read 8192: .text:168:a7a6a95e02dfdcdf28beeea65928d94a
    rvv i8 resident-b M=8 N=64 K=256: .text:528:326c6bc99af94830285908da5695d510
    zvt f32 resident-b M=128 N=128 K=256: .text:1836:abe8be4b43e6409b9227aa1521f0c4d7
    zvt i8 resident-b M=128 N=128 K=512: .text:1836:4febd282fb5c4371d7bc7aa1800e11a9 |}]

let callee_saved = Gpr.[ s0; s1; s2; s3; s4; s5; s6; s7; s8; s9; s10; s11 ]

(* Whatever callee-saved register a kernel writes, its prologue stores and its
   epilogue reloads, against [sp], which it moves back. *)
let%expect_test "a callee-saved register a kernel writes is stored and restored"
    =
  List.iter
    (fun (name, f, _) ->
      let insns =
        List.filter_map (function Item.Insn i -> Some i | _ -> None) (items f)
      in
      let writes r =
        List.exists
          (fun i -> List.exists (Insn.Reg_ref.equal (Gpr r)) (Insn.writes i))
          insns
      in
      let at_sp f =
        List.filter_map
          (fun i ->
            match (i : Insn.t) with
            | Sw { src; base; _ } when f && Gpr.equal base Gpr.sp -> Some src
            | Lw { dst; base; _ } when (not f) && Gpr.equal base Gpr.sp ->
                Some dst
            | _ -> None)
          insns
      in
      let written = List.filter writes callee_saved in
      let covered rs =
        List.for_all (fun r -> List.exists (Gpr.equal r) rs) written
      in
      let names rs = String.concat " " (List.map (Fmt.str "%a" Gpr.pp) rs) in
      Fmt.pr "%s: %s@." name
        (if written = [] then "caller-saved only"
         else if covered (at_sp true) && covered (at_sp false) then
           "saves " ^ names written
         else "WRITES " ^ names written ^ " UNSAVED"))
    kernels;
  [%expect
    {|
    sram->dtcm dma/16: caller-saved only
    sram->dtcm dma/4: caller-saved only
    sram->dtcm rvv-copy/m1: caller-saved only
    sram->dtcm rvv-copy/m8: caller-saved only
    sram->dtcm rvv-read/m8: caller-saved only
    sram->dtcm rvv-write/m8: caller-saved only
    sram->dtcm scalar-copy: caller-saved only
    overlap alu 8192: caller-saved only
    overlap dtcm-read 8192: caller-saved only
    overlap dtcm-write 8192: caller-saved only
    overlap sram-read 8192: caller-saved only
    rvv i8 resident-b M=8 N=64 K=256: saves s0 s1 s2 s3 s4 s5 s6 s7 s8 s9
    zvt f32 resident-b M=128 N=128 K=256: saves s0 s1 s2 s3 s4 s5 s6 s7 s8 s9
    zvt i8 resident-b M=128 N=128 K=512: saves s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 |}]
