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
    (fun (engine, prologue, out, width, depth) ->
      { G.blocks = 4; depth; engine; out; prologue; width })
    G.
      [
        (Engine.Rvv_bf16, Prologue.Shared, Out.Dma, 32, 256);
        (Rvv_f32, Shared, Dma, 32, 128);
        (Rvv_i8, Shared, Dma, 64, 256);
        (Zvt_f32, Dma, Dma, 128, 256);
        (Zvt_i8, Shared, Core, 128, 512);
      ]

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
    rvv bf16 resident-b b-shared c-by-dma M=12 N=32 K=256: .text:588:5ac9afdaa4a10555aa91a48eea6361ff
    rvv f32 resident-b b-shared c-by-dma M=12 N=32 K=128: .text:568:6473902c495e3cfc005a8b2242a6f115
    rvv i8 resident-b b-shared c-by-dma M=8 N=64 K=256: .text:596:10fa5423f03383c538390e5968808180
    zvt f32 resident-b b-by-dma c-by-dma M=128 N=128 K=256: .text:1904:e5c6f4942aa85d0635daa41231903ff7
    zvt i8 resident-b b-shared c-by-core M=128 N=128 K=512: .text:1904:dbb507fa9936cdec417882cf1f191556 |}]

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
    rvv bf16 resident-b b-shared c-by-dma M=12 N=32 K=256: saves s0 s1 s2 s3 s4 s5 s6 s7 s8 s9
    rvv f32 resident-b b-shared c-by-dma M=12 N=32 K=128: saves s0 s1 s2 s3 s4 s5 s6 s7 s8 s9
    rvv i8 resident-b b-shared c-by-dma M=8 N=64 K=256: saves s0 s1 s2 s3 s4 s5 s6 s7 s8 s9
    zvt f32 resident-b b-by-dma c-by-dma M=128 N=128 K=256: saves s0 s1 s2 s3 s4 s5 s6 s7 s8 s9
    zvt i8 resident-b b-shared c-by-core M=128 N=128 K=512: saves s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 |}]

(* The model against the cycles the gemm crams measured. *)
module M = Dma_model

let%expect_test "the model against the measured runs" =
  List.iter
    (fun (engine, prologue, out, blocks, width, depth, measured) ->
      let t = { G.blocks; depth; engine; out; prologue; width } in
      let p = M.predict t in
      Fmt.pr "%s: model %.0f, measured %d (%+.0f%%)@." (G.name t) p measured
        (100. *. (p -. float_of_int measured) /. float_of_int measured))
    G.
      [
        (Engine.Rvv_i8, Prologue.Dma, Out.Dma, 128, 64, 256, 289157);
        (Rvv_i8, Shared, Dma, 128, 64, 256, 279456);
        (Rvv_i8, Shared, Core, 128, 64, 256, 289387);
        (Zvt_i8, Dma, Dma, 4, 32, 128, 33771);
        (Zvt_i8, Shared, Core, 4, 32, 128, 18705);
        (Zvt_i8, Dma, Dma, 8, 128, 512, 312123);
        (Zvt_i8, Dma, Core, 8, 128, 512, 225176);
        (Zvt_i8, Shared, Core, 8, 128, 512, 183697);
        (Zvt_i8, Shared, Core, 8, 128, 1024, 331417);
        (Zvt_f32, Dma, Dma, 8, 128, 256, 512109);
        (Zvt_f32, Shared, Core, 8, 128, 256, 331417);
      ];
  [%expect
    {|
    rvv i8 resident-b b-by-dma c-by-dma M=256 N=64 K=256: model 296825, measured 289157 (+3%)
    rvv i8 resident-b b-shared c-by-dma M=256 N=64 K=256: model 287118, measured 279456 (+3%)
    rvv i8 resident-b b-shared c-by-core M=256 N=64 K=256: model 300784, measured 289387 (+4%)
    zvt i8 resident-b b-by-dma c-by-dma M=128 N=32 K=128: model 33704, measured 33771 (-0%)
    zvt i8 resident-b b-shared c-by-core M=128 N=32 K=128: model 18970, measured 18705 (+1%)
    zvt i8 resident-b b-by-dma c-by-dma M=256 N=128 K=512: model 316683, measured 312123 (+1%)
    zvt i8 resident-b b-by-dma c-by-core M=256 N=128 K=512: model 225193, measured 225176 (+0%)
    zvt i8 resident-b b-shared c-by-core M=256 N=128 K=512: model 184190, measured 183697 (+0%)
    zvt i8 resident-b b-shared c-by-core M=256 N=128 K=1024: model 344815, measured 331417 (+4%)
    zvt f32 resident-b b-by-dma c-by-dma M=256 N=128 K=256: model 520284, measured 512109 (+2%)
    zvt f32 resident-b b-shared c-by-core M=256 N=128 K=256: model 344815, measured 331417 (+4%) |}]

(* Runs measured once, outside the crams and not used to set the rates. *)
let%expect_test "the model against runs it was not fitted to" =
  List.iter
    (fun (engine, prologue, out, blocks, width, depth, measured) ->
      let t = { G.blocks; depth; engine; out; prologue; width } in
      let p = M.predict t in
      Fmt.pr "%s: model %.0f, measured %d (%+.0f%%)@." (G.name t) p measured
        (100. *. (p -. float_of_int measured) /. float_of_int measured))
    G.
      [
        (Engine.Zvt_i8, Prologue.Shared, Out.Core, 8, 256, 512, 352458);
        (Zvt_i8, Shared, Dma, 8, 256, 512, 433341);
        (Zvt_f32, Shared, Core, 8, 128, 512, 655368);
        (Rvv_bf16, Shared, Dma, 64, 32, 256, 258608);
        (Rvv_f32, Shared, Dma, 64, 32, 128, 168368);
        (Zvt_i8, Shared, Core, 16, 256, 1024, 1097986);
        (Zvt_i8, Shared, Dma, 16, 512, 1024, 1961336);
        (Zvt_i8, Shared, Core, 16, 512, 1024, 2173969);
      ];
  [%expect
    {|
    zvt i8 resident-b b-shared c-by-core M=256 N=256 K=512: model 353189, measured 352458 (+0%)
    zvt i8 resident-b b-shared c-by-dma M=256 N=256 K=512: model 438280, measured 433341 (+1%)
    zvt f32 resident-b b-shared c-by-core M=256 N=128 K=512: model 681773, measured 655368 (+4%)
    rvv bf16 resident-b b-shared c-by-dma M=192 N=32 K=256: model 262388, measured 258608 (+1%)
    rvv f32 resident-b b-shared c-by-dma M=192 N=32 K=128: model 171603, measured 168368 (+2%)
    zvt i8 resident-b b-shared c-by-core M=512 N=256 K=1024: model 1092949, measured 1097986 (-0%)
    zvt i8 resident-b b-shared c-by-dma M=512 N=512 K=1024: model 2025066, measured 1961336 (+3%)
    zvt i8 resident-b b-shared c-by-core M=512 N=512 K=1024: model 2158595, measured 2173969 (-1%) |}]

(* DTCM for the kernel's slots: the SoC's, less the stack (2 KiB), the
   mailbox and the alignment the layout gives them. *)
let normal = (32 * 1024) - 3072
let highmem = (1024 * 1024) - 3072

let%expect_test "the fastest variants of a problem, by the model" =
  List.iter
    (fun (soc, budget, engines, m, width, depth) ->
      Fmt.pr "%s, %d x %d x %d:@." soc m width depth;
      List.concat_map
        (fun engine -> M.variants ~budget ~engine ~m ~width ~depth)
        engines
      |> List.sort (fun (x, _) (y, _) -> Float.compare x y)
      |> List.iteri (fun k (cycles, t) ->
          if k < 3 then
            Fmt.pr "  %.0f cycles, %.1f ops/cycle: %s@." cycles
              (float_of_int (G.ops t) /. cycles)
              (G.name t)))
    G.Engine.
      [
        ("32 KiB DTCM", normal, [ Rvv_i8; Zvt_i8 ], 256, 64, 256);
        ("32 KiB DTCM", normal, [ Zvt_i8 ], 512, 32, 128);
        ("1 MiB DTCM", highmem, [ Rvv_i8; Zvt_i8 ], 512, 64, 1024);
        ("1 MiB DTCM", highmem, [ Zvt_i8 ], 512, 256, 1024);
        ("1 MiB DTCM", highmem, [ Zvt_i8 ], 512, 512, 1024);
        ("1 MiB DTCM", highmem, [ Zvt_f32 ], 512, 256, 512);
      ];
  [%expect
    {|
    32 KiB DTCM, 256 x 64 x 256:
      287118 cycles, 29.2 ops/cycle: rvv i8 resident-b b-shared c-by-dma M=256 N=64 K=256
      296825 cycles, 28.3 ops/cycle: rvv i8 resident-b b-by-dma c-by-dma M=256 N=64 K=256
      300784 cycles, 27.9 ops/cycle: rvv i8 resident-b b-shared c-by-core M=256 N=64 K=256
    32 KiB DTCM, 512 x 32 x 128:
      65980 cycles, 63.6 ops/cycle: zvt i8 resident-b b-shared c-by-core M=512 N=32 K=128
      68543 cycles, 61.2 ops/cycle: zvt i8 resident-b b-by-dma c-by-core M=512 N=32 K=128
      121159 cycles, 34.6 ops/cycle: zvt i8 resident-b b-shared c-by-dma M=512 N=32 K=128
    1 MiB DTCM, 512 x 64 x 1024:
      546749 cycles, 122.7 ops/cycle: zvt i8 resident-b b-shared c-by-core M=512 N=64 K=1024
      587751 cycles, 114.2 ops/cycle: zvt i8 resident-b b-by-dma c-by-core M=512 N=64 K=1024
      650882 cycles, 103.1 ops/cycle: zvt i8 resident-b b-shared c-by-dma M=512 N=64 K=1024
    1 MiB DTCM, 512 x 256 x 1024:
      1092949 cycles, 245.6 ops/cycle: zvt i8 resident-b b-shared c-by-core M=512 N=256 K=1024
      1226946 cycles, 218.8 ops/cycle: zvt i8 resident-b b-shared c-by-dma M=512 N=256 K=1024
      1256958 cycles, 213.6 ops/cycle: zvt i8 resident-b b-by-dma c-by-core M=512 N=256 K=1024
    1 MiB DTCM, 512 x 512 x 1024:
      2025066 cycles, 265.1 ops/cycle: zvt i8 resident-b b-shared c-by-dma M=512 N=512 K=1024
      2158595 cycles, 248.7 ops/cycle: zvt i8 resident-b b-shared c-by-core M=512 N=512 K=1024
      2350004 cycles, 228.5 ops/cycle: zvt i8 resident-b b-by-dma c-by-dma M=512 N=512 K=1024
    1 MiB DTCM, 512 x 256 x 512:
      1960131 cycles, 68.5 ops/cycle: zvt f32 resident-b b-shared c-by-core M=512 N=256 K=512
      2021986 cycles, 66.4 ops/cycle: zvt f32 resident-b b-shared c-by-dma M=512 N=256 K=512
      2288150 cycles, 58.7 ops/cycle: zvt f32 resident-b b-by-dma c-by-core M=512 N=256 K=512 |}]
