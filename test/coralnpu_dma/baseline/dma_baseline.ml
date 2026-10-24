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
    (fun (engine, geometry, prologue, out, width, depth) ->
      { G.blocks = 4; depth; engine; geometry; out; prologue; width })
    G.
      [
        (Engine.Rvv_bf16, Geometry.Resident_b, Prologue.Shared, Out.Dma, 32, 256);
        (Rvv_f32, Resident_b, Shared, Dma, 32, 128);
        (Rvv_i8, Resident_b, Shared, Dma, 64, 256);
        (Zvt_f32, Resident_b, Dma, Dma, 128, 256);
        (Zvt_i8, Resident_b, Shared, Core, 128, 512);
        (Zvt_i8, Resident_b, Shared, Split 8, 512, 1024);
        (Rvv_bf16, Resident_a { panel = 32 }, Shared, Dma, 1024, 128);
        (Rvv_i8, Resident_a { panel = 64 }, Shared, Core, 512, 128);
        (Zvt_i8, Resident_a { panel = 32 }, Shared, Core, 1024, 1024);
        (Zvt_i8, Resident_a { panel = 64 }, Shared, Dma, 1024, 1024);
        (Rvv_f32, Row_major_a, Dma, Core, 32, 128);
        (Rvv_i8, Row_major_a, Shared, Dma, 64, 256);
        (Zvt_i8, Row_major_a, Shared, Split 8, 256, 512);
        (Rvv_bf16, Split_k { chunk = 64 }, Shared, Dma, 32, 256);
        (Rvv_i8, Split_k { chunk = 128 }, Shared, Dma, 64, 1024);
        (Zvt_i8, Split_k { chunk = 128 }, Shared, Dma, 64, 512);
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
    zvt i8 resident-b b-shared c-by-core M=128 N=128 K=512: .text:1904:dbb507fa9936cdec417882cf1f191556
    zvt i8 resident-b b-shared c-split-8 M=128 N=512 K=1024: .text:8452:fc64708209aaf5e989b2f7bdb2dce18f
    rvv bf16 resident-a/32 a-shared c-by-dma M=12 N=1024 K=128: .text:636:7254291c8f3dd9a06c254379204cd9aa
    rvv i8 resident-a/64 a-shared c-by-core M=8 N=512 K=128: .text:652:242ebbe6e3f45ff4ecb57aeb4a9d2576
    zvt i8 resident-a/32 a-shared c-by-core M=128 N=1024 K=1024: .text:868:2bc6427349784694f3a1e8b44af79f86
    zvt i8 resident-a/64 a-shared c-by-dma M=128 N=1024 K=1024: .text:1224:1cf562adcadd8ea74b6f902cba497d1c
    rvv f32 row-major-a b-by-dma c-by-core M=12 N=32 K=128: .text:648:82533843a4bd1113aec33219b8ad8fbb
    rvv i8 row-major-a b-shared c-by-dma M=8 N=64 K=256: .text:676:c4b5d93bc229d5824edb7f3013994273
    zvt i8 row-major-a b-shared c-split-8 M=128 N=256 K=512: .text:4148:0c7b03a9437e93c9690de5fb26b9a845
    rvv bf16 split-k/64 b-shared c-by-dma M=12 N=32 K=256: .text:660:48e8ff9383138b1519e52c382b4f1a30
    rvv i8 split-k/128 b-shared c-by-dma M=8 N=64 K=1024: .text:664:bdf06f3d249ff812747581ef89c105e3
    zvt i8 split-k/128 b-shared c-by-dma M=128 N=64 K=512: .text:1544:1ed1c92757d2e507b9c92b3bad9f52f3 |}]

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
    zvt i8 resident-b b-shared c-by-core M=128 N=128 K=512: saves s0 s1 s2 s3 s4 s5 s6 s7 s8 s9
    zvt i8 resident-b b-shared c-split-8 M=128 N=512 K=1024: saves s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10
    rvv bf16 resident-a/32 a-shared c-by-dma M=12 N=1024 K=128: saves s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11
    rvv i8 resident-a/64 a-shared c-by-core M=8 N=512 K=128: saves s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11
    zvt i8 resident-a/32 a-shared c-by-core M=128 N=1024 K=1024: saves s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11
    zvt i8 resident-a/64 a-shared c-by-dma M=128 N=1024 K=1024: saves s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11
    rvv f32 row-major-a b-by-dma c-by-core M=12 N=32 K=128: saves s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s11
    rvv i8 row-major-a b-shared c-by-dma M=8 N=64 K=256: saves s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s11
    zvt i8 row-major-a b-shared c-split-8 M=128 N=256 K=512: saves s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11
    rvv bf16 split-k/64 b-shared c-by-dma M=12 N=32 K=256: saves s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11
    rvv i8 split-k/128 b-shared c-by-dma M=8 N=64 K=1024: saves s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11
    zvt i8 split-k/128 b-shared c-by-dma M=128 N=64 K=512: saves s0 s1 s2 s3 s4 s5 s6 s7 s8 s9 s10 s11 |}]

(* The model against the cycles the gemm crams measured. *)
module M = Dma_model

let%expect_test "the model against the measured runs" =
  List.iter
    (fun (engine, geometry, prologue, out, blocks, width, depth, measured) ->
      let t = { G.blocks; depth; engine; geometry; out; prologue; width } in
      let p = M.predict t in
      Fmt.pr "%s: model %.0f, measured %d (%+.0f%%)@." (G.name t) p measured
        (100. *. (p -. float_of_int measured) /. float_of_int measured))
    G.
      [
        ( Engine.Rvv_i8,
          Geometry.Resident_b,
          Prologue.Dma,
          Out.Dma,
          128,
          64,
          256,
          289157 );
        (Rvv_i8, Resident_b, Shared, Dma, 128, 64, 256, 279456);
        (Rvv_i8, Resident_b, Shared, Core, 128, 64, 256, 289387);
        (Zvt_i8, Resident_b, Dma, Dma, 4, 32, 128, 33771);
        (Zvt_i8, Resident_b, Shared, Core, 4, 32, 128, 18705);
        (Zvt_i8, Resident_b, Dma, Dma, 8, 128, 512, 312123);
        (Zvt_i8, Resident_b, Dma, Core, 8, 128, 512, 225176);
        (Zvt_i8, Resident_b, Shared, Core, 8, 128, 512, 183697);
        (Zvt_i8, Resident_b, Shared, Core, 8, 128, 1024, 331417);
        (Zvt_f32, Resident_b, Dma, Dma, 8, 128, 256, 512109);
        (Zvt_f32, Resident_b, Shared, Core, 8, 128, 256, 331417);
        (Zvt_i8, Resident_b, Shared, Split 20, 16, 256, 1024, 1047424);
        (Rvv_i8, Resident_a { panel = 64 }, Shared, Dma, 16, 512, 128, 148370);
        (Rvv_i8, Resident_a { panel = 64 }, Shared, Core, 16, 512, 128, 154678);
        (Rvv_bf16, Resident_a { panel = 32 }, Shared, Dma, 8, 1024, 128, 518720);
        (Zvt_i8, Resident_a { panel = 32 }, Shared, Core, 1, 1024, 128, 125666);
        (Zvt_i8, Resident_a { panel = 32 }, Shared, Core, 4, 1024, 1024, 1152274);
        (Rvv_i8, Split_k { chunk = 128 }, Shared, Dma, 64, 64, 1024, 584634);
        (Zvt_i8, Split_k { chunk = 128 }, Shared, Dma, 8, 32, 1024, 741512);
        (Rvv_i8, Row_major_a, Shared, Dma, 128, 64, 256, 294312);
        (Zvt_i8, Row_major_a, Shared, Core, 8, 128, 512, 328054);
      ];
  [%expect
    {|
    rvv i8 resident-b b-by-dma c-by-dma M=256 N=64 K=256: model 289529, measured 289157 (+0%)
    rvv i8 resident-b b-shared c-by-dma M=256 N=64 K=256: model 279765, measured 279456 (+0%)
    rvv i8 resident-b b-shared c-by-core M=256 N=64 K=256: model 293488, measured 289387 (+1%)
    zvt i8 resident-b b-by-dma c-by-dma M=128 N=32 K=128: model 33704, measured 33771 (-0%)
    zvt i8 resident-b b-shared c-by-core M=128 N=32 K=128: model 18970, measured 18705 (+1%)
    zvt i8 resident-b b-by-dma c-by-dma M=256 N=128 K=512: model 316683, measured 312123 (+1%)
    zvt i8 resident-b b-by-dma c-by-core M=256 N=128 K=512: model 224794, measured 225176 (-0%)
    zvt i8 resident-b b-shared c-by-core M=256 N=128 K=512: model 183791, measured 183697 (+0%)
    zvt i8 resident-b b-shared c-by-core M=256 N=128 K=1024: model 344815, measured 331417 (+4%)
    zvt f32 resident-b b-by-dma c-by-dma M=256 N=128 K=256: model 520227, measured 512109 (+2%)
    zvt f32 resident-b b-shared c-by-core M=256 N=128 K=256: model 344815, measured 331417 (+4%)
    zvt i8 resident-b b-shared c-split-20 M=512 N=256 K=1024: model 1018883, measured 1047424 (-3%)
    rvv i8 resident-a/64 a-shared c-by-dma M=32 N=512 K=128: model 146539, measured 148370 (-1%)
    rvv i8 resident-a/64 a-shared c-by-core M=32 N=512 K=128: model 156653, measured 154678 (+1%)
    rvv bf16 resident-a/32 a-shared c-by-dma M=24 N=1024 K=128: model 512608, measured 518720 (-1%)
    zvt i8 resident-a/32 a-shared c-by-core M=32 N=1024 K=128: model 128661, measured 125666 (+2%)
    zvt i8 resident-a/32 a-shared c-by-core M=128 N=1024 K=1024: model 1209469, measured 1152274 (+5%)
    rvv i8 split-k/128 b-shared c-by-dma M=128 N=64 K=1024: model 620712, measured 584634 (+6%)
    zvt i8 split-k/128 b-shared c-by-dma M=256 N=32 K=1024: model 736720, measured 741512 (-1%)
    rvv i8 row-major-a b-shared c-by-dma M=256 N=64 K=256: model 294660, measured 294312 (+0%)
    zvt i8 row-major-a b-shared c-by-core M=256 N=128 K=512: model 329427, measured 328054 (+0%) |}]

(* Runs measured once, outside the crams and not used to set the rates. *)
let%expect_test "the model against runs it was not fitted to" =
  List.iter
    (fun (engine, geometry, prologue, out, blocks, width, depth, measured) ->
      let t = { G.blocks; depth; engine; geometry; out; prologue; width } in
      let p = M.predict t in
      Fmt.pr "%s: model %.0f, measured %d (%+.0f%%)@." (G.name t) p measured
        (100. *. (p -. float_of_int measured) /. float_of_int measured))
    G.
      [
        ( Engine.Zvt_i8,
          Geometry.Resident_b,
          Prologue.Shared,
          Out.Core,
          8,
          256,
          512,
          352458 );
        (Zvt_i8, Resident_b, Shared, Dma, 8, 256, 512, 433341);
        (Zvt_f32, Resident_b, Shared, Core, 8, 128, 512, 655368);
        (Rvv_bf16, Resident_b, Shared, Dma, 64, 32, 256, 258608);
        (Rvv_f32, Resident_b, Shared, Dma, 64, 32, 128, 168368);
        (Zvt_i8, Resident_b, Shared, Core, 16, 256, 1024, 1097986);
        (Zvt_i8, Resident_b, Shared, Dma, 16, 512, 1024, 1961336);
        (Zvt_i8, Resident_b, Shared, Core, 16, 512, 1024, 2173969);
        (Zvt_i8, Resident_b, Shared, Split 8, 16, 512, 1024, 1971328);
        (Zvt_i8, Resident_a { panel = 64 }, Shared, Core, 1, 1024, 128, 126832);
        (Zvt_i8, Resident_a { panel = 32 }, Shared, Dma, 4, 1024, 1024, 1621126);
        (Zvt_i8, Split_k { chunk = 1024 }, Shared, Dma, 8, 256, 4096, 3914286);
      ];
  [%expect
    {|
    zvt i8 resident-b b-shared c-by-core M=256 N=256 K=512: model 352790, measured 352458 (+0%)
    zvt i8 resident-b b-shared c-by-dma M=256 N=256 K=512: model 438223, measured 433341 (+1%)
    zvt f32 resident-b b-shared c-by-core M=256 N=128 K=512: model 681773, measured 655368 (+4%)
    rvv bf16 resident-b b-shared c-by-dma M=192 N=32 K=256: model 258740, measured 258608 (+0%)
    rvv f32 resident-b b-shared c-by-dma M=192 N=32 K=128: model 167955, measured 168368 (-0%)
    zvt i8 resident-b b-shared c-by-core M=512 N=256 K=1024: model 1092094, measured 1097986 (-1%)
    zvt i8 resident-b b-shared c-by-dma M=512 N=512 K=1024: model 2024895, measured 1961336 (+3%)
    zvt i8 resident-b b-shared c-by-core M=512 N=512 K=1024: model 2157683, measured 2173969 (-1%)
    zvt i8 resident-b b-shared c-split-8 M=512 N=512 K=1024: model 1864665, measured 1971328 (-5%)
    zvt i8 resident-a/64 a-shared c-by-core M=32 N=1024 K=128: model 129227, measured 126832 (+2%)
    zvt i8 resident-a/32 a-shared c-by-dma M=128 N=1024 K=1024: model 1678436, measured 1621126 (+4%)
    zvt i8 split-k/1024 b-shared c-by-dma M=256 N=256 K=4096: model 3979023, measured 3914286 (+2%) |}]

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
        ("32 KiB DTCM", normal, [ Rvv_i8; Zvt_i8 ], 32, 512, 128);
        ("32 KiB DTCM", normal, [ Rvv_i8; Zvt_i8 ], 32, 1024, 128);
        ("32 KiB DTCM", normal, [ Rvv_bf16; Rvv_f32 ], 24, 1024, 128);
        ("32 KiB DTCM", normal, [ Rvv_i8; Zvt_i8 ], 256, 64, 1024);
        ("32 KiB DTCM", normal, [ Zvt_i8 ], 256, 32, 1024);
        ("1 MiB DTCM", highmem, [ Zvt_i8 ], 128, 1024, 1024);
        ("1 MiB DTCM", highmem, [ Zvt_i8 ], 128, 2048, 1024);
        ("1 MiB DTCM", highmem, [ Rvv_i8 ], 64, 1024, 1024);
        ("1 MiB DTCM", highmem, [ Zvt_i8 ], 256, 256, 4096);
        ("1 MiB DTCM", highmem, [ Zvt_i8 ], 256, 512, 2048);
      ];
  [%expect
    {|
    32 KiB DTCM, 256 x 64 x 256:
      279765 cycles, 30.0 ops/cycle: rvv i8 resident-b b-shared c-by-dma M=256 N=64 K=256
      289529 cycles, 29.0 ops/cycle: rvv i8 resident-b b-by-dma c-by-dma M=256 N=64 K=256
      293488 cycles, 28.6 ops/cycle: rvv i8 resident-b b-shared c-by-core M=256 N=64 K=256
    32 KiB DTCM, 512 x 32 x 128:
      65980 cycles, 63.6 ops/cycle: zvt i8 resident-b b-shared c-by-core M=512 N=32 K=128
      68543 cycles, 61.2 ops/cycle: zvt i8 resident-b b-by-dma c-by-core M=512 N=32 K=128
      72539 cycles, 57.8 ops/cycle: zvt i8 resident-b b-shared c-split-28 M=512 N=32 K=128
    1 MiB DTCM, 512 x 64 x 1024:
      411886 cycles, 162.9 ops/cycle: zvt i8 resident-a/32 a-shared c-by-core M=512 N=64 K=1024
      417361 cycles, 160.8 ops/cycle: zvt i8 resident-a/32 a-shared c-by-dma M=512 N=64 K=1024
      437484 cycles, 153.4 ops/cycle: zvt i8 resident-a/64 a-shared c-by-core M=512 N=64 K=1024
    1 MiB DTCM, 512 x 256 x 1024:
      1018883 cycles, 263.5 ops/cycle: zvt i8 resident-b b-shared c-split-20 M=512 N=256 K=1024
      1025795 cycles, 261.7 ops/cycle: zvt i8 resident-b b-shared c-split-16 M=512 N=256 K=1024
      1043305 cycles, 257.3 ops/cycle: zvt i8 resident-b b-shared c-split-24 M=512 N=256 K=1024
    1 MiB DTCM, 512 x 512 x 1024:
      1864665 cycles, 287.9 ops/cycle: zvt i8 resident-b b-shared c-split-8 M=512 N=512 K=1024
      1913511 cycles, 280.6 ops/cycle: zvt i8 resident-b b-shared c-split-12 M=512 N=512 K=1024
      1924377 cycles, 279.0 ops/cycle: zvt i8 resident-b b-shared c-split-4 M=512 N=512 K=1024
    1 MiB DTCM, 512 x 256 x 512:
      1861642 cycles, 72.1 ops/cycle: zvt f32 resident-b b-shared c-split-16 M=512 N=256 K=512
      1871094 cycles, 71.7 ops/cycle: zvt f32 resident-b b-shared c-split-12 M=512 N=256 K=512
      1886065 cycles, 71.2 ops/cycle: zvt f32 resident-b b-shared c-split-20 M=512 N=256 K=512
    32 KiB DTCM, 32 x 512 x 128:
      65980 cycles, 63.6 ops/cycle: zvt i8 resident-a/32 a-shared c-by-core M=32 N=512 K=128
      67403 cycles, 62.2 ops/cycle: zvt i8 resident-a/64 a-shared c-by-core M=32 N=512 K=128
      68543 cycles, 61.2 ops/cycle: zvt i8 resident-a/32 a-by-dma c-by-core M=32 N=512 K=128
    32 KiB DTCM, 32 x 1024 x 128:
      128661 cycles, 65.2 ops/cycle: zvt i8 resident-a/32 a-shared c-by-core M=32 N=1024 K=128
      129227 cycles, 64.9 ops/cycle: zvt i8 resident-a/64 a-shared c-by-core M=32 N=1024 K=128
      131223 cycles, 63.9 ops/cycle: zvt i8 resident-a/32 a-by-dma c-by-core M=32 N=1024 K=128
    32 KiB DTCM, 24 x 1024 x 128:
      512608 cycles, 12.3 ops/cycle: rvv bf16 resident-a/32 a-shared c-by-dma M=24 N=1024 K=128
      516452 cycles, 12.2 ops/cycle: rvv bf16 resident-a/32 a-by-dma c-by-dma M=24 N=1024 K=128
      551689 cycles, 11.4 ops/cycle: rvv bf16 resident-a/32 a-shared c-by-core M=24 N=1024 K=128
    32 KiB DTCM, 256 x 64 x 1024:
      1188612 cycles, 28.2 ops/cycle: rvv i8 split-k/128 b-shared c-by-dma M=256 N=64 K=1024
      1193738 cycles, 28.1 ops/cycle: rvv i8 split-k/128 b-by-dma c-by-dma M=256 N=64 K=1024
      1775357 cycles, 18.9 ops/cycle: rvv i8 split-k/64 b-shared c-by-dma M=256 N=64 K=1024
    32 KiB DTCM, 256 x 32 x 1024:
      736720 cycles, 22.8 ops/cycle: zvt i8 split-k/128 b-shared c-by-dma M=256 N=32 K=1024
      739282 cycles, 22.7 ops/cycle: zvt i8 split-k/128 b-by-dma c-by-dma M=256 N=32 K=1024
      1204625 cycles, 13.9 ops/cycle: zvt i8 split-k/64 b-shared c-by-dma M=256 N=32 K=1024
    1 MiB DTCM, 128 x 1024 x 1024:
      1209469 cycles, 221.9 ops/cycle: zvt i8 resident-a/32 a-shared c-by-core M=128 N=1024 K=1024
      1231671 cycles, 217.9 ops/cycle: zvt i8 resident-a/64 a-shared c-by-core M=128 N=1024 K=1024
      1278643 cycles, 209.9 ops/cycle: zvt i8 resident-a/128 a-shared c-by-core M=128 N=1024 K=1024
    1 MiB DTCM, 128 x 2048 x 1024:
      2362340 cycles, 227.3 ops/cycle: zvt i8 resident-a/32 a-shared c-by-core M=128 N=2048 K=1024
      2382830 cycles, 225.3 ops/cycle: zvt i8 resident-a/64 a-shared c-by-core M=128 N=2048 K=1024
      2428946 cycles, 221.0 ops/cycle: zvt i8 resident-a/128 a-shared c-by-core M=128 N=2048 K=1024
    1 MiB DTCM, 64 x 1024 x 1024:
      4370378 cycles, 30.7 ops/cycle: rvv i8 resident-a/64 a-shared c-by-dma M=64 N=1024 K=1024
      4411380 cycles, 30.4 ops/cycle: rvv i8 resident-a/64 a-by-dma c-by-dma M=64 N=1024 K=1024
      4418663 cycles, 30.4 ops/cycle: rvv i8 resident-a/64 a-shared c-by-core M=64 N=1024 K=1024
    1 MiB DTCM, 256 x 256 x 4096:
      3979023 cycles, 134.9 ops/cycle: zvt i8 split-k/1024 b-shared c-by-dma M=256 N=256 K=4096
      4143032 cycles, 129.6 ops/cycle: zvt i8 split-k/1024 b-by-dma c-by-dma M=256 N=256 K=4096
      5894892 cycles, 91.1 ops/cycle: zvt i8 split-k/512 b-shared c-by-dma M=256 N=256 K=4096
    1 MiB DTCM, 256 x 512 x 2048:
      1959276 cycles, 274.0 ops/cycle: zvt i8 resident-a/32 a-shared c-by-core M=256 N=512 K=2048
      2016220 cycles, 266.3 ops/cycle: zvt i8 resident-a/64 a-shared c-by-core M=256 N=512 K=2048
      2052472 cycles, 261.6 ops/cycle: zvt i8 resident-a/32 a-shared c-by-dma M=256 N=512 K=2048 |}]
