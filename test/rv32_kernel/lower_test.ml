open Machine_rivet_riscv32
module K = Rv32_kernel
module Item = Rv32_kernel_item
module Insn = Rv32_kernel_insn
module Imm = Rv32_kernel_imm
module Label = Rv32_kernel_label
module Lower = Rv32_kernel_lower
module Vtype = Rv32_kernel_vtype
open Rv32_kernel_reg

let finalize ~name items =
  match Err.payload (K.finalize ~name items) with
  | Ok k -> k
  | Error e -> Fmt.failwith "%a" K.pp_error e

let imm12 = Imm.Imm12.of_int

let vtype sew lmul =
  {
    Vtype.sew;
    lmul;
    tail = Vtype.Policy.Agnostic;
    mask = Vtype.Policy.Agnostic;
  }

(* One sample of each variant beside the form [Rv32_asm] makes for it. *)
let samples =
  let open Rv32_asm in
  let loop = Label.named "loop" in
  [
    ( Insn.Add { dst = Gpr.a0; lhs = Gpr.a1; rhs = Gpr.t3 },
      insn "add" [ reg "a0"; reg "a1"; reg "t3" ] );
    ( Addi { dst = Gpr.a1; src = Gpr.a1; imm = imm12 (-1) },
      insn "addi" [ reg "a1"; reg "a1"; imm (-1) ] );
    ( Bnez { src = Gpr.a1; target = loop },
      insn "bnez" [ reg "a1"; sym ".Lk_loop" ] );
    ( Flh { dst = Fpr.fa0; base = Gpr.a1; offset = imm12 2 },
      insn "flh" [ reg "fa0"; mem "a1" 2 ] );
    ( Flw { dst = Fpr.fa4; base = Gpr.a1; offset = imm12 8 },
      insn "flw" [ reg "fa4"; mem "a1" 8 ] );
    (Li { dst = Gpr.a0; imm = Imm.Li.of_int 7 }, insn "li" [ reg "a0"; imm 7 ]);
    ( Lui { dst = Gpr.t3; imm = Imm.U20.of_int 1 },
      insn "lui" [ reg "t3"; imm 1 ] );
    ( Lw { dst = Gpr.s1; base = Gpr.a0; offset = imm12 4 },
      insn "lw" [ reg "s1"; mem "a0" 4 ] );
    (Mv { dst = Gpr.a2; src = Gpr.s1 }, insn "mv" [ reg "a2"; reg "s1" ]);
    (Ret, insn "ret" []);
    ( Srli { dst = Gpr.t3; src = Gpr.a4; shamt = Imm.Shamt.of_int 8 },
      insn "srli" [ reg "t3"; reg "a4"; imm 8 ] );
    ( Sw { src = Gpr.s0; base = Gpr.sp; offset = imm12 0 },
      insn "sw" [ reg "s0"; mem "sp" 0 ] );
    ( Vfmacc_vf { acc = Vreg.v8; scalar = Fpr.fa4; src = Vreg.v0 },
      insn "vfmacc.vf" [ reg "v8"; reg "fa4"; reg "v0" ] );
    ( Vfwmaccbf16_vf { acc = Vreg.v8; scalar = Fpr.fa0; src = Vreg.v0 },
      insn "vfwmaccbf16.vf" [ reg "v8"; reg "fa0"; reg "v0" ] );
    ( Vle { eew = Vtype.Sew.E16; dst = Vreg.v4; base = Gpr.a3 },
      insn "vle16.v" [ reg "v4"; mem "a3" 0 ] );
    ( Vmacc_vx { acc = Vreg.v8; scalar = Gpr.a4; src = Vreg.v0 },
      insn "vmacc.vx" [ reg "v8"; reg "a4"; reg "v0" ] );
    ( Vmv_v_i { dst = Vreg.v16; imm = Imm.Simm5.zero },
      insn "vmv.v.i" [ reg "v16"; imm 0 ] );
    ( Vmv_v_x { dst = Vreg.v8; src = Gpr.t1 },
      insn "vmv.v.x" [ reg "v8"; reg "t1" ] );
    ( Vse { eew = Vtype.Sew.E32; src = Vreg.v16; base = Gpr.a7 },
      insn "vse32.v" [ reg "v16"; mem "a7" 0 ] );
    ( Vsetvli
        {
          avl = Vtype.Avl.Reg Gpr.t6;
          vtype = vtype Vtype.Sew.E16 Vtype.Lmul.M4;
        },
      insn "vsetvli"
        [ reg "zero"; reg "t6"; sym "e16"; sym "m4"; sym "ta"; sym "ma" ] );
    ( Vsetvli
        {
          avl = Vtype.Avl.Vlmax { vl = Gpr.t0 };
          vtype = vtype Vtype.Sew.E8 Vtype.Lmul.M2;
        },
      insn "vsetvli"
        [ reg "t0"; reg "zero"; sym "e8"; sym "m2"; sym "ta"; sym "ma" ] );
    ( Vv { op = Insn.Vv_op.Vadd; dst = Vreg.v16; a = Vreg.v8; b = Vreg.v12 },
      insn "vadd.vv" [ reg "v16"; reg "v8"; reg "v12" ] );
    ( Vwmacc_vx { acc = Vreg.v16; scalar = Gpr.t4; src = Vreg.v4 },
      insn "vwmacc.vx" [ reg "v16"; reg "t4"; reg "v4" ] );
  ]

let rivet_string = function
  | Asm_core.Normalized_ast.Instruction { insn; _ } ->
      Fmt.str "%a" Riscv32_encode.Instruction.pp insn
  | _ -> assert false

let lower_profile = Rv32_profile.Rv32imf_zve32f_zvfbfwma_ilp32f

let lower ?(profile = lower_profile) items =
  Err.payload (Lower.module_ ~profile (finalize ~name:"k" items))

(* Variant-level equivalence, independent of the peak kernels. *)
let%expect_test "each variant lowers to the form Rv32_asm makes" =
  List.iter
    (fun (typed, expected) ->
      let items = [ Item.label (Label.named "loop"); Item.insn typed ] in
      match lower items with
      | Error e -> Fmt.pr "REFUSED: %a@." Lower.pp_error e
      | Ok m -> (
          match m.Asm_core.Normalized_ast.items with
          | [ _; got ] ->
              if rivet_string got <> rivet_string expected then
                Fmt.pr "DIFFERS: %s vs %s@." (rivet_string got)
                  (rivet_string expected)
          | _ -> Fmt.pr "unexpected module shape@."))
    samples;
  Fmt.pr "%d variants compared@." (List.length samples);
  [%expect {| 23 variants compared |}]

let bf16 =
  [
    Item.insn
      (Insn.Vfwmaccbf16_vf { acc = Vreg.v8; scalar = Fpr.fa0; src = Vreg.v0 });
  ]

let%expect_test "a profile refusal is the item's, with Rivet's feature kind" =
  (match
     lower ~profile:Rv32_profile.Rv32imf_zve32f_ilp32f
       (Item.note "bf16 step" :: bf16)
   with
  | Ok _ -> Fmt.pr "lowered@."
  | Error e -> Fmt.pr "%a@." Lower.pp_error e);
  (match lower bf16 with
  | Ok _ -> Fmt.pr "lowered under the BF16 profile@."
  | Error e -> Fmt.pr "%a@." Lower.pp_error e);
  [%expect
    {|
    item 1 (bf16 step): vfwmaccbf16.vf requires feature zvfbfwma, which is not enabled
    lowered under the BF16 profile |}]

let%expect_test "the error's detection origin is Rivet's, not the adapter's" =
  (match
     Lower.module_ ~profile:Rv32_profile.Rv32imf_zve32f_ilp32f
       (finalize ~name:"k" bf16)
   with
  | Ok _ -> Fmt.pr "lowered@."
  | Error e ->
      let origin = Err.Error.origin e in
      let source =
        Option.bind origin Err.Origin.source
        |> Option.map (Fmt.str "%a" Err.Source.pp)
      in
      Fmt.pr "%s@." (Option.value source ~default:"no source"));
  [%expect {| targets/riscv_family/riscv_family_encode.ml:1532:25-32 |}]

let plan items =
  match lower items with
  | Error e -> Fmt.pr "lower: %a@." Lower.pp_error e
  | Ok m -> (
      match
        Err.payload (Rv32_image.plan ~profile:lower_profile ~entry:"e" [ m ])
      with
      | Error e -> Fmt.pr "plan: %a@." Rv32_image.Error.pp e
      | Ok laid -> (
          match Err.payload (Rv32_image.bind ~base:0x10000L laid) with
          | Ok _ -> Fmt.pr "bound@."
          | Error e -> Fmt.pr "bind: %a@." Rv32_image.Error.pp e))

let entry =
  [
    Item.text;
    Item.export ~name:"e" (Label.named "e");
    Item.label (Label.named "e");
  ]

(* Two things only the image planner can see. The encoder does not check a
   vector register group against LMUL (v1 at m4 binds), and a branch distance
   exists only once the layout does. The second fails at bind with Rivet's
   rendered text and its own origin ("synthesized by image"): the item's
   producer origin does not reach it, and no typed wrapper does either. *)
let%expect_test "what only the image planner sees" =
  plan
    (entry
    @ [
        Item.note "misaligned group";
        Item.insn
          (Insn.Vsetvli
             {
               avl = Vtype.Avl.Reg Gpr.t6;
               vtype = vtype Vtype.Sew.E8 Vtype.Lmul.M4;
             });
        Item.insn
          (Insn.Vv
             { op = Insn.Vv_op.Vadd; dst = Vreg.v1; a = Vreg.v2; b = Vreg.v3 });
        Item.insn Insn.Ret;
      ]);
  plan
    (entry
    @ [
        Item.note "far branch";
        Item.insn (Insn.Bnez { src = Gpr.a0; target = Label.named "far" });
        Item.align 8192;
        Item.label (Label.named "far");
        Item.insn Insn.Ret;
      ]);
  [%expect
    {|
    bound
    bind: <synthesized by image>: error[riscv32.fixup]: branch immediate is out of range |}]
