(* Each vector form the softmax needs, one function on the core over every
   pairing of eight special binary32 values, against OCaml's binary32: the
   encodings and their semantics, not only the assembler's spelling. *)

module R = Machine_rivet_riscv32
module Item = R.Rv32_kernel_item
module Insn = R.Rv32_kernel_insn
module Imm = R.Rv32_kernel_imm
module Vtype = R.Rv32_kernel_vtype
module Label = R.Rv32_kernel_label
module Gpr = R.Rv32_kernel_reg.Gpr
module Vreg = R.Rv32_kernel_reg.Vreg
module Z = Coralnpu_kernels_zvt
module H = Kernel_harness

let insn i = Item.insn i
let name = "forms"

let e32m1 =
  {
    Vtype.sew = Vtype.Sew.E32;
    lmul = Vtype.Lmul.M1;
    tail = Vtype.Policy.Agnostic;
    mask = Vtype.Policy.Agnostic;
  }

let imm12 = Imm.Imm12.of_int

(* [x] in a0, [y] in a1, the outputs in a2 (six vectors of four per step), the
   element count (a multiple of four) in a3. *)
let items =
  let loop = Label.named "loop" in
  let v = Vreg.(v8, v9, v10, v11, v12, v13) in
  let v8, v9, v10, v11, v12, v13 = v in
  let store r =
    [
      insn (Insn.Vse { eew = Vtype.Sew.E32; src = r; base = Gpr.a2 });
      insn (Insn.Addi { dst = Gpr.a2; src = Gpr.a2; imm = imm12 16 });
    ]
  in
  [ Item.text; Item.align 4; Item.export ~name (Label.named name) ]
  @ [ Item.label (Label.named name) ]
  @ [
      insn (Insn.Li { dst = Gpr.t3; imm = Imm.Li.of_int 4 });
      insn (Insn.Vsetvli { avl = Vtype.Avl.Reg Gpr.t3; vtype = e32m1 });
      insn (Insn.Li { dst = Gpr.t0; imm = Imm.Li.of_int 0 });
      insn (Insn.Lui { dst = Gpr.t0; imm = Imm.U20.of_int 0x40E00 });
      insn (Insn.Vmv_v_x { dst = Vreg.v14; src = Gpr.t0 });
      Item.label loop;
      insn (Insn.Vle { eew = Vtype.Sew.E32; dst = v8; base = Gpr.a0 });
      insn (Insn.Vle { eew = Vtype.Sew.E32; dst = v9; base = Gpr.a1 });
      insn (Insn.Valu { op = Insn.Alu_op.Vfsub; dst = v10; a = v8; b = v9 });
    ]
  @ store v10
  @ [ insn (Insn.Valu { op = Insn.Alu_op.Vfdiv; dst = v10; a = v8; b = v9 }) ]
  @ store v10
  @ [ insn (Insn.Valu { op = Insn.Alu_op.Vfmax; dst = v10; a = v8; b = v9 }) ]
  @ store v10
  @ [
      insn (Insn.Vmcmp { cmp = Insn.Cmp.Flt; dst = Vreg.v0; a = v8; b = v9 });
      insn (Insn.Vmerge_vvm { dst = v11; otherwise = v9; chosen = v8 });
    ]
  @ store v11
  @ [
      insn (Insn.Vmcmp { cmp = Insn.Cmp.Feq; dst = Vreg.v0; a = v8; b = v9 });
      insn (Insn.Vmerge_vvm { dst = v12; otherwise = v8; chosen = Vreg.v14 });
    ]
  @ store v12
  @ [
      insn (Insn.Vsrl_vi { dst = v13; src = v8; shamt = Imm.Shamt.of_int 16 });
      insn (Insn.Vand_vi { dst = v13; src = v13; imm = Imm.Simm5.of_int 1 });
    ]
  @ store v13
  @ [
      insn (Insn.Addi { dst = Gpr.a0; src = Gpr.a0; imm = imm12 16 });
      insn (Insn.Addi { dst = Gpr.a1; src = Gpr.a1; imm = imm12 16 });
      insn (Insn.Addi { dst = Gpr.a3; src = Gpr.a3; imm = imm12 (-4) });
      insn (Insn.Bnez { src = Gpr.a3; target = loop });
      insn Insn.Ret;
    ]

let specials = [| 0.; -0.; 1.5; -1.5; infinity; neg_infinity; nan; 1e-40 |]

(* RISC-V [fmax]: a NaN is dropped unless both are, and +0 is above -0 *)
let fmax a b =
  if Float.is_nan a then b
  else if Float.is_nan b then a
  else if a > b then a
  else if b > a then b
  else if Float.sign_bit a then b
  else a

let%expect_test "the softmax's vector forms on the core" =
  let n = 64 in
  let x = Array.init n (fun i -> specials.(i mod 8))
  and y = Array.init n (fun i -> specials.(i / 8)) in
  let r32 = F32_math.round32 in
  let expect =
    [|
      ("vfsub", fun a b -> r32 (a -. b));
      ("vfdiv", fun a b -> r32 (a /. b));
      ("vfmax", fmax);
      ("vmflt + vmerge", fun a b -> if a < b then a else b);
      ("vmfeq + vmerge", fun a b -> if a = b then 7. else a);
      ( "vsrl + vand",
        fun a _ ->
          Int32.float_of_bits
            (Int32.logand
               (Int32.shift_right_logical (Int32.bits_of_float a) 16)
               1l) );
    |]
  in
  match Z.Kernel_module.of_items ~zvt:false ~name items with
  | Error e -> Fmt.pr "%s@." e
  | Ok m ->
      (match
         H.run ~symbol:name ~modules:[ m ]
           ~args:[ `Slot 0; `Slot 1; `Slot 2; `Imm n ]
           ~inputs:[ H.f32_bytes x; H.f32_bytes y ]
           ~outputs:[ 4 * 6 * n ]
           ()
       with
      | Error e -> Fmt.pr "%s@." e
      | Ok { outputs = out :: _; _ } ->
          Array.iteri
            (fun form (what, f) ->
              let wrong = ref 0 in
              for i = 0 to n - 1 do
                let step = i / 4 and lane = i mod 4 in
                let got =
                  Int32.float_of_bits
                    (String.get_int32_le out
                       (4 * ((step * 24) + (form * 4) + lane)))
                in
                if not (H.same_f32 got (f x.(i) y.(i))) then incr wrong
              done;
              Fmt.pr "%s: %s@." what
                (if !wrong = 0 then "all 64 pairs equal"
                 else Fmt.str "%d of 64 DIFFER" !wrong))
            expect
      | Ok _ -> assert false);
      [%expect
        {|
    vfsub: all 64 pairs equal
    vfdiv: all 64 pairs equal
    vfmax: all 64 pairs equal
    vmflt + vmerge: all 64 pairs equal
    vmfeq + vmerge: all 64 pairs equal
    vsrl + vand: all 64 pairs equal |}]
