(* [F32_math.exp] emitted over vector registers ({!Vexpr}): one function that
   loads, evaluates and stores a vector at a time, against [F32_math.Value.exp]
   bit for bit over the cases that tell the staged comparisons apart, with the
   registers the emission needs and the cycles an element takes at each
   register grouping. *)

module R = Machine_rivet_riscv32
module Item = R.Rv32_kernel_item
module Insn = R.Rv32_kernel_insn
module Imm = R.Rv32_kernel_imm
module Vtype = R.Rv32_kernel_vtype
module Label = R.Rv32_kernel_label
module Gpr = R.Rv32_kernel_reg.Gpr
module Vreg = R.Rv32_kernel_reg.Vreg
module Z = Coralnpu_kernels_zvt
module M = F32_math.Make (Z.Vexpr.Ops)
module H = Kernel_harness

let insn i = Item.insn i
let name = "vexp"
let imm12 = Imm.Imm12.of_int
let vreg i = Result.get_ok (Vreg.of_index i)

let lmul_of = function
  | 1 -> Vtype.Lmul.M1
  | 2 -> Vtype.Lmul.M2
  | 4 -> Vtype.Lmul.M4
  | _ -> Vtype.Lmul.M8

(* x in a0, the results in a1, the element count in a2 (a multiple of the
   vector length, 4 [lmul]) *)
let items ~lmul =
  let loop = Label.named "loop" in
  let vl = 4 * lmul in
  let x = Z.Vexpr.input 0 in
  let input = vreg lmul in
  let free =
    List.filter_map
      (fun i ->
        if i mod lmul = 0 && i <> 0 && i <> lmul then Some (vreg i) else None)
      (List.init 32 Fun.id)
  in
  match
    Z.Vexpr.emit ~lmul ~scratch:Gpr.t0 ~free
      ~inputs:[| (input, `Consumed) |]
      (M.exp x)
  with
  | Error e -> Error e
  | Ok plan ->
      let eew = Vtype.Sew.E32 in
      Ok
        ( plan,
          [ Item.text; Item.align 4; Item.export ~name (Label.named name) ]
          @ [ Item.label (Label.named name) ]
          @ [
              insn (Insn.Li { dst = Gpr.t3; imm = Imm.Li.of_int vl });
              insn
                (Insn.Vsetvli
                   {
                     avl = Vtype.Avl.Reg Gpr.t3;
                     vtype =
                       {
                         Vtype.sew = eew;
                         lmul = lmul_of lmul;
                         tail = Vtype.Policy.Agnostic;
                         mask = Vtype.Policy.Agnostic;
                       };
                   });
              Item.label loop;
              insn (Insn.Vle { eew; dst = input; base = Gpr.a0 });
            ]
          @ plan.Z.Vexpr.Plan.items
          @ [
              insn
                (Insn.Vse { eew; src = plan.Z.Vexpr.Plan.result; base = Gpr.a1 });
              insn
                (Insn.Addi { dst = Gpr.a0; src = Gpr.a0; imm = imm12 (4 * vl) });
              insn
                (Insn.Addi { dst = Gpr.a1; src = Gpr.a1; imm = imm12 (4 * vl) });
              insn (Insn.Addi { dst = Gpr.a2; src = Gpr.a2; imm = imm12 (-vl) });
              insn (Insn.Bnez { src = Gpr.a2; target = loop });
              insn Insn.Ret;
            ] )

let samples =
  let specials =
    [
      neg_infinity;
      -104.;
      -103.99;
      -100.;
      -90.;
      -87.5;
      -87.3;
      -87.;
      -20.;
      -1.;
      -0.5;
      -1e-10;
      -0.;
      0.;
      1e-10;
      0.5;
      1.;
      10.;
      20.;
      87.;
      88.;
      88.7;
      88.72;
      88.73;
      89.;
      89.5;
      100.;
      infinity;
      nan;
      1e-40;
      -1e-40;
      3.4e38;
      -3.4e38;
    ]
  in
  let random =
    List.init
      (1024 - List.length specials)
      (fun i ->
        let h = (i + 1) * 2654435761 land 0x3fffffff in
        (float_of_int (h mod 4001) -. 2200.) /. 20.)
  in
  Array.of_list (List.map F32_math.round32 (specials @ random))

let%expect_test "exp over vector registers on the core" =
  let n = Array.length samples in
  Fmt.pr "%d inputs@." n;
  List.iter
    (fun lmul ->
      match items ~lmul with
      | Error e -> Fmt.pr "lmul %d: %s@." lmul e
      | Ok (plan, items) -> (
          match Z.Kernel_module.of_items ~zvt:false ~name items with
          | Error e -> Fmt.pr "lmul %d: %s@." lmul e
          | Ok m -> (
              match
                H.run ~symbol:name ~modules:[ m ]
                  ~args:[ `Slot 0; `Slot 1; `Imm n ]
                  ~inputs:[ H.f32_bytes samples ]
                  ~outputs:[ 4 * n ]
                  ()
              with
              | Error e -> Fmt.pr "lmul %d: %s@." lmul e
              | Ok { outputs = out :: _; cycles; retired } ->
                  let got = H.f32_array out in
                  let wrong = ref 0 in
                  Array.iteri
                    (fun i x ->
                      if not (H.same_f32 got.(i) (F32_math.Value.exp x)) then
                        incr wrong)
                    samples;
                  Fmt.pr
                    "lmul %d: %d instructions, %d registers at most, %s, %Ld \
                     cycles (%.1f per element), %Ld retired@."
                    lmul
                    (List.length plan.Z.Vexpr.Plan.items)
                    plan.Z.Vexpr.Plan.peak
                    (if !wrong = 0 then "all equal"
                     else Fmt.str "%d DIFFER" !wrong)
                    cycles
                    (Int64.to_float cycles /. float_of_int n)
                    retired
              | Ok _ -> assert false)))
    [ 1; 2; 4 ];
  [%expect
    {|
    1024 inputs
    lmul 1: 352 instructions, 7 registers at most, all equal, 193339 cycles (188.8 per element), 91684 retired
    lmul 2: 352 instructions, 7 registers at most, all equal, 109627 cycles (107.1 per element), 45860 retired
    lmul 4: 352 instructions, 7 registers at most, all equal, 67387 cycles (65.8 per element), 22948 retired |}]
