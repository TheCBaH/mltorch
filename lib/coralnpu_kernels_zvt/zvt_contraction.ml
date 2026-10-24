module R = Machine_rivet_riscv32
module Item = R.Rv32_kernel_item
module Insn = R.Rv32_kernel_insn
module Imm = R.Rv32_kernel_imm
module Label = R.Rv32_kernel_label
module Reg = R.Rv32_kernel_reg
module Vtype = R.Rv32_kernel_vtype
module Gpr = Reg.Gpr
module C = Contraction

module Mutation = struct
  type t = Full_tn | Swapped_operands
end

exception Refused of string

let refuse why = raise (Refused why)
let tile = 16

let bytes ?(size = 4L) (n : int64) =
  let b = Int64.mul size n in
  if Int64.compare b 0x7FFF_FFFFL > 0 then refuse "an offset past 31 bits"
  else Int64.to_int b

let ok = function Ok x -> x | Error _ -> refuse "a register or immediate"
let insn i = Item.insn i

(* [rd = n] for any 32-bit [n] *)
let li rd n =
  match Err.payload (Insn.load_imm rd n) with
  | Ok is -> List.map insn is
  | Error _ -> refuse "a constant"

(* [rd = rs + n], through [t0] when [n] is past an [addi] *)
let add_from rd rs n =
  if Imm.Imm12.(in_range (of_int n)) then
    [ insn (Insn.Addi { dst = rd; src = rs; imm = Imm.Imm12.of_int n }) ]
  else li Gpr.t0 n @ [ insn (Insn.Add { dst = rd; lhs = rs; rhs = Gpr.t0 }) ]

let msettn n =
  li Gpr.t3 n @ [ insn (Insn.Msettn { dst = Gpr.zero; avl = Gpr.t3 }) ]

let items ?mutation (d : C.t) =
  (match (d.C.numerics, d.C.seed) with
  | C.Numerics.Ordered, C.Seed.Zero -> ()
  | C.Numerics.Fused_sequential, _ ->
      refuse "the fused requirement: the engine rounds each product"
  | _, C.Seed.Tensor -> refuse "a contraction seeded from its output");
  (* an integer product: [vtmms] for a signed A, [vtmmu] for an unsigned one,
     [altfmt] for a signed B; four k steps a tile multiply at e8 m1, their
     slices two registers apart, into 32-bit accumulators *)
  let int8 =
    match d.C.format with
    | C.Element.F32 -> None
    | C.Element.I8 { a; b } ->
        Some
          ( (match a with
            | C.Sign.Signed -> Insn.Tile_op.Vtmms
            | C.Sign.Unsigned -> Insn.Tile_op.Vtmmu),
            match b with C.Sign.Signed -> 0x1C0 | C.Sign.Unsigned -> 0xC0 )
    | C.Element.Bf16 ->
        refuse "bfloat16 operands: the engine rounds two k steps at once"
  in
  let size = C.Element.bytes d.C.format in
  let tk = if Option.is_some int8 then 4 else 1 in
  if bytes d.C.c.C.Output.col <> 4 then refuse "C's columns not contiguous";
  let m = Int64.to_int d.C.m and n = Int64.to_int d.C.n in
  let k = Int64.to_int d.C.k and batch = Int64.to_int d.C.batch in
  let blocks = (m + tile - 1) / tile * ((n + tile - 1) / tile) * batch in
  if blocks > 256 then refuse "more than 256 output tiles";
  let a_outer = bytes ~size d.C.a.C.Operand.outer
  and a_k = bytes ~size d.C.a.C.Operand.k in
  let b_outer = bytes ~size d.C.b.C.Operand.outer
  and b_k = bytes ~size d.C.b.C.Operand.k in
  let c_row = bytes d.C.c.C.Output.row in
  let mt0 = ok (Reg.Mtreg.of_index 0) in
  let v8 = ok (Reg.Vreg.of_index 8) and v16 = ok (Reg.Vreg.of_index 16) in
  let labels = ref 0 in
  let fresh () =
    incr labels;
    Label.named (Printf.sprintf "zvt_c%d" !labels)
  in
  let eew = if Option.is_some int8 then Vtype.Sew.E8 else Vtype.Sew.E32 in
  let load dst ptr stride_bytes stride_reg =
    if stride_bytes = Int64.to_int size then
      insn (Insn.Vle { eew; dst; base = ptr })
    else insn (Insn.Vlse { eew; dst; base = ptr; stride = stride_reg })
  in
  let slice base j = ok (Reg.Vreg.of_index (base + (2 * j))) in
  (* one block of C: rows [i0, i0 + tm), columns [j0, j0 + tn), batch [bb] *)
  let block bb i0 tm j0 tn =
    let a_at = (bb * bytes ~size d.C.a.C.Operand.batch) + (i0 * a_outer) in
    let b_at = (bb * bytes ~size d.C.b.C.Operand.batch) + (j0 * b_outer) in
    let c_at = (bb * bytes d.C.c.C.Output.batch) + (i0 * c_row) + (j0 * 4) in
    let tn =
      if mutation = Some Mutation.Full_tn && tn < tile then tile else tn
    in
    let set ~tk ~widen ~vtype =
      li Gpr.t1 ((tm lsl 10) lor (tk lsl 5) lor widen)
      @ li Gpr.t2 vtype
      @ [
          insn
            (Insn.Msetmtype { dst = Gpr.zero; mtype = Gpr.t1; vtype = Gpr.t2 });
        ]
    in
    let compute_config =
      match int8 with
      | Some (_, vtype) -> set ~tk:4 ~widen:3 ~vtype
      | None -> set ~tk:1 ~widen:1 ~vtype:0xD2
    in
    let configure =
      compute_config @ msettn tn @ [ insn (Insn.Vtzero { dst = mt0 }) ]
    in
    (* [count] of a step's [tk] slices from t4 (A) and t5 (B), slice [j] at
       [j] k strides (through t0 past the first), the rest zero; then the tile multiply *)
    let step count =
      let operand base ptr ~stride ~k_bytes stride_reg =
        List.concat
          (List.init tk (fun j ->
               if j = 0 then [ load (slice base 0) ptr stride stride_reg ]
               else if j < count then
                 add_from Gpr.t0 ptr (j * k_bytes)
                 @ [ load (slice base j) Gpr.t0 stride stride_reg ]
               else
                 [
                   insn
                     (Insn.Vmv_v_i { dst = slice base j; imm = Imm.Simm5.zero });
                 ]))
      in
      msettn tm
      @ operand 8 Gpr.t4 ~stride:a_outer ~k_bytes:a_k Gpr.s1
      @ msettn tn
      @ operand 16 Gpr.t5 ~stride:b_outer ~k_bytes:b_k Gpr.s2
      @ [
          (let a, b =
             if mutation = Some Mutation.Swapped_operands then (v16, v8)
             else (v8, v16)
           in
           let op =
             match int8 with Some (op, _) -> op | None -> Insn.Tile_op.Vtfmm
           in
           insn (Insn.Vtmm { op; acc = mt0; a; b }));
        ]
      @ add_from Gpr.t4 Gpr.t4 (tk * a_k)
      @ add_from Gpr.t5 Gpr.t5 (tk * b_k)
    in
    let steps =
      if k = 0 then []
      else
        let full = k / tk and rest = k mod tk in
        add_from Gpr.t4 Gpr.a0 a_at
        @ add_from Gpr.t5 Gpr.a1 b_at
        @ (if full = 0 then []
           else
             let l = fresh () in
             li Gpr.t6 full
             @ [ Item.label l ]
             @ step tk
             @ [
                 insn
                   (Insn.Addi
                      {
                        dst = Gpr.t6;
                        src = Gpr.t6;
                        imm = Imm.Imm12.of_int (-1);
                      });
                 insn (Insn.Bnez { src = Gpr.t6; target = l });
               ])
        @ if rest > 0 then step rest else []
    in
    (* the 32-bit accumulators stored under the binary32 configuration *)
    let stores =
      (if Option.is_some int8 then set ~tk:1 ~widen:1 ~vtype:0xD2 else [])
      @ msettn tn
      @ add_from Gpr.t4 Gpr.a2 c_at
      @ List.concat
          (List.init tm (fun r ->
               li Gpr.t3 r
               @ [
                   insn
                     (Insn.Vtse
                        { eew = Vtype.Sew.E32; slice = Gpr.t3; base = Gpr.t4 });
                 ]
               @ add_from Gpr.t4 Gpr.t4 c_row))
    in
    configure @ steps @ stores
  in
  let blocks =
    List.concat
      (List.init batch (fun bb ->
           List.concat
             (List.init
                ((m + tile - 1) / tile)
                (fun bi ->
                  let i0 = bi * tile in
                  List.concat
                    (List.init
                       ((n + tile - 1) / tile)
                       (fun bj ->
                         let j0 = bj * tile in
                         block bb i0 (min tile (m - i0)) j0 (min tile (n - j0))))))))
  in
  let name = C.symbol d in
  let entry = Label.named name in
  let save = [ Gpr.s1; Gpr.s2 ] in
  [ Item.text; Item.align 4; Item.export ~name entry; Item.label entry ]
  @ [
      insn
        (Insn.Addi { dst = Gpr.sp; src = Gpr.sp; imm = Imm.Imm12.of_int (-16) });
    ]
  @ List.mapi
      (fun j s ->
        insn
          (Insn.Sw { src = s; base = Gpr.sp; offset = Imm.Imm12.of_int (4 * j) }))
      save
  @ li Gpr.s1 a_outer @ li Gpr.s2 b_outer @ blocks
  @ List.mapi
      (fun j s ->
        insn
          (Insn.Lw { dst = s; base = Gpr.sp; offset = Imm.Imm12.of_int (4 * j) }))
      save
  @ [
      insn (Insn.Addi { dst = Gpr.sp; src = Gpr.sp; imm = Imm.Imm12.of_int 16 });
      insn (Insn.Li { dst = Gpr.a0; imm = Imm.Li.of_int 0 });
      insn Insn.Ret;
    ]

let zvt =
  Result.map
    (R.Rv32_kernel_lower.Extension.make ~feature:Insn.zvt_feature)
    (Coral_isa.Coral_zvt.definition ())

let module_ ?mutation d =
  match items ?mutation d with
  | exception Refused why -> Error why
  | items -> (
      match zvt with
      | Error _ -> Error "no Zvt encoder"
      | Ok ext -> (
          match
            Err.payload (R.Rv32_kernel.finalize ~name:(C.symbol d) items)
          with
          | Error e -> Error (Fmt.str "%a" R.Rv32_kernel.pp_error e)
          | Ok k -> (
              match
                Err.payload
                  (R.Rv32_kernel_lower.module_ ~extensions:[ ext ]
                     ~profile:R.Rv32_profile.Rv32imf_zve32f_ilp32f k)
              with
              | Ok m -> Ok m
              | Error e -> Error (Fmt.str "%a" R.Rv32_kernel_lower.pp_error e)))
      )
