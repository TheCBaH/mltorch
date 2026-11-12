module R = Machine_rivet_riscv32
module Item = R.Rv32_kernel_item
module Insn = R.Rv32_kernel_insn
module Imm = R.Rv32_kernel_imm
module Label = R.Rv32_kernel_label
module Reg = R.Rv32_kernel_reg
module Vtype = R.Rv32_kernel_vtype
module Gpr = Reg.Gpr
module C = Contraction
module Kernel_refusal = Coralnpu_kernels.Kernel_refusal

module Mutation = struct
  type t = Full_tn | Swapped_operands
end

let tile = 16

let bytes esc ?(size = 4L) (n : int64) =
  let b = Int64.mul size n in
  if Int64.compare b 0x7FFF_FFFFL > 0 then
    Kernel_refusal.refuse esc Offset_past_31_bits
  else Int64.to_int b

let ok esc = function
  | Ok x -> x
  | Error _ -> Kernel_refusal.refuse esc Register_or_immediate

let insn i = Item.insn i

(* [rd = n] for any 32-bit [n] *)
let li esc rd n =
  match Err.payload (Insn.load_imm rd n) with
  | Ok is -> List.map insn is
  | Error _ -> Kernel_refusal.refuse esc Constant

(* [rd = rs + n], through [t0] when [n] is past an [addi] *)
let add_from esc rd rs n =
  if Imm.Imm12.(in_range (of_int n)) then
    [ insn (Insn.Addi { dst = rd; src = rs; imm = Imm.Imm12.of_int n }) ]
  else
    li esc Gpr.t0 n @ [ insn (Insn.Add { dst = rd; lhs = rs; rhs = Gpr.t0 }) ]

let msettn esc n =
  li esc Gpr.t3 n @ [ insn (Insn.Msettn { dst = Gpr.zero; avl = Gpr.t3 }) ]

let items esc ?mutation (d : C.t) =
  let ok x = ok esc x
  and li rd n = li esc rd n
  and bytes ?size n = bytes esc ?size n
  and add_from rd rs n = add_from esc rd rs n
  and msettn n = msettn esc n in
  (match (d.C.numerics, d.C.format, d.C.seed) with
  | C.Numerics.Fused_sequential, _, _ ->
      Kernel_refusal.refuse esc Fused_requirement
  | C.Numerics.Ordered, C.Element.Bf16, _ ->
      Kernel_refusal.refuse esc Bf16_ordered
  | _, _, C.Seed.Tensor -> Kernel_refusal.refuse esc Seeded_from_output
  | (C.Numerics.Ordered | C.Numerics.Paired), _, C.Seed.Zero -> ());
  (* an integer product: [vtmms] for a signed A, [vtmmu] for an unsigned one,
     [altfmt] for a signed B; four k steps a tile multiply at e8 m1, their
     slices two registers apart, into 32-bit accumulators. A bfloat16 one is
     [vtfmm.alt]: two k steps at e16 m2 with [altfmt], their slices four
     registers apart, each tile multiply adding the sum of both
     products, rounded to odd, to the accumulator *)
  let int8 =
    match d.C.format with
    | C.Element.F32 | C.Element.Bf16 -> None
    | C.Element.I8 { a; b } ->
        Some
          ( (match a with
            | C.Sign.Signed -> Insn.Tile_op.Vtmms
            | C.Sign.Unsigned -> Insn.Tile_op.Vtmmu),
            match b with C.Sign.Signed -> 0x1C0 | C.Sign.Unsigned -> 0xC0 )
  in
  let bf16 = d.C.format = C.Element.Bf16 in
  let size = C.Element.bytes d.C.format in
  let tk = if Option.is_some int8 then 4 else if bf16 then 2 else 1 in
  let slice_gap = if bf16 then 4 else 2 in
  if bytes d.C.c.C.Output.col <> 4 then
    Kernel_refusal.refuse esc C_columns_not_contiguous;
  let m = Int64.to_int d.C.m and n = Int64.to_int d.C.n in
  let k = Int64.to_int d.C.k and batch = Int64.to_int d.C.batch in
  let blocks = (m + tile - 1) / tile * ((n + tile - 1) / tile) * batch in
  if blocks > 256 then Kernel_refusal.refuse esc Too_many_output_tiles;
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
  let eew =
    if Option.is_some int8 then Vtype.Sew.E8
    else if bf16 then Vtype.Sew.E16
    else Vtype.Sew.E32
  in
  let load dst ptr stride_bytes stride_reg =
    if stride_bytes = Int64.to_int size then
      insn (Insn.Vle { eew; dst; base = ptr })
    else insn (Insn.Vlse { eew; dst; base = ptr; stride = stride_reg })
  in
  let slice base j = ok (Reg.Vreg.of_index (base + (slice_gap * j))) in
  (* one block of C: rows [i0, i0 + tm), columns [j0, j0 + tn), batch [bb];
     [relative]: B and C are addressed from [s3] and [s4], which a loop over
     column tiles advances, instead of from their arguments *)
  let block ?(relative = false) bb i0 tm j0 tn =
    let a_at = (bb * bytes ~size d.C.a.C.Operand.batch) + (i0 * a_outer) in
    let b_base, b_at =
      if relative then (Gpr.s3, 0)
      else (Gpr.a1, (bb * bytes ~size d.C.b.C.Operand.batch) + (j0 * b_outer))
    in
    let c_base, c_at =
      if relative then (Gpr.s4, 0)
      else (Gpr.a2, (bb * bytes d.C.c.C.Output.batch) + (i0 * c_row) + (j0 * 4))
    in
    let tn =
      if mutation = Some Mutation.Full_tn && tn < tile then tile else tn
    in
    let set ~tk ~widen ~vtype =
      li Gpr.t1 ((tm lsl 10) lor (tk lsl 5) lor widen)
      @ li Gpr.t2 vtype
      @ [ insn (Insn.Msetmtype { mtype = Gpr.t1; vtype = Gpr.t2 }) ]
    in
    let compute_config =
      match int8 with
      | Some (_, vtype) -> set ~tk:4 ~widen:3 ~vtype
      | None when bf16 -> set ~tk:2 ~widen:2 ~vtype:0x1C9
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
             match int8 with
             | Some (op, _) -> op
             | None when bf16 -> Insn.Tile_op.Vtfmm_alt
             | None -> Insn.Tile_op.Vtfmm
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
        @ add_from Gpr.t5 b_base b_at
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
      (if Option.is_some int8 || bf16 then set ~tk:1 ~widen:1 ~vtype:0xD2
       else [])
      @ msettn tn
      @ add_from Gpr.t4 c_base c_at
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
  (* Per row of tiles, the full column tiles as one loop over [s5] (B and C
     advanced in [s3], [s4]) when there are two or more, then the partial one:
     the code grows with the rows of tiles, not with their product. *)
  let blocks =
    List.concat
      (List.init batch (fun bb ->
           List.concat
             (List.init
                ((m + tile - 1) / tile)
                (fun bi ->
                  let i0 = bi * tile in
                  let tm = min tile (m - i0) in
                  let full = n / tile and rest = n mod tile in
                  (if full >= 2 then
                     let l = fresh () in
                     add_from Gpr.s3 Gpr.a1
                       (bb * bytes ~size d.C.b.C.Operand.batch)
                     @ add_from Gpr.s4 Gpr.a2
                         ((bb * bytes d.C.c.C.Output.batch) + (i0 * c_row))
                     @ li Gpr.s5 full
                     @ [ Item.label l ]
                     @ block ~relative:true bb i0 tm 0 tile
                     @ add_from Gpr.s3 Gpr.s3 (tile * b_outer)
                     @ add_from Gpr.s4 Gpr.s4 (tile * 4)
                     @ [
                         insn
                           (Insn.Addi
                              {
                                dst = Gpr.s5;
                                src = Gpr.s5;
                                imm = Imm.Imm12.of_int (-1);
                              });
                         insn (Insn.Bnez { src = Gpr.s5; target = l });
                       ]
                   else
                     List.concat
                       (List.init full (fun bj ->
                            block bb i0 tm (bj * tile) tile)))
                  @ if rest > 0 then block bb i0 tm (full * tile) rest else []))))
  in
  let name = C.symbol d in
  let entry = Label.named name in
  let save = [ Gpr.s1; Gpr.s2; Gpr.s3; Gpr.s4; Gpr.s5 ] in
  [ Item.text; Item.align 4; Item.export ~name entry; Item.label entry ]
  @ [
      insn
        (Insn.Addi { dst = Gpr.sp; src = Gpr.sp; imm = Imm.Imm12.of_int (-32) });
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
      insn (Insn.Addi { dst = Gpr.sp; src = Gpr.sp; imm = Imm.Imm12.of_int 32 });
      insn (Insn.Li { dst = Gpr.a0; imm = Imm.Li.of_int 0 });
      insn Insn.Ret;
    ]

let zvt =
  Result.map
    (R.Rv32_kernel_lower.Extension.make ~feature:Insn.zvt_feature)
    (Coral_isa.Coral_zvt.definition ())

let module_ ?mutation d =
  Kernel_refusal.generate @@ fun esc ->
  let encoder pp e = Kernel_refusal.refuse esc (Encoder (Fmt.str "%a" pp e)) in
  let items = items esc ?mutation d in
  let ext =
    match zvt with
    | Ok ext -> ext
    | Error _ -> Kernel_refusal.refuse esc (Encoder "no Zvt encoder")
  in
  let k =
    match Err.payload (R.Rv32_kernel.finalize ~name:(C.symbol d) items) with
    | Ok k -> k
    | Error e -> encoder R.Rv32_kernel.pp_error e
  in
  match
    Err.payload
      (R.Rv32_kernel_lower.module_ ~extensions:[ ext ]
         ~profile:R.Rv32_profile.Rv32imf_zve32f_ilp32f k)
  with
  | Ok m -> m
  | Error e -> encoder R.Rv32_kernel_lower.pp_error e
