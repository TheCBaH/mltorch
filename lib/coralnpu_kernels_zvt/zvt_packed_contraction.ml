module R = Machine_rivet_riscv32
module Item = R.Rv32_kernel_item
module Insn = R.Rv32_kernel_insn
module Imm = R.Rv32_kernel_imm
module Label = R.Rv32_kernel_label
module Reg = R.Rv32_kernel_reg
module Vtype = R.Rv32_kernel_vtype
module Gpr = Reg.Gpr
module C = Contraction
module P = Peak_zvt
module K = Coralnpu_kernels.Peak_kernels

module Mutation = struct
  type t = A_pitch | Short_copy | Short_k
end

module Stage = struct
  type t = Always | Outside of { base : int; bytes : int }
end

exception Refused of string

let refuse why = raise (Refused why)
let tile = 16
let insn i = Item.insn i

let bytes (n : int64) =
  let b = Int64.mul 2L n in
  if Int64.compare b 0x7FFF_FFFFL > 0 || Int64.compare b 0L < 0 then
    refuse "an offset past 31 bits"
  else Int64.to_int b

let ok = function Ok x -> x | Error _ -> refuse "a register or immediate"

let li rd n =
  match Err.payload (Insn.load_imm rd n) with
  | Ok is -> List.map insn is
  | Error _ -> refuse "a constant"

(* [rd = rs + n], through [tmp] when [n] is past an [addi] *)
let add_from ~tmp rd rs n =
  if n = 0 && rd = rs then []
  else if Imm.Imm12.(in_range (of_int n)) then
    [ insn (Insn.Addi { dst = rd; src = rs; imm = Imm.Imm12.of_int n }) ]
  else li tmp n @ [ insn (Insn.Add { dst = rd; lhs = rs; rhs = tmp }) ]

let sew16_m4 =
  {
    Vtype.sew = Vtype.Sew.E16;
    lmul = Vtype.Lmul.M4;
    tail = Vtype.Policy.Agnostic;
    mask = Vtype.Policy.Agnostic;
  }

let set_vl n =
  li Gpr.t3 n
  @ [ insn (Insn.Vsetvli { avl = Vtype.Avl.Reg Gpr.t3; vtype = sew16_m4 }) ]

(* the block plans: shapes, then each distinct block's rows *)
let plan ~rows ~depth ~n =
  P.Tile_gemm.plan Insn.Tile_op.Vtfmm_alt
    (P.Tile.shape Insn.Tile_op.Vtfmm_alt)
    {
      K.Gemm.rows;
      width = n;
      depth;
      blocks = 1;
      src = K.Elem.Bf16;
      dst = K.Elem.F32;
      sa = 1;
      sb = 1;
    }

(* A contraction whose C is stored transposed (its rows one element apart) is
   run as its transpose, the operands exchanged: C's columns are then
   contiguous, and it is A's and B's k rows that are read in place or packed. *)
let swapped (d : C.t) = not (Int64.equal d.C.c.C.Output.col 1L)
let normal d = if swapped d then C.transpose d else d

let geometry (d : C.t) =
  let d = normal d in
  (match (d.C.format, d.C.numerics, d.C.seed) with
  | C.Element.Bf16, C.Numerics.Paired, C.Seed.Zero -> ()
  | _ -> refuse "not a paired bfloat16 contraction from zero");
  if not (Int64.equal d.C.c.C.Output.col 1L) then
    refuse "neither of C's strides is one element";
  let m = Int64.to_int d.C.m and n = Int64.to_int d.C.n in
  let k = Int64.to_int d.C.k in
  if k < 1 then refuse "an empty k";
  if m > 0x1_0000 || n > 0x1_0000 || k > 0x1_0000 then
    refuse "an extent past 2^16";
  let depth = (k + 3) / 4 * 4 in
  let full = m / 32 and rest = m mod 32 in
  let tails =
    if rest = 0 then [] else if rest <= 16 then [ rest ] else [ 16; rest - 16 ]
  in
  (m, n, k, depth, full, tails)

(* An operand whose rows of k are whole contiguous runs of the tile's columns
   (A transposed, B as it is) is read where it is, with its row pitch, when k
   needs no zero steps and the pitch is an [addi]'s. *)
let in_place (d : C.t) =
  let d = normal d in
  let _, _, k, depth, _, _ = geometry d in
  let fits x = x >= 0 && x < 2048 in
  ( k = depth
    && Int64.equal d.C.a.C.Operand.outer 1L
    && fits (2 * Int64.to_int d.C.a.C.Operand.k),
    k = depth
    && Int64.equal d.C.b.C.Operand.outer 1L
    && fits (2 * Int64.to_int d.C.b.C.Operand.k) )

let sizes ?stage_b (d : C.t) =
  let d = normal d in
  let _, n, _, depth, full, tails = geometry d in
  let a_ip, b_ip = in_place d in
  let rows = (if full > 0 then [ 32 ] else []) @ tails in
  let b_plan = plan ~rows:(List.hd rows) ~depth ~n in
  let b_bytes =
    if b_ip then
      if stage_b <> None then (depth * 2 * Int64.to_int d.C.b.C.Operand.k) + 128
      else 0
    else
      List.fold_left
        (fun acc cols -> acc + (depth * List.length cols * 32))
        128
        (P.Tile_gemm.passes b_plan)
  in
  let a_bytes =
    if a_ip then 0
    else
      List.fold_left
        (fun acc r -> max acc (((r + 15) / 16 * 32 * depth) + 128))
        0 rows
  in
  (b_bytes, a_bytes)

(* The bytes past an operand's end that the blocks' load ahead reads: a step
   of k rows of the operand read in place, and a tile's width. *)
let overread d0 =
  let d = normal d0 in
  let a_ip, b_ip = in_place d in
  let one ip (o : C.Operand.t) =
    if ip then (4 * 2 * Int64.to_int o.C.Operand.k) + 128 else 0
  in
  max (one a_ip d.C.a) (one b_ip d.C.b)

(* Whether every operand it packs goes by words, two k steps a strided load
   (see [strided_rows]): else an element a load, which the plain function
   matches or beats. *)
let fast d0 =
  let d = normal d0 in
  let a_ip, b_ip = in_place d in
  let words ip (o : C.Operand.t) =
    ip
    ||
    let stride = 2 * Int64.to_int o.C.Operand.outer in
    Int64.equal o.C.Operand.k 1L && stride mod 4 = 0 && stride > 4
  in
  words a_ip d.C.a && words b_ip d.C.b

(* A B read in place is worth copying into the DTCM when more than one block
   of A reads it. *)
let reuses_b d0 =
  match normal d0 with
  | exception Refused _ -> false
  | d -> (
      match in_place d with
      | exception Refused _ -> false
      | _, b_ip -> b_ip && Int64.compare d.C.m 32L > 0)

let panel_bytes ?stage_b d =
  match sizes ?stage_b d with
  | exception Refused why -> Error why
  | exception Invalid_argument why -> Error why
  | b_bytes, a_bytes -> Ok (b_bytes + a_bytes)

let items ?mutation ?stage_b (d0 : C.t) =
  let d = normal d0 in
  let m, n, k, depth, full, tails = geometry d in
  let b_bytes, a_bytes = sizes ?stage_b d in
  let a_ip, b_ip = in_place d in
  let name = C.symbol d0 in
  let panel_b = Label.named (name ^ "_pb")
  and panel_a = Label.named (name ^ "_pa") in
  let la dst target = insn (Insn.La { dst; target }) in
  let k_packed = if mutation = Some Mutation.Short_k then k - 1 else k in
  let a_outer = bytes d.C.a.C.Operand.outer and a_k = bytes d.C.a.C.Operand.k in
  let b_outer = bytes d.C.b.C.Operand.outer and b_k = bytes d.C.b.C.Operand.k in
  let c_row = 4 * Int64.to_int d.C.c.C.Output.row in
  let a_outer_real =
    if mutation = Some Mutation.A_pitch then a_outer + 2 else a_outer
  in
  let v0 = ok (Reg.Vreg.of_index 0) in
  let labels = ref 0 in
  let fresh stem =
    incr labels;
    Label.named (Printf.sprintf "zp_%s%d" stem !labels)
  in
  (* [count] trips of [body] *)
  let loop count body =
    if count <= 0 then []
    else
      let l = fresh "l" in
      li Gpr.t2 count
      @ [ Item.label l ]
      @ body
      @ [
          insn
            (Insn.Addi
               { dst = Gpr.t2; src = Gpr.t2; imm = Imm.Imm12.of_int (-1) });
          insn (Insn.Bnez { src = Gpr.t2; target = l });
        ]
  in
  let load stride_bytes stride_reg =
    if stride_bytes = 2 then
      insn (Insn.Vle { eew = Vtype.Sew.E16; dst = v0; base = Gpr.t0 })
    else
      insn
        (Insn.Vlse
           { eew = Vtype.Sew.E16; dst = v0; base = Gpr.t0; stride = stride_reg })
  in
  let store =
    insn (Insn.Vse { eew = Vtype.Sew.E16; src = v0; base = Gpr.t1 })
  in
  (* [count] k rows of a panel from the operand at t0 into the panel at t1,
     each a vector over the tile's indices ([stride] bytes apart, held in
     [stride_reg]) and [row_bytes] apart in the panel. Such a load takes an
     element a cycle. Where two k steps share a word (the stride a whole number
     of words, k adjacent in memory), a strided load of words brings both for
     the cost of one, and two narrowing shifts split them into the two k steps'
     slices. *)
  let strided_rows ~stride ~stride_reg ~k_bytes ~row_bytes count =
    let v8 = ok (Reg.Vreg.of_index 8) and v4 = ok (Reg.Vreg.of_index 4) in
    let pair =
      [
        insn
          (Insn.Vlse
             {
               eew = Vtype.Sew.E32;
               dst = v8;
               base = Gpr.t0;
               stride = stride_reg;
             });
        insn (Insn.Vnsrl_wi { dst = v0; src = v8; shamt = Imm.Shamt.of_int 0 });
        insn (Insn.Vnsrl_wi { dst = v4; src = v8; shamt = Imm.Shamt.of_int 16 });
        store;
      ]
      @ add_from ~tmp:Gpr.t4 Gpr.t1 Gpr.t1 row_bytes
      @ [ insn (Insn.Vse { eew = Vtype.Sew.E16; src = v4; base = Gpr.t1 }) ]
      @ add_from ~tmp:Gpr.t4 Gpr.t1 Gpr.t1 row_bytes
      @ add_from ~tmp:Gpr.t4 Gpr.t0 Gpr.t0 4
    in
    let single =
      [ load stride stride_reg; store ]
      @ add_from ~tmp:Gpr.t4 Gpr.t0 Gpr.t0 k_bytes
      @ add_from ~tmp:Gpr.t4 Gpr.t1 Gpr.t1 row_bytes
    in
    if k_bytes = 2 && stride mod 4 = 0 && stride > 4 then
      loop (count / 2) pair @ loop (count mod 2) single
    else loop count single
  in
  (* B's panels: pass [p] two tile columns at a time, k rows of up to 32
     columns packed one after another *)
  let b_plan = plan ~rows:(if full > 0 then 32 else List.hd tails) ~depth ~n in
  let sew16_m8 = { sew16_m4 with Vtype.lmul = Vtype.Lmul.M8 } in
  let v8 = ok (Reg.Vreg.of_index 8) in
  let pack_b =
    if b_ip && stage_b <> None then
      (* B's rows, pitch and all, copied into the panel by the core, 64
         elements at a time and the rest; B's address for the blocks is in s11 *)
      let elements =
        (depth * Int64.to_int d.C.b.C.Operand.k)
        - if mutation = Some Mutation.Short_copy then 64 else 0
      in
      let chunk n =
        li Gpr.t3 n
        @ [
            insn (Insn.Vsetvli { avl = Vtype.Avl.Reg Gpr.t3; vtype = sew16_m8 });
            insn (Insn.Vle { eew = Vtype.Sew.E16; dst = v8; base = Gpr.t0 });
            insn (Insn.Vse { eew = Vtype.Sew.E16; src = v8; base = Gpr.t1 });
          ]
        @ add_from ~tmp:Gpr.t4 Gpr.t0 Gpr.t0 (2 * n)
        @ add_from ~tmp:Gpr.t4 Gpr.t1 Gpr.t1 (2 * n)
      in
      let skip = fresh "in" in
      let outside =
        match stage_b with
        | Some (Stage.Outside { base; bytes }) ->
            (* B inside the DTCM is read where it is *)
            let bits =
              let rec go b n = if n lsr (b + 1) = 0 then b else go (b + 1) n in
              go 0 bytes
            in
            li Gpr.t0 base
            @ [
                insn (Insn.Sub { dst = Gpr.t0; lhs = Gpr.s2; rhs = Gpr.t0 });
                insn
                  (Insn.Srli
                     {
                       dst = Gpr.t0;
                       src = Gpr.t0;
                       shamt = Imm.Shamt.of_int bits;
                     });
                insn (Insn.Beqz { src = Gpr.t0; target = skip });
              ]
        | Some Stage.Always | None -> []
      in
      [ insn (Insn.Mv { dst = Gpr.s11; src = Gpr.s2 }) ]
      @ outside
      @ [
          insn (Insn.Mv { dst = Gpr.t0; src = Gpr.s2 });
          insn (Insn.Mv { dst = Gpr.t1; src = Gpr.s8 });
        ]
      @ loop (elements / 64) (chunk 64)
      @ (if elements mod 64 = 0 then [] else chunk (elements mod 64))
      @ [ insn (Insn.Mv { dst = Gpr.s11; src = Gpr.s8 }); Item.label skip ]
    else if b_ip then []
    else
      let at = ref 0 in
      List.concat
        (List.mapi
           (fun p cols ->
             let vl = min 32 (n - (32 * p)) in
             let pass_bytes = depth * List.length cols * 32 in
             let code =
               set_vl vl
               @ add_from ~tmp:Gpr.t4 Gpr.t0 Gpr.s2 (32 * p * b_outer)
               @ add_from ~tmp:Gpr.t4 Gpr.t1 Gpr.s8 !at
               @ strided_rows ~stride:b_outer ~stride_reg:Gpr.s10 ~k_bytes:b_k
                   ~row_bytes:(List.length cols * 32)
                   k_packed
             in
             at := !at + pass_bytes;
             code)
           (P.Tile_gemm.passes b_plan))
  in
  (* A's block of [rows] rows from s6: k steps of [rows] elements, a tile row
     or two wide *)
  let pack_a rows =
    if a_ip then []
    else
      set_vl rows
      @ [ insn (Insn.Mv { dst = Gpr.t0; src = Gpr.s6 }) ]
      @ [ la Gpr.t1 panel_a ]
      @ strided_rows ~stride:a_outer_real ~stride_reg:Gpr.s9 ~k_bytes:a_k
          ~row_bytes:(if rows > tile then 64 else 32)
          k_packed
  in
  let compute ~prefix rows =
    let p = plan ~rows ~depth ~n in
    let a_pitch =
      if a_ip then
        Some
          ((2 * Int64.to_int d.C.a.C.Operand.k)
          + if mutation = Some Mutation.A_pitch then 2 else 0)
      else None
    and b_pitch =
      if b_ip then Some (2 * Int64.to_int d.C.b.C.Operand.k) else None
    and b_pass = if b_ip then Some 64 else None in
    let blocks =
      ok (P.Tile_gemm.block_items ~prefix ?a_pitch ?b_pitch ?b_pass ~c_row p)
    in
    (if a_ip then [ insn (Insn.Mv { dst = Gpr.a3; src = Gpr.s6 }) ]
     else [ la Gpr.a3 panel_a ])
    @ [
        insn
          (Insn.Mv
             {
               dst = Gpr.a7;
               src =
                 (if b_ip then if stage_b <> None then Gpr.s11 else Gpr.s2
                  else Gpr.s8);
             });
        insn (Insn.Mv { dst = Gpr.a4; src = Gpr.s7 });
      ]
    @ List.map Fun.id blocks
  in
  let advance rows =
    add_from ~tmp:Gpr.t4 Gpr.s6 Gpr.s6 (rows * a_outer)
    @ add_from ~tmp:Gpr.t4 Gpr.s7 Gpr.s7 (rows * c_row)
  in
  let full_blocks =
    if full = 0 then []
    else
      let l = fresh "blk" in
      li Gpr.s5 full
      @ [ Item.label l ]
      @ pack_a 32
      @ compute ~prefix:"zp_full" 32
      @ advance 32
      @ [
          insn
            (Insn.Addi
               { dst = Gpr.s5; src = Gpr.s5; imm = Imm.Imm12.of_int (-1) });
          insn (Insn.Bnez { src = Gpr.s5; target = l });
        ]
  in
  let tail_blocks =
    List.concat
      (List.mapi
         (fun j rows ->
           pack_a rows
           @ compute ~prefix:(Printf.sprintf "zp_tail%d" j) rows
           @ advance rows)
         tails)
  in
  let batch = Int64.to_int d.C.batch in
  let per_batch =
    pack_b
    @ [
        insn (Insn.Mv { dst = Gpr.s6; src = Gpr.s1 });
        insn (Insn.Mv { dst = Gpr.s7; src = Gpr.s3 });
      ]
    @ full_blocks @ tail_blocks
    @ add_from ~tmp:Gpr.t4 Gpr.s1 Gpr.s1 (bytes d.C.a.C.Operand.batch)
    @ add_from ~tmp:Gpr.t4 Gpr.s2 Gpr.s2 (bytes d.C.b.C.Operand.batch)
    @ add_from ~tmp:Gpr.t4 Gpr.s3 Gpr.s3 (4 * Int64.to_int d.C.c.C.Output.batch)
  in
  let batch_loop =
    let l = fresh "batch" in
    li Gpr.s4 batch
    @ [ Item.label l ]
    @ per_batch
    @ [
        insn
          (Insn.Addi { dst = Gpr.s4; src = Gpr.s4; imm = Imm.Imm12.of_int (-1) });
        insn (Insn.Bnez { src = Gpr.s4; target = l });
      ]
  in
  ignore m;
  let saved = Gpr.[ s1; s2; s3; s4; s5; s6; s7; s8; s9; s10; s11 ] in
  let entry = Label.named name in
  [ Item.text; Item.align 4; Item.export ~name entry; Item.label entry ]
  @ [
      insn
        (Insn.Addi { dst = Gpr.sp; src = Gpr.sp; imm = Imm.Imm12.of_int (-48) });
    ]
  @ List.mapi
      (fun j s ->
        insn
          (Insn.Sw { src = s; base = Gpr.sp; offset = Imm.Imm12.of_int (4 * j) }))
      saved
  @ [
      insn
        (Insn.Mv { dst = Gpr.s1; src = (if swapped d0 then Gpr.a1 else Gpr.a0) });
      insn
        (Insn.Mv { dst = Gpr.s2; src = (if swapped d0 then Gpr.a0 else Gpr.a1) });
      insn (Insn.Mv { dst = Gpr.s3; src = Gpr.a2 });
      la Gpr.s8 panel_b;
    ]
  @ li Gpr.s9 a_outer_real @ li Gpr.s10 b_outer @ batch_loop
  @ List.mapi
      (fun j s ->
        insn
          (Insn.Lw { dst = s; base = Gpr.sp; offset = Imm.Imm12.of_int (4 * j) }))
      saved
  @ [
      insn (Insn.Addi { dst = Gpr.sp; src = Gpr.sp; imm = Imm.Imm12.of_int 48 });
      insn (Insn.Li { dst = Gpr.a0; imm = Imm.Li.of_int 0 });
      insn Insn.Ret;
      Item.bss;
      Item.align 64;
      Item.label panel_b;
      Item.zero b_bytes;
      Item.align 64;
      Item.label panel_a;
      Item.zero a_bytes;
    ]

let module_ ?mutation ?stage_b d =
  match items ?mutation ?stage_b d with
  | exception Refused why -> Error why
  | exception Invalid_argument why -> Error why
  | items -> (
      match P.zvt with
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
