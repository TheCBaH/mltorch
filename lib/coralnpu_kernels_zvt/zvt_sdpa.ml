module R = Machine_rivet_riscv32
module Item = R.Rv32_kernel_item
module Insn = R.Rv32_kernel_insn
module Imm = R.Rv32_kernel_imm
module Label = R.Rv32_kernel_label
module Reg = R.Rv32_kernel_reg
module Vtype = R.Rv32_kernel_vtype
module Gpr = Reg.Gpr
module Vreg = Reg.Vreg
module A = Attention_block
module P = Peak_zvt
module K = Coralnpu_kernels.Peak_kernels
module M = F32_math.Make (Vexpr.Ops)

(* The phases a measurement may leave out: the result is wrong, the cycles are
   the others'. Alphabetical. *)
module Phase = struct
  type t = Context | Exp | Normalize | Scores
end

module Mutation = struct
  type t =
    | Max_stale
    | Rescale_omitted
    | Seed_ignored
    | V_in_place_in_dtcm
    | V_not_advanced
    | V_tail_unzeroed
end

exception Refused = Zvt_sdpa_geometry.Refused

let refuse = Zvt_sdpa_geometry.refuse
let bq = Zvt_sdpa_geometry.bq
let insn i = Item.insn i
let imm12 = Imm.Imm12.of_int
let vreg i = match Vreg.of_index i with Ok r -> r | Error _ -> assert false
let ok = function Ok x -> x | Error _ -> refuse "a register or immediate"

let li rd n =
  match Err.payload (Insn.load_imm rd n) with
  | Ok is -> List.map insn is
  | Error _ -> refuse "a constant"

(* [rd = rs + n], through [tmp] when [n] is past an [addi] *)
let add_from ~tmp rd rs n =
  if n = 0 && rd = rs then []
  else if Imm.Imm12.(in_range (of_int n)) then
    [ insn (Insn.Addi { dst = rd; src = rs; imm = imm12 n }) ]
  else li tmp n @ [ insn (Insn.Add { dst = rd; lhs = rs; rhs = tmp }) ]

let vtype sew lmul =
  {
    Vtype.sew;
    lmul;
    tail = Vtype.Policy.Agnostic;
    mask = Vtype.Policy.Agnostic;
  }

let e16m2 = vtype Vtype.Sew.E16 Vtype.Lmul.M2
let e16m4 = vtype Vtype.Sew.E16 Vtype.Lmul.M4
let e16m8 = vtype Vtype.Sew.E16 Vtype.Lmul.M8
let e32m2 = vtype Vtype.Sew.E32 Vtype.Lmul.M2
let e32m4 = vtype Vtype.Sew.E32 Vtype.Lmul.M4
let e32m8 = vtype Vtype.Sew.E32 Vtype.Lmul.M8

(* [vl] elements under [vt], the count through [tmp] *)
let vcfg ~tmp vt vl =
  li tmp vl @ [ insn (Insn.Vsetvli { avl = Vtype.Avl.Reg tmp; vtype = vt }) ]

(* [vl] elements under [vt], the count in [r] *)
let vcfg_reg r vt =
  [ insn (Insn.Vsetvli { avl = Vtype.Avl.Reg r; vtype = vt }) ]

let tile = 16

module Geometry = Zvt_sdpa_geometry.Geometry

let plan = Zvt_sdpa_geometry.plan

let panel_bytes t =
  match Geometry.make t with
  | exception Refused why -> Error why
  | exception Invalid_argument why -> Error why
  | g -> Ok (Geometry.total g)

let block_queries t =
  match panel_bytes t with Ok _ -> Ok bq | Error why -> Error why

(* Frame offsets, bytes from [sp]: the saved registers, then the loop state. *)
module Frame = struct
  let ra = 0
  let saved k = 4 * (k + 1) (* s0 .. s11 *)
  let q = 52
  let k = 56
  let v = 60
  let mask = 64
  let o = 68
  let batch = 72
  let head = 76
  let qleft = 80
  let kleft = 84
  let qptr = 88
  let optr = 92
  let mptr = 96
  let kptr = 100
  let vptr = 104
  let mkptr = 108
  let vq = 112
  let vk = 116
  let size = 128
end

let lw rd off = insn (Insn.Lw { dst = rd; base = Gpr.sp; offset = imm12 off })
let sw rs off = insn (Insn.Sw { src = rs; base = Gpr.sp; offset = imm12 off })
let la dst target = insn (Insn.La { dst; target })
let addi dst src n = insn (Insn.Addi { dst; src; imm = imm12 n })

let items ?mutation ?dtcm ?(skip = []) (t : A.t) =
  let skipped p l = if List.mem p skip then [] else l in
  let g = Geometry.make t in
  let d = g.Geometry.d and bk = g.Geometry.bk and depth = g.Geometry.depth in
  let name = A.symbol t in
  let label stem = Label.named (name ^ "_" ^ stem) in
  let n_labels = ref 0 in
  let fresh stem =
    incr n_labels;
    Label.named (Printf.sprintf "%s_%s%d" name stem !n_labels)
  in
  let int64 x =
    if Int64.compare x 0L < 0 || Int64.compare x 0x3FFF_FFFFL > 0 then
      refuse "a stride past 30 bits"
    else Int64.to_int x
  in
  let bytes2 x = 2 * int64 x and bytes4 x = 4 * int64 x in
  let tile_l = label "tile"
  and oacc_l = label "oacc"
  and qp_l = label "qpanel"
  and kp_l = label "kpanel"
  and stage_l = label "stage"
  and arrays_l = label "arrays" in
  (* the five arrays are consecutive, one vector each *)
  let arr k = (arrays_l, k * bq * 4) in
  let m_arr = arr 0 and mn_arr = arr 1 and l_arr = arr 2 and a_arr = arr 3 in
  let s_arr = arr 4 and ms_arr = arr 5 in
  let at (lbl, off) dst =
    la dst lbl :: (if off = 0 then [] else add_from ~tmp:Gpr.t6 dst dst off)
  in
  (* [count] trips of [body] with the counter in [r] *)
  let loop r body =
    let l = fresh "l" in
    [ Item.label l ]
    @ body
    @ [ addi r r (-1); insn (Insn.Bnez { src = r; target = l }) ]
  in
  (* zero [size] bytes from the address in [dst] (advanced) *)
  let zero_bytes ~dst size =
    let full = size / 128 and rest = size mod 128 in
    vcfg ~tmp:Gpr.t3 e32m8 32
    @ [ insn (Insn.Vmv_v_i { dst = vreg 8; imm = Imm.Simm5.zero }) ]
    @ (if full = 0 then []
       else
         li Gpr.t2 full
         @ loop Gpr.t2
             [
               insn (Insn.Vse { eew = Vtype.Sew.E32; src = vreg 8; base = dst });
               addi dst dst 128;
             ])
    @
    if rest = 0 then []
    else
      vcfg ~tmp:Gpr.t3 e32m8 (rest / 4)
      @ [ insn (Insn.Vse { eew = Vtype.Sew.E32; src = vreg 8; base = dst }) ]
  in
  (* a row of [d] bfloat16 elements in vectors of at most 64, the most one
     register group holds at e16m8 on a 128-bit vector unit *)
  let row_chunks =
    let rec go left = if left <= 0 then [] else min 64 left :: go (left - 64) in
    go d
  in
  (* [cnt] trips of [row i vl] over the chunks of a row, each under its own
     [e16m8] configuration; one chunk sets it once, before the loop, followed
     by [once] *)
  let rows_loop ?(once = []) cnt row =
    match row_chunks with
    | [ vl ] -> vcfg ~tmp:Gpr.t3 e16m8 vl @ once @ loop cnt (row 0 vl)
    | _ ->
        loop cnt
          (List.concat
             (List.mapi
                (fun i vl -> vcfg ~tmp:Gpr.t3 e16m8 vl @ row i vl)
                row_chunks))
  in
  (* [rows] (in [cnt], clobbered) rows of [d] elements from [t0], [src_stride]
     bytes apart, to [t1], dense *)
  let copy_rows ~cnt ~src_stride =
    let last = List.length row_chunks - 1 in
    rows_loop cnt (fun i vl ->
        [
          insn (Insn.Vle { eew = Vtype.Sew.E16; dst = vreg 8; base = Gpr.t0 });
          insn (Insn.Vse { eew = Vtype.Sew.E16; src = vreg 8; base = Gpr.t1 });
        ]
        @ add_from ~tmp:Gpr.t4 Gpr.t0 Gpr.t0
            (if i = last then src_stride - (2 * (d - vl)) else 2 * vl)
        @ [ addi Gpr.t1 Gpr.t1 (2 * vl) ])
  in
  (* [count] k rows of a panel, as the packed contraction packs them: each a
     vector of the active [vl] over the tile's indices, [stride] bytes apart
     (in [stride_reg]) from [t0], [row_bytes] apart in the panel at [t1]. Two k
     steps share a word: one strided load of words and two narrowing shifts. *)
  let strided_rows ~stride_reg ~row_bytes count =
    let v8 = vreg 8 and v4 = vreg 4 and v0 = vreg 0 in
    let store r =
      insn (Insn.Vse { eew = Vtype.Sew.E16; src = r; base = Gpr.t1 })
    in
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
        store v0;
      ]
      @ add_from ~tmp:Gpr.t4 Gpr.t1 Gpr.t1 row_bytes
      @ [ store v4 ]
      @ add_from ~tmp:Gpr.t4 Gpr.t1 Gpr.t1 row_bytes
      @ add_from ~tmp:Gpr.t4 Gpr.t0 Gpr.t0 4
    in
    let single =
      [
        insn
          (Insn.Vlse
             {
               eew = Vtype.Sew.E16;
               dst = v0;
               base = Gpr.t0;
               stride = stride_reg;
             });
        store v0;
      ]
      @ add_from ~tmp:Gpr.t4 Gpr.t0 Gpr.t0 2
      @ add_from ~tmp:Gpr.t4 Gpr.t1 Gpr.t1 row_bytes
    in
    let rep n body = if n <= 0 then [] else li Gpr.t2 n @ loop Gpr.t2 body in
    rep (count / 2) pair @ rep (count mod 2) single
  in
  (* the block product, the registers set as the packed function sets them *)
  let compute ~prefix ~rows ~depth ~n ~c_row ?a_pitch ?b_pitch ?b_pass
      ~accumulate ~a3 ~a7 ~a4 () =
    let p = plan ~rows ~depth ~n in
    let blocks =
      ok
        (P.Tile_gemm.block_items ~prefix ?a_pitch ?b_pitch ?b_pass ~c_row
           ~accumulate p)
    in
    a3 @ a7 @ a4 @ blocks
  in
  (* ---- the fragments ---------------------------------------------------- *)
  let prologue =
    [
      insn (Insn.Addi { dst = Gpr.sp; src = Gpr.sp; imm = imm12 (-Frame.size) });
    ]
    @ [ sw Gpr.ra Frame.ra ]
    @ List.mapi
        (fun k s -> sw s (Frame.saved k))
        Gpr.[ s0; s1; s2; s3; s4; s5; s6; s7; s8; s9; s10; s11 ]
    @ [
        sw Gpr.a0 Frame.q;
        sw Gpr.a1 Frame.k;
        sw Gpr.a2 Frame.v;
        sw Gpr.a3 Frame.o;
        sw Gpr.a4 Frame.mask;
      ]
  in
  let epilogue =
    [ lw Gpr.ra Frame.ra ]
    @ List.mapi
        (fun k s -> lw s (Frame.saved k))
        Gpr.[ s0; s1; s2; s3; s4; s5; s6; s7; s8; s9; s10; s11 ]
    @ [
        insn (Insn.Addi { dst = Gpr.sp; src = Gpr.sp; imm = imm12 Frame.size });
        insn (Insn.Li { dst = Gpr.a0; imm = Imm.Li.of_int 0 });
        insn Insn.Ret;
      ]
  in
  (* [vq] / [vk] for the block in progress: the full size, or the tail on the
     last block; [left] is the frame slot of the blocks left, this one included *)
  let set_valid ~left ~slot ~full ~tail =
    let full_l = fresh "full" and done_l = fresh "done" in
    [
      lw Gpr.t0 left;
      addi Gpr.t0 Gpr.t0 (-1);
      insn (Insn.Bnez { src = Gpr.t0; target = full_l });
    ]
    @ li Gpr.t1 tail
    @ [ sw Gpr.t1 slot; insn (Insn.J { target = done_l }); Item.label full_l ]
    @ li Gpr.t1 full
    @ [ sw Gpr.t1 slot; Item.label done_l ]
  in
  (* Q's rows [vq] from the tensor into the staging buffer, then the panel zeroed
     and packed with a run-time [vl] *)
  let stage_q =
    [ lw Gpr.t0 Frame.qptr ]
    @ at (stage_l, 0) Gpr.t1
    @ [ lw Gpr.t5 Frame.vq ]
    @ copy_rows ~cnt:Gpr.t5 ~src_stride:(bytes2 t.A.q.A.Strides.row)
  in
  let pack_q =
    at (qp_l, 0) Gpr.t1
    @ zero_bytes ~dst:Gpr.t1 g.Geometry.q_panel
    @ [ lw Gpr.t3 Frame.vq ]
    @ vcfg_reg Gpr.t3 e16m4
    @ li Gpr.s10 (2 * d)
    @ at (stage_l, 0) Gpr.t0
    @ at (qp_l, 0) Gpr.t1
    @ strided_rows ~stride_reg:Gpr.s10 ~row_bytes:32 d
  in
  let init_state =
    vcfg ~tmp:Gpr.t3 e32m4 bq @ li Gpr.t3 0xFF800000
    @ [ insn (Insn.Vmv_v_x { dst = vreg 4; src = Gpr.t3 }) ]
    @ at m_arr Gpr.t0
    @ [
        insn (Insn.Vse { eew = Vtype.Sew.E32; src = vreg 4; base = Gpr.t0 });
        insn (Insn.Vmv_v_i { dst = vreg 4; imm = Imm.Simm5.zero });
      ]
    @ at l_arr Gpr.t0
    @ [ insn (Insn.Vse { eew = Vtype.Sew.E32; src = vreg 4; base = Gpr.t0 }) ]
    @ at s_arr Gpr.t0
    @ [ insn (Insn.Vse { eew = Vtype.Sew.E32; src = vreg 4; base = Gpr.t0 }) ]
    @ at (oacc_l, 0) Gpr.t1
    @ zero_bytes ~dst:Gpr.t1 g.Geometry.oacc_bytes
  in
  (* K's rows [vk] staged, the panel zeroed on a short block, packed, and the
     scores block computed into the tile *)
  let scores =
    [ lw Gpr.t0 Frame.kptr ]
    @ at (stage_l, 0) Gpr.t1
    @ [ lw Gpr.t5 Frame.vk ]
    @ copy_rows ~cnt:Gpr.t5 ~src_stride:(bytes2 t.A.k.A.Strides.row)
    @ (let skip = fresh "full" in
       [
         lw Gpr.t0 Frame.vk;
         addi Gpr.t0 Gpr.t0 (-bk);
         insn (Insn.Beqz { src = Gpr.t0; target = skip });
       ]
       @ at (kp_l, 0) Gpr.t1
       @ zero_bytes ~dst:Gpr.t1 g.Geometry.k_panel
       @ [ Item.label skip ])
    @ [ lw Gpr.t3 Frame.vk ]
    @ vcfg_reg Gpr.t3 e16m4
    @ li Gpr.s9 (2 * d)
    @ at (stage_l, 0) Gpr.t0
    @ at (kp_l, 0) Gpr.t1
    @ strided_rows ~stride_reg:Gpr.s9
        ~row_bytes:(if bk > tile then 64 else 32)
        d
    @ compute ~prefix:(name ^ "_sc") ~rows:bk ~depth ~n:bq ~c_row:(4 * bq)
        ~accumulate:false
        ~a3:(at (kp_l, 0) Gpr.a3)
        ~a7:(at (qp_l, 0) Gpr.a7)
        ~a4:(at (tile_l, 0) Gpr.a4)
        ()
  in
  let scale_bits =
    Int32.to_int (Int32.bits_of_float t.A.scale) land 0xFFFF_FFFF
  in
  let has_mask = t.A.mask <> None in
  let mrow, mcol =
    match t.A.mask with
    | Some m -> (bytes4 m.A.Strides.row, bytes4 m.A.Strides.col)
    | None -> (0, 0)
  in
  (* pass 1: z = RN(RN(S * scale) + mask) over the tile's [vk] rows, in place,
     the maximum folded into v8 *)
  let pass1 =
    at m_arr Gpr.t0 @ vcfg ~tmp:Gpr.t3 e32m4 bq
    @ [ insn (Insn.Vle { eew = Vtype.Sew.E32; dst = vreg 8; base = Gpr.t0 }) ]
    @ li Gpr.t3 scale_bits
    @ [ insn (Insn.Vmv_v_x { dst = vreg 12; src = Gpr.t3 }) ]
    @ at (tile_l, 0) Gpr.s1
    @ [ lw Gpr.s4 Frame.vk ]
    @ (if has_mask then
         [ lw Gpr.s5 Frame.mkptr; lw Gpr.s7 Frame.vq ]
         @ li Gpr.s6 mrow @ li Gpr.s8 bq
       else [])
    @ loop Gpr.s4
        ([
           insn (Insn.Vle { eew = Vtype.Sew.E32; dst = vreg 4; base = Gpr.s1 });
           insn
             (Insn.Vv
                { op = Insn.Vv_op.Vfmul; dst = vreg 4; a = vreg 4; b = vreg 12 });
         ]
        @ (if has_mask then
             vcfg_reg Gpr.s7 e32m4
             @ [
                 insn
                   (Insn.Vlse
                      {
                        eew = Vtype.Sew.E32;
                        dst = vreg 16;
                        base = Gpr.s5;
                        stride = Gpr.s6;
                      });
               ]
             @ vcfg_reg Gpr.s8 e32m4
             @ [
                 insn
                   (Insn.Vv
                      {
                        op = Insn.Vv_op.Vfadd;
                        dst = vreg 4;
                        a = vreg 4;
                        b = vreg 16;
                      });
               ]
             @ add_from ~tmp:Gpr.t4 Gpr.s5 Gpr.s5 mcol
           else [])
        @ [
            insn (Insn.Vse { eew = Vtype.Sew.E32; src = vreg 4; base = Gpr.s1 });
            insn
              (Insn.Valu
                 {
                   op = Insn.Alu_op.Vfmax;
                   dst = vreg 8;
                   a = vreg 8;
                   b = vreg 4;
                 });
            addi Gpr.s1 Gpr.s1 (4 * bq);
          ])
  in
  let module O = Vexpr.Ops in
  let neg_inf = O.const Float.neg_infinity in
  let groups = List.init 7 (fun i -> vreg (4 * (i + 1))) in
  let without regs =
    List.filter (fun g -> not (List.exists (Vreg.equal g) regs)) groups
  in
  let load32 r lbl_off =
    at lbl_off Gpr.t0
    @ [ insn (Insn.Vle { eew = Vtype.Sew.E32; dst = r; base = Gpr.t0 }) ]
  and store32 r lbl_off =
    at lbl_off Gpr.t0
    @ [ insn (Insn.Vse { eew = Vtype.Sew.E32; src = r; base = Gpr.t0 }) ]
  in
  (* [expr] over the inputs in v4 and v8 (consumed), its result stored *)
  let evaluate expr dest =
    match
      Vexpr.emit ~lmul:4 ~scratch:Gpr.t0
        ~free:(without [ vreg 4; vreg 8 ])
        ~inputs:[| (vreg 4, `Consumed); (vreg 8, `Consumed) |]
        expr
    with
    | Error e -> refuse e
    | Ok plan -> plan.Vexpr.Plan.items @ store32 plan.Vexpr.Plan.result dest
  in
  let m_in = Vexpr.input 0 and mn_in = Vexpr.input 1 in
  (* exp (v4 - v8), both consumed, emitted once and called from the two places
     that need it; its result is in [exp_r] *)
  let exp_l = label "exp" in
  let exp_plan =
    match
      Vexpr.emit ~lmul:4 ~scratch:Gpr.t0
        ~free:(without [ vreg 4; vreg 8 ])
        ~inputs:[| (vreg 4, `Consumed); (vreg 8, `Consumed) |]
        (M.exp (O.sub (Vexpr.input 0) (Vexpr.input 1)))
    with
    | Error e -> refuse e
    | Ok plan -> plan
  in
  let exp_r = exp_plan.Vexpr.Plan.result in
  let exp_routine =
    [ Item.label exp_l ] @ exp_plan.Vexpr.Plan.items @ [ insn Insn.Ret ]
  in
  let call_exp = insn (Insn.Call { target = exp_l }) in
  (* a = exp (m - m'), and 1 where m' = -inf (then m = -inf too): both operands
     are 0 there. ms = m', or 0 where m' = -inf, so that pass 2's z - ms is
     -inf for a masked score and exp gives +0 as the select on z = -inf does.
     v8 holds m' on entry. *)
  let a_step =
    let m_eff = O.select_eq mn_in neg_inf (O.const 0.) m_in in
    vcfg ~tmp:Gpr.t3 e32m4 bq
    @ store32 (vreg 8) mn_arr
    @ load32 (vreg 4) mn_arr
    @ evaluate
        (O.select_eq (Vexpr.input 0) neg_inf (O.const 0.) (Vexpr.input 0))
        ms_arr
    (* m_eff into a's array, then a = exp (m_eff - ms), ms being mn_eff *)
    @ load32 (vreg 4) m_arr
    @ load32 (vreg 8) mn_arr
    @ evaluate m_eff a_arr
    @ load32 (vreg 4) a_arr
    @ load32 (vreg 8) ms_arr
    @ [ call_exp ] @ store32 exp_r a_arr
  in
  (* m := m' once the block is done *)
  let commit_max =
    vcfg ~tmp:Gpr.t3 e32m4 bq @ at mn_arr Gpr.t0
    @ [ insn (Insn.Vle { eew = Vtype.Sew.E32; dst = vreg 4; base = Gpr.t0 }) ]
    @ at m_arr Gpr.t0
    @ [ insn (Insn.Vse { eew = Vtype.Sew.E32; src = vreg 4; base = Gpr.t0 }) ]
  in
  (* O_acc rows [0, vq) scaled by a (broadcast); chunks of at most 32 *)
  let chunks =
    let rec go off acc =
      if off >= d then List.rev acc
      else go (off + 32) ((off, min 32 (d - off)) :: acc)
    in
    go 0 []
  in
  let rescale =
    if mutation = Some Mutation.Rescale_omitted then []
    else
      at (oacc_l, 0) Gpr.s1
      @ at a_arr Gpr.s2
      @ [ lw Gpr.s4 Frame.vq ]
      @ loop Gpr.s4
          (List.concat_map
             (fun (off, vl) ->
               vcfg ~tmp:Gpr.t3 e32m8 vl
               @ [
                   insn
                     (Insn.Vlse
                        {
                          eew = Vtype.Sew.E32;
                          dst = vreg 8;
                          base = Gpr.s2;
                          stride = Gpr.zero;
                        });
                 ]
               @ add_from ~tmp:Gpr.t4 Gpr.t0 Gpr.s1 (4 * off)
               @ [
                   insn
                     (Insn.Vle
                        { eew = Vtype.Sew.E32; dst = vreg 16; base = Gpr.t0 });
                   insn
                     (Insn.Vv
                        {
                          op = Insn.Vv_op.Vfmul;
                          dst = vreg 16;
                          a = vreg 16;
                          b = vreg 8;
                        });
                   insn
                     (Insn.Vse
                        { eew = Vtype.Sew.E32; src = vreg 16; base = Gpr.t0 });
                 ])
             chunks
          @ add_from ~tmp:Gpr.t4 Gpr.s1 Gpr.s1 (4 * d)
          @ [ addi Gpr.s2 Gpr.s2 4 ])
  in
  (* pass 2 over the tile's [vk] rows: p = exp (z - m') (+0 where z = -inf),
     s += p, and p rounded to bfloat16 into the P^T rows *)
  let pass2 =
    let r = exp_r in
    let t1, t2, t3 =
      match without [ r ] with
      | a :: b :: c :: _ -> (a, b, c)
      | _ -> refuse "no registers left for the rounding"
    in
    let m_reg = if mutation = Some Mutation.Max_stale then m_arr else ms_arr in
    at (tile_l, 0) Gpr.s1
    @ at (tile_l, 0) Gpr.s2
    @ at m_reg Gpr.s9 @ at s_arr Gpr.s10 @ li Gpr.s8 bq
    @ [ lw Gpr.s4 Frame.vk ]
    @ loop Gpr.s4
        (vcfg_reg Gpr.s8 e32m4
        @ [
            insn (Insn.Vle { eew = Vtype.Sew.E32; dst = vreg 4; base = Gpr.s1 });
            insn (Insn.Vle { eew = Vtype.Sew.E32; dst = vreg 8; base = Gpr.s9 });
          ]
        @ [ call_exp ]
        @ [
            insn (Insn.Vle { eew = Vtype.Sew.E32; dst = t3; base = Gpr.s10 });
            insn (Insn.Vv { op = Insn.Vv_op.Vfadd; dst = t3; a = t3; b = r });
            insn (Insn.Vse { eew = Vtype.Sew.E32; src = t3; base = Gpr.s10 });
            (* the rounding to bfloat16: x + 0x7FFF + bit 16 of x, the high half *)
            insn
              (Insn.Vsrl_vi { dst = t1; src = r; shamt = Imm.Shamt.of_int 16 });
            insn (Insn.Vand_vi { dst = t1; src = t1; imm = Imm.Simm5.of_int 1 });
          ]
        @ li Gpr.t0 0x7FFF
        @ [
            insn (Insn.Vmv_v_x { dst = t2; src = Gpr.t0 });
            insn (Insn.Vv { op = Insn.Vv_op.Vadd; dst = t2; a = r; b = t2 });
            insn (Insn.Vv { op = Insn.Vv_op.Vadd; dst = t2; a = t2; b = t1 });
          ]
        @ vcfg_reg Gpr.s8 e16m2
        @ [
            insn
              (Insn.Vnsrl_wi { dst = t1; src = t2; shamt = Imm.Shamt.of_int 16 });
            insn (Insn.Vse { eew = Vtype.Sew.E16; src = t1; base = Gpr.s2 });
            addi Gpr.s1 Gpr.s1 (4 * bq);
            addi Gpr.s2 Gpr.s2 (2 * bq);
          ])
  in
  (* the P^T rows past [vk] are +0 (the loop left [s2] at row [vk]) *)
  let zero_p_tail =
    let skip = fresh "full" in
    [
      lw Gpr.t0 Frame.vk;
      addi Gpr.t0 Gpr.t0 (-bk);
      insn (Insn.Beqz { src = Gpr.t0; target = skip });
    ]
    @ [ lw Gpr.t2 Frame.vk ]
    @ li Gpr.t0 bk
    @ [ insn (Insn.Sub { dst = Gpr.t2; lhs = Gpr.t0; rhs = Gpr.t2 }) ]
    @ vcfg ~tmp:Gpr.t3 e32m2 (bq / 2)
    @ [ insn (Insn.Vmv_v_i { dst = vreg 4; imm = Imm.Simm5.zero }) ]
    @ loop Gpr.t2
        [
          insn (Insn.Vse { eew = Vtype.Sew.E32; src = vreg 4; base = Gpr.s2 });
          addi Gpr.s2 Gpr.s2 (2 * bq);
        ]
    @ [ Item.label skip ]
  in
  (* l = RN(RN(l * a) + s); s := 0 *)
  let l_update =
    vcfg ~tmp:Gpr.t3 e32m4 bq @ at l_arr Gpr.t0
    @ [ insn (Insn.Vle { eew = Vtype.Sew.E32; dst = vreg 4; base = Gpr.t0 }) ]
    @ at a_arr Gpr.t0
    @ [
        insn (Insn.Vle { eew = Vtype.Sew.E32; dst = vreg 8; base = Gpr.t0 });
        insn
          (Insn.Vv
             { op = Insn.Vv_op.Vfmul; dst = vreg 4; a = vreg 4; b = vreg 8 });
      ]
    @ at s_arr Gpr.t0
    @ [
        insn (Insn.Vle { eew = Vtype.Sew.E32; dst = vreg 12; base = Gpr.t0 });
        insn
          (Insn.Vv
             { op = Insn.Vv_op.Vfadd; dst = vreg 4; a = vreg 4; b = vreg 12 });
      ]
    @ at l_arr Gpr.t0
    @ [
        insn (Insn.Vse { eew = Vtype.Sew.E32; src = vreg 4; base = Gpr.t0 });
        insn (Insn.Vmv_v_i { dst = vreg 12; imm = Imm.Simm5.zero });
      ]
    @ at s_arr Gpr.t0
    @ [ insn (Insn.Vse { eew = Vtype.Sew.E32; src = vreg 12; base = Gpr.t0 }) ]
  in
  (* V's rows [vk] staged with zero rows after them, and the context block
     accumulating into O_acc *)
  let context =
    let staged =
      [ lw Gpr.t0 Frame.vptr ]
      @ at (stage_l, 0) Gpr.t1
      @ [ lw Gpr.t5 Frame.vk ]
      @ copy_rows ~cnt:Gpr.t5 ~src_stride:(bytes2 t.A.v.A.Strides.row)
      @
      if mutation = Some Mutation.V_tail_unzeroed then []
      else
        let skip = fresh "full" in
        [
          lw Gpr.t0 Frame.vk;
          addi Gpr.t0 Gpr.t0 (-bk);
          insn (Insn.Beqz { src = Gpr.t0; target = skip });
        ]
        @ [ lw Gpr.t2 Frame.vk ]
        @ li Gpr.t0 bk
        @ [ insn (Insn.Sub { dst = Gpr.t2; lhs = Gpr.t0; rhs = Gpr.t2 }) ]
        @ (let zero =
             insn (Insn.Vmv_v_i { dst = vreg 8; imm = Imm.Simm5.zero })
           in
           rows_loop ~once:[ zero ] Gpr.t2 (fun _ vl ->
               (if List.length row_chunks = 1 then [] else [ zero ])
               @ [
                   insn
                     (Insn.Vse
                        { eew = Vtype.Sew.E16; src = vreg 8; base = Gpr.t1 });
                   addi Gpr.t1 Gpr.t1 (2 * vl);
                 ]))
        @ [ Item.label skip ]
    in
    (* the defect: a V already in the DTCM is read where it is, with neither
       the zero rows after its last key nor readable bytes after it *)
    let prepare, b_source =
      match (mutation, dtcm) with
      | Some Mutation.V_in_place_in_dtcm, Some (base, bytes) ->
          let bits =
            let rec go b n = if n lsr (b + 1) = 0 then b else go (b + 1) n in
            go 0 bytes
          in
          let staged_l = fresh "staged" and joined = fresh "joined" in
          ( li Gpr.t1 base
            @ [
                lw Gpr.t0 Frame.vptr;
                insn (Insn.Sub { dst = Gpr.t1; lhs = Gpr.t0; rhs = Gpr.t1 });
                insn
                  (Insn.Srli
                     {
                       dst = Gpr.t1;
                       src = Gpr.t1;
                       shamt = Imm.Shamt.of_int bits;
                     });
                insn (Insn.Bnez { src = Gpr.t1; target = staged_l });
                insn (Insn.Mv { dst = Gpr.s11; src = Gpr.t0 });
                insn (Insn.J { target = joined });
                Item.label staged_l;
              ]
            @ staged
            @ at (stage_l, 0) Gpr.s11
            @ [ Item.label joined ],
            [ insn (Insn.Mv { dst = Gpr.a7; src = Gpr.s11 }) ] )
      | _ -> (staged, at (stage_l, 0) Gpr.a7)
    in
    prepare
    @ compute ~prefix:(name ^ "_cx") ~rows:bq ~depth:bk ~n:d ~c_row:(4 * d)
        ~a_pitch:(2 * bq) ~b_pitch:(2 * d) ~b_pass:64
        ~accumulate:(mutation <> Some Mutation.Seed_ignored)
        ~a3:(at (tile_l, 0) Gpr.a3)
        ~a7:b_source
        ~a4:(at (oacc_l, 0) Gpr.a4)
        ()
  in
  let advance_k =
    [ lw Gpr.t0 Frame.kptr ]
    @ add_from ~tmp:Gpr.t4 Gpr.t0 Gpr.t0 (bk * bytes2 t.A.k.A.Strides.row)
    @ [ sw Gpr.t0 Frame.kptr; lw Gpr.t0 Frame.vptr ]
    @ (if mutation = Some Mutation.V_not_advanced then []
       else add_from ~tmp:Gpr.t4 Gpr.t0 Gpr.t0 (bk * bytes2 t.A.v.A.Strides.row))
    @ [ sw Gpr.t0 Frame.vptr ]
    @
    if has_mask then
      [ lw Gpr.t0 Frame.mkptr ]
      @ add_from ~tmp:Gpr.t4 Gpr.t0 Gpr.t0 (bk * mcol)
      @ [ sw Gpr.t0 Frame.mkptr ]
    else []
  in
  (* O = O_acc / l, +0 for a row whose l is +0; rows [0, vq) to the output *)
  let normalize =
    at (oacc_l, 0) Gpr.s1
    @ at l_arr Gpr.s2
    @ [ lw Gpr.s3 Frame.optr; lw Gpr.s4 Frame.vq ]
    @
    let zero_row = fresh "zero" and next = fresh "next" in
    loop Gpr.s4
      ([
         insn (Insn.Lw { dst = Gpr.t0; base = Gpr.s2; offset = imm12 0 });
         insn (Insn.Beqz { src = Gpr.t0; target = zero_row });
       ]
      @ List.concat_map
          (fun (off, vl) ->
            vcfg ~tmp:Gpr.t3 e32m8 vl
            @ [
                insn
                  (Insn.Vlse
                     {
                       eew = Vtype.Sew.E32;
                       dst = vreg 8;
                       base = Gpr.s2;
                       stride = Gpr.zero;
                     });
              ]
            @ add_from ~tmp:Gpr.t4 Gpr.t0 Gpr.s1 (4 * off)
            @ add_from ~tmp:Gpr.t4 Gpr.t1 Gpr.s3 (4 * off)
            @ [
                insn
                  (Insn.Vle
                     { eew = Vtype.Sew.E32; dst = vreg 16; base = Gpr.t0 });
                insn
                  (Insn.Valu
                     {
                       op = Insn.Alu_op.Vfdiv;
                       dst = vreg 16;
                       a = vreg 16;
                       b = vreg 8;
                     });
                insn
                  (Insn.Vse
                     { eew = Vtype.Sew.E32; src = vreg 16; base = Gpr.t1 });
              ])
          chunks
      @ [ insn (Insn.J { target = next }); Item.label zero_row ]
      @ List.concat_map
          (fun (off, vl) ->
            vcfg ~tmp:Gpr.t3 e32m8 vl
            @ [ insn (Insn.Vmv_v_i { dst = vreg 16; imm = Imm.Simm5.zero }) ]
            @ add_from ~tmp:Gpr.t4 Gpr.t1 Gpr.s3 (4 * off)
            @ [
                insn
                  (Insn.Vse
                     { eew = Vtype.Sew.E32; src = vreg 16; base = Gpr.t1 });
              ])
          chunks
      @ [ Item.label next ]
      @ add_from ~tmp:Gpr.t4 Gpr.s1 Gpr.s1 (4 * d)
      @ [ addi Gpr.s2 Gpr.s2 4 ]
      @ add_from ~tmp:Gpr.t4 Gpr.s3 Gpr.s3 (bytes4 t.A.o.A.Strides.row))
  in
  (* ---- the loop nest ---------------------------------------------------- *)
  let k_loop =
    let l = fresh "kblock" in
    let exit_l = fresh "kexit" in
    li Gpr.t0 g.Geometry.k_blocks
    @ [
        sw Gpr.t0 Frame.kleft;
        lw Gpr.t0 Frame.k;
        sw Gpr.t0 Frame.kptr;
        lw Gpr.t0 Frame.v;
        sw Gpr.t0 Frame.vptr;
        lw Gpr.t0 Frame.mptr;
        sw Gpr.t0 Frame.mkptr;
        Item.label l;
      ]
    @ set_valid ~left:Frame.kleft ~slot:Frame.vk ~full:bk
        ~tail:g.Geometry.k_tail
    @ skipped Phase.Scores scores
    @ pass1 @ a_step @ rescale @ skipped Phase.Exp pass2 @ zero_p_tail
    @ l_update @ commit_max
    @ skipped Phase.Context context
    @ advance_k
    @ [
        lw Gpr.t0 Frame.kleft;
        addi Gpr.t0 Gpr.t0 (-1);
        sw Gpr.t0 Frame.kleft;
        insn (Insn.Beqz { src = Gpr.t0; target = exit_l });
        insn (Insn.J { target = l });
        Item.label exit_l;
      ]
  in
  let q_loop =
    let l = fresh "qblock" in
    let exit_l = fresh "qexit" in
    li Gpr.t0 g.Geometry.q_blocks
    @ [
        sw Gpr.t0 Frame.qleft;
        lw Gpr.t0 Frame.q;
        sw Gpr.t0 Frame.qptr;
        lw Gpr.t0 Frame.o;
        sw Gpr.t0 Frame.optr;
        lw Gpr.t0 Frame.mask;
        sw Gpr.t0 Frame.mptr;
        Item.label l;
      ]
    @ set_valid ~left:Frame.qleft ~slot:Frame.vq ~full:bq
        ~tail:g.Geometry.q_tail
    @ stage_q @ pack_q @ init_state @ k_loop
    @ skipped Phase.Normalize normalize
    @ [ lw Gpr.t0 Frame.qptr ]
    @ add_from ~tmp:Gpr.t4 Gpr.t0 Gpr.t0 (bq * bytes2 t.A.q.A.Strides.row)
    @ [ sw Gpr.t0 Frame.qptr; lw Gpr.t0 Frame.optr ]
    @ add_from ~tmp:Gpr.t4 Gpr.t0 Gpr.t0 (bq * bytes4 t.A.o.A.Strides.row)
    @ [ sw Gpr.t0 Frame.optr ]
    @ (if has_mask then
         [ lw Gpr.t0 Frame.mptr ]
         @ add_from ~tmp:Gpr.t4 Gpr.t0 Gpr.t0 (bq * mrow)
         @ [ sw Gpr.t0 Frame.mptr ]
       else [])
    @ [
        lw Gpr.t0 Frame.qleft;
        addi Gpr.t0 Gpr.t0 (-1);
        sw Gpr.t0 Frame.qleft;
        insn (Insn.Beqz { src = Gpr.t0; target = exit_l });
        insn (Insn.J { target = l });
        Item.label exit_l;
      ]
  in
  (* batch and head loops advance the five base pointers *)
  let bump slot n =
    [ lw Gpr.t0 slot ]
    @ add_from ~tmp:Gpr.t4 Gpr.t0 Gpr.t0 n
    @ [ sw Gpr.t0 slot ]
  in
  let strides = (t.A.q, t.A.k, t.A.v, t.A.o) in
  let head_step =
    let q, k, v, o = strides in
    bump Frame.q (bytes2 q.A.Strides.head)
    @ bump Frame.k (bytes2 k.A.Strides.head)
    @ bump Frame.v (bytes2 v.A.Strides.head)
    @ bump Frame.o (bytes4 o.A.Strides.head)
    @
    match t.A.mask with
    | Some m -> bump Frame.mask (bytes4 m.A.Strides.head)
    | None -> []
  in
  let batch_step =
    let heads = int64 t.A.heads in
    let q, k, v, o = strides in
    let over unit (s : A.Strides.t) =
      (unit * int64 s.A.Strides.batch) - (heads * unit * int64 s.A.Strides.head)
    in
    bump Frame.q (over 2 q)
    @ bump Frame.k (over 2 k)
    @ bump Frame.v (over 2 v)
    @ bump Frame.o (over 4 o)
    @ match t.A.mask with Some m -> bump Frame.mask (over 4 m) | None -> []
  in
  let batch_l = fresh "batch" and head_l = fresh "head" in
  let exit_batch_l = fresh "bexit" and exit_head_l = fresh "hexit" in
  let body =
    li Gpr.t0 (int64 t.A.batch)
    @ [ sw Gpr.t0 Frame.batch; Item.label batch_l ]
    @ li Gpr.t0 (int64 t.A.heads)
    @ [ sw Gpr.t0 Frame.head; Item.label head_l ]
    @ q_loop @ head_step
    @ [
        lw Gpr.t0 Frame.head;
        addi Gpr.t0 Gpr.t0 (-1);
        sw Gpr.t0 Frame.head;
        insn (Insn.Beqz { src = Gpr.t0; target = exit_head_l });
        insn (Insn.J { target = head_l });
        Item.label exit_head_l;
      ]
    @ batch_step
    @ [
        lw Gpr.t0 Frame.batch;
        addi Gpr.t0 Gpr.t0 (-1);
        sw Gpr.t0 Frame.batch;
        insn (Insn.Beqz { src = Gpr.t0; target = exit_batch_l });
        insn (Insn.J { target = batch_l });
        Item.label exit_batch_l;
      ]
  in
  ignore dtcm;
  let entry = Label.named name in
  [ Item.text; Item.align 4; Item.export ~name entry; Item.label entry ]
  @ prologue @ body @ epilogue @ exp_routine @ [ Item.bss ]
  @ List.concat_map
      (fun (l, size) ->
        [ Item.align 64; Item.label l; Item.zero (Geometry.align64 size) ])
      [
        (tile_l, g.Geometry.tile_bytes);
        (oacc_l, g.Geometry.oacc_bytes);
        (qp_l, g.Geometry.q_panel);
        (kp_l, g.Geometry.k_panel);
        (stage_l, g.Geometry.stage_bytes);
        (arrays_l, g.Geometry.arrays_bytes);
      ]

let module_ ?mutation ?budget ?dtcm ?skip t =
  match items ?mutation ?dtcm ?skip t with
  | exception Refused why -> Error why
  | exception Invalid_argument why -> Error why
  | items -> (
      let g = Geometry.make t in
      match budget with
      | Some b when Geometry.total g > b ->
          Error
            (Printf.sprintf "the panels need %d bytes, %d available"
               (Geometry.total g) b)
      | _ -> Kernel_module.of_items ~name:(A.symbol t) items)
