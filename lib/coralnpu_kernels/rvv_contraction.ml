module A = Machine_rivet_riscv32.Rv32_asm
module C = Contraction

module Mutation = struct
  type t =
    | A_row_stride
    | Aliased_ring
    | C_current_half
    | No_ring_wait
    | No_share
    | No_wait
    | Tail_vl
end

let symbol = C.symbol
let i = A.insn
let r = A.reg

exception Refused of string

let refuse why = raise (Refused why)

(* A byte offset of [n] elements of [size] bytes (binary32 by default) as an
   [int], refused past 31 bits. *)
let bytes ?(size = 4L) (n : int64) =
  let b = Int64.mul size n in
  if Int64.compare b 0x7FFF_FFFFL > 0 then refuse "an offset past 31 bits"
  else Int64.to_int b

let small n = n >= -2048 && n < 2048

(* [rd += n] for any 32-bit [n]: an [addi], or through [s3]. *)
let add_const rd n =
  if n = 0 then []
  else if small n then [ i "addi" [ r rd; r rd; A.imm n ] ]
  else A.load_imm "s3" n @ [ i "add" [ r rd; r rd; r "s3" ] ]

(* [rd = rs + n]. *)
let add_from rd rs n =
  if small n then [ i "addi" [ r rd; r rs; A.imm n ] ]
  else A.load_imm "s3" n @ [ i "add" [ r rd; r rs; r "s3" ] ]

let tile = 32

(* The subsystem DMA engine's registers ({!Coral_dma}): base, [CTRL], [STATUS]
   and [DESC_ADDR] offsets, start, done and error bits. Restated here, since
   this library stays clear of the simulator libraries. *)
let dma_base = 0x4005_0000
let dma_ctrl = 0x00
let dma_status = 0x04
let dma_desc_addr = 0x08
let dma_start = 3
let dma_done_or_error = 6

module T = Coral_transfer.Transfer_plan

(* The staged operands are first copied into the function's own DTCM panels
   by one DMA chain the core starts and waits for, a 16-byte-beat descriptor
   an operand it writes itself from its arguments; the panels then stand for
   the operands. A source not 16-byte aligned, or a chain that ends in error,
   leaves every operand where it is. *)
let staging ?mutation (d : C.t) ~name (st : T.t) =
  let operands =
    List.filter_map Fun.id
      [
        (match st.T.a with
        | T.A.Copied -> Some ("a0", T.span d `A, name ^ "_a")
        | T.A.In_place | T.A.Streamed _ -> None);
        (match st.T.b with
        | T.B.Copied | T.B.Shared -> Some ("a1", T.span d `B, name ^ "_b")
        | T.B.In_place -> None);
      ]
  in
  (* a shared B: the chain copies its head, the core's RVV copy the rest
     beside it, in 256-byte trips (two e8 m8 groups), three quarters of it *)
  let core_b =
    match st.T.b with
    | T.B.Shared -> T.span d `B * 3 / 4 / 256 * 256
    | T.B.Copied | T.B.In_place -> 0
  in
  let dma_size (src, size, _) = if src = "a1" then size - core_b else size in
  if operands = [] then ([], [])
  else (
    List.iter
      (fun (_, size, _) ->
        if size > 0xFF_FFF0 then refuse "an operand past one DMA transfer")
      operands;
    let la rd sym = i "la" [ r rd; A.sym sym ] in
    let flags len = len lor (4 lsl 24) in
    let skip = name ^ "_unstaged" and wait = name ^ "_wait" in
    let pd = name ^ "_d" in
    let n = List.length operands in
    let aligned =
      match operands with
      | [ (x, _, _) ] -> [ i "andi" [ r "t0"; r x; A.imm 15 ] ]
      | _ ->
          [
            i "or" [ r "t0"; r "a0"; r "a1" ];
            i "andi" [ r "t0"; r "t0"; A.imm 15 ];
          ]
    in
    let descriptors =
      List.concat
        (List.mapi
           (fun j (src, size, panel) ->
             let o = 32 * j in
             [
               i "sw" [ r src; A.mem "t0" o ];
               la "t1" panel;
               i "sw" [ r "t1"; A.mem "t0" (o + 4) ];
             ]
             @ A.load_imm "t2" (flags (dma_size (src, size, panel)))
             @ [ i "sw" [ r "t2"; A.mem "t0" (o + 8) ] ]
             @
             if j = n - 1 then [ i "sw" [ r "zero"; A.mem "t0" (o + 12) ] ]
             else
               [
                 i "addi" [ r "t3"; r "t0"; A.imm (o + 32) ];
                 i "sw" [ r "t3"; A.mem "t0" (o + 12) ];
               ])
           operands)
    in
    let code =
      aligned
      @ [ i "bnez" [ r "t0"; A.sym skip ]; la "t0" pd ]
      @ descriptors
      @ [ i "fence" [] ]
      @ A.load_imm "t4" dma_base
      @ [
          i "sw" [ r "t0"; A.mem "t4" dma_desc_addr ];
          i "li" [ r "t5"; A.imm dma_start ];
          i "sw" [ r "t5"; A.mem "t4" dma_ctrl ];
        ]
      @ (if core_b = 0 || mutation = Some Mutation.No_share then []
         else
           let l = name ^ "_share" and head = T.span d `B - core_b in
           add_from "t1" "a1" head
           @ [ la "t2" (name ^ "_b") ]
           @ add_const "t2" head
           @ A.load_imm "t3" (core_b / 256)
           @ [
               i "li" [ r "t5"; A.imm 128 ];
               i "vsetvli"
                 [
                   r "zero";
                   r "t5";
                   A.sym "e8";
                   A.sym "m8";
                   A.sym "ta";
                   A.sym "ma";
                 ];
               A.label l;
               i "vle8.v" [ r "v0"; A.mem "t1" 0 ];
               i "addi" [ r "t1"; r "t1"; A.imm 128 ];
               i "vle8.v" [ r "v8"; A.mem "t1" 0 ];
               i "addi" [ r "t1"; r "t1"; A.imm 128 ];
               i "vse8.v" [ r "v0"; A.mem "t2" 0 ];
               i "addi" [ r "t2"; r "t2"; A.imm 128 ];
               i "vse8.v" [ r "v8"; A.mem "t2" 0 ];
               i "addi" [ r "t2"; r "t2"; A.imm 128 ];
               i "addi" [ r "t3"; r "t3"; A.imm (-1) ];
               i "bnez" [ r "t3"; A.sym l ];
             ])
      @ (if mutation = Some Mutation.No_wait then []
         else
           [
             A.label wait;
             i "lw" [ r "t2"; A.mem "t4" dma_status ];
             i "andi" [ r "t3"; r "t2"; A.imm dma_done_or_error ];
             i "beqz" [ r "t3"; A.sym wait ];
             i "andi" [ r "t3"; r "t2"; A.imm 4 ];
             i "bnez" [ r "t3"; A.sym skip ];
           ])
      @ List.map (fun (src, _, panel) -> la src panel) operands
      @ [ A.label skip ]
    and data =
      [ A.bss ]
      @ List.concat_map
          (fun (_, size, panel) -> [ A.align 64; A.label panel; A.zero size ])
          operands
      @ [ A.align 32; A.label pd; A.zero (32 * n) ]
    in
    (code, data))

let items ?mutation ?plan (d : C.t) () =
  let stage =
    match plan with
    | None -> T.in_place
    | Some w ->
        if C.name (T.Witness.contraction w) <> C.name d then
          refuse "a transfer plan checked for another contraction";
        T.Witness.plan w
  in
  (match d.C.seed with
  | C.Seed.Zero -> ()
  | C.Seed.Tensor -> refuse "a contraction seeded from its output");
  if d.C.numerics = C.Numerics.Paired then
    refuse "the paired requirement: only the matrix engine rounds pairs";
  let n = Int64.to_int d.C.n and m = Int64.to_int d.C.m in
  let k = Int64.to_int d.C.k and batch = Int64.to_int d.C.batch in
  let fused = d.C.numerics = C.Numerics.Fused_sequential in
  let bf16 = d.C.format = C.Element.Bf16 in
  let int8 =
    match d.C.format with
    | C.Element.I8 { a; b } -> Some (a, b)
    | C.Element.Bf16 | C.Element.F32 -> None
  in
  let size = C.Element.bytes d.C.format in
  let rows = T.group_rows d in
  let accs = [ "v8"; "v16"; "v24" ] in
  let fregs = [ "fa0"; "fa1"; "fa2" ] in
  let a_outer =
    bytes ~size d.C.a.C.Operand.outer
    + if mutation = Some Mutation.A_row_stride then Int64.to_int size else 0
  and a_k = bytes ~size d.C.a.C.Operand.k in
  let b_outer = bytes ~size d.C.b.C.Operand.outer
  and b_k = bytes ~size d.C.b.C.Operand.k in
  let c_row = bytes d.C.c.C.Output.row and c_col = bytes d.C.c.C.Output.col in
  let labels = ref 0 in
  let fresh () =
    incr labels;
    Printf.sprintf ".Lc%d" !labels
  in
  let loop count ~counter body =
    if count <= 0 then []
    else if count = 1 then body
    else
      let l = fresh () in
      A.load_imm counter count
      @ [ A.label l ]
      @ body
      @ [
          i "addi" [ r counter; r counter; A.imm (-1) ];
          i "bnez" [ r counter; A.sym l ];
        ]
  in
  let set_vl vl =
    A.load_imm "s0" vl
    @ [
        i "vsetvli"
          [ r "zero"; r "s0"; A.sym "e32"; A.sym "m8"; A.sym "ta"; A.sym "ma" ];
      ]
  in
  (* [rows] rows of C from the group bases (a5: A, a6: C), under the tile's
     configuration *)
  let group nrows =
    let accs = List.filteri (fun j _ -> j < nrows) accs in
    let fs = List.filteri (fun j _ -> j < nrows) fregs in
    (* the rows' A values through one pointer and immediate offsets when the
       last row's fits a load's, else a pointer a row *)
    let shared = small ((nrows - 1) * a_outer) in
    let ptrs =
      if shared then [ "t0" ]
      else List.filteri (fun j _ -> j < nrows) [ "t0"; "t1"; "t2" ]
    in
    let at =
      List.init nrows (fun j ->
          if shared then ("t0", j * a_outer) else (List.nth ptrs j, 0))
    in
    (* an integer A value goes to an integer register: t1, t2 and s4, free
       when the rows share t0 *)
    if Option.is_some int8 && not shared then
      refuse "integer A rows too far apart for one pointer";
    let xs = List.filteri (fun j _ -> j < nrows) [ "t1"; "t2"; "s4" ] in
    let zero = List.map (fun v -> i "vmv.v.i" [ r v; A.imm 0 ]) accs in
    let start =
      List.concat (List.mapi (fun j p -> add_from p "a5" (j * a_outer)) ptrs)
      @ [ i "mv" [ r "a7"; r "a3" ] ]
    in
    (* a bfloat16 is the high half of its binary32: A's scalar shifted into
       place, B's row loaded at EEW 16 into the high half of v0's group (where
       the widening may overlap its source) and widened into it *)
    let int_step (sa, sb) =
      List.map2
        (fun x (p, o) ->
          i
            (match sa with C.Sign.Signed -> "lb" | C.Sign.Unsigned -> "lbu")
            [ r x; A.mem p o ])
        xs at
      (* B's row at EEW 8 into the top quarter of v0's group, widened into it *)
      @ (if b_outer = 1 then [ i "vle8.v" [ r "v6"; A.mem "a7" 0 ] ]
         else [ i "vlse8.v" [ r "v6"; A.mem "a7" 0; r "s1" ] ])
      @ [
          i
            (match sb with
            | C.Sign.Signed -> "vsext.vf4"
            | C.Sign.Unsigned -> "vzext.vf4")
            [ r "v0"; r "v6" ];
        ]
      @ List.map2 (fun v x -> i "vmacc.vx" [ r v; r x; r "v0" ]) accs xs
      @ List.concat_map (fun p -> add_const p a_k) ptrs
      @ add_const "a7" b_k
    in
    (* the rows' A values of the step [extra] bytes ahead into [fs] *)
    let load_a fs ~extra =
      List.concat_map
        (fun (f, (p, o)) ->
          if bf16 then
            [
              i "lhu" [ r "s4"; A.mem p (o + extra) ];
              i "slli" [ r "s4"; r "s4"; A.imm 16 ];
              i "fmv.w.x" [ r f; r "s4" ];
            ]
          else [ i "flw" [ r f; A.mem p (o + extra) ] ])
        (List.combine fs at)
    in
    (* B's row of the step, the multiply-adds with the A values in [fs], and
       B's pointer to the next step *)
    let compute fs =
      (if bf16 then
         (if b_outer = 2 then [ i "vle16.v" [ r "v4"; A.mem "a7" 0 ] ]
          else [ i "vlse16.v" [ r "v4"; A.mem "a7" 0; r "s1" ] ])
         @ [
             i "vzext.vf2" [ r "v0"; r "v4" ];
             i "vsll.vi" [ r "v0"; r "v0"; A.imm 16 ];
           ]
       else if b_outer = 4 then [ i "vle32.v" [ r "v0"; A.mem "a7" 0 ] ]
       else [ i "vlse32.v" [ r "v0"; A.mem "a7" 0; r "s1" ] ])
      @ List.concat
          (List.map2
             (fun v f ->
               if fused then [ i "vfmacc.vf" [ r v; r f; r "v0" ] ]
               else
                 [
                   i "vfmul.vf" [ r "v24"; r "v0"; r f ];
                   i "vfadd.vv" [ r v; r v; r "v24" ];
                 ])
             accs fs)
      @ add_const "a7" b_k
    in
    let advance n = List.concat_map (fun p -> add_const p (n * a_k)) ptrs in
    (* k steps a pair at a time, software pipelined as the peak kernels are:
       the next step's A values load into the second register set while the
       current step's multiply-adds run, A's pointers advance once a pair,
       and the last one or two steps are peeled so that no load reads past
       A's last step *)
    let fs' = List.filteri (fun j _ -> j < nrows) [ "fa3"; "fa4"; "fa5" ] in
    let pipelined =
      let last = List.fold_left (fun m (_, o) -> max m o) 0 at in
      small (last + (2 * a_k))
    in
    let steps () =
      match int8 with
      | Some signs -> loop k ~counter:"t3" (int_step signs)
      | None when not pipelined ->
          loop k ~counter:"t3" (load_a fs ~extra:0 @ compute fs @ advance 1)
      | None ->
          let pairs = (k - 1) / 2 in
          load_a fs ~extra:0
          @ loop pairs ~counter:"t3"
              (load_a fs' ~extra:a_k @ compute fs
              @ load_a fs ~extra:(2 * a_k)
              @ compute fs' @ advance 2)
          @
          if k - 1 - (2 * pairs) = 1 then
            load_a fs' ~extra:a_k @ compute fs @ compute fs'
          else compute fs
    in
    let stores =
      List.concat
        (List.mapi
           (fun j v ->
             add_from "t0" "a6" (j * c_row)
             @
             if c_col = 4 then [ i "vse32.v" [ r v; A.mem "t0" 0 ] ]
             else [ i "vsse32.v" [ r v; A.mem "t0" 0; r "s2" ] ])
           accs)
    in
    zero @ (if k > 0 then start @ steps () else []) @ stores
  in
  (* every row group of one tile; then a5 and a6 are past the last row *)
  let tile_body ~m vl =
    set_vl vl
    @ [ i "mv" [ r "a5"; r "a0" ]; i "mv" [ r "a6"; r "a4" ] ]
    @ loop (m / rows) ~counter:"t4"
        (group rows
        @ add_const "a5" (rows * a_outer)
        @ add_const "a6" (rows * c_row))
    @ if m mod rows > 0 then group (m mod rows) else []
  in
  let full = n / tile and rest = n mod tile in
  (* every tile of [m] rows from A at a0 into C at a2 *)
  let tiles ~m =
    [ i "mv" [ r "a3"; r "a1" ]; i "mv" [ r "a4"; r "a2" ] ]
    @ loop full ~counter:"t5"
        (tile_body ~m tile
        @ add_const "a3" (tile * b_outer)
        @ add_const "a4" (tile * c_col))
    @
    if rest > 0 then
      tile_body ~m (if mutation = Some Mutation.Tail_vl then rest - 1 else rest)
    else []
  in
  let name = symbol d in
  (* A streamed by blocks of [rb] rows through the ring halves: block j + 1's
     chain is started once block j's has arrived, before block j's product,
     into the half block j - 1 used; s5 counts the blocks left, s6 is the
     current half, s7 the next block's source, s8 the current block's C, s9
     and s10 the batch's A and C, s11 the other half *)
  let ring_data, streamed =
    match stage.T.a with
    | T.A.Copied | T.A.In_place -> ([], None)
    | T.A.Streamed { rows = rb } ->
        let la rd sym = i "la" [ r rd; A.sym sym ] in
        let h0 = name ^ "_h0" and h1 = name ^ "_h1" and rd = name ^ "_r" in
        let fail = name ^ "_dma_failed" in
        let blocks = (m + rb - 1) / rb in
        let last = m - ((blocks - 1) * rb) in
        let row = bytes ~size d.C.k in
        let flags len = len lor (4 lsl 24) in
        let block_bytes = rb * row
        and last_bytes = ((last * row) + 15) land lnot 15 in
        (* with C by the DMA a half holds the block's C after its A, and the
           chain started before block j's product also carries block j - 1's
           C out of the other half (a second descriptor, from the second
           block on); a last chain carries the last two blocks' out *)
        let c_dma = stage.T.out = T.Out.Dma in
        let c_block = rb * c_row in
        let half = block_bytes + if c_dma then c_block else 0 in
        let c_out ~at ~src ~dst ~bytes =
          (* descriptor [at] of the chain at t0: [bytes] of C from [src] to
             [dst] (registers), next 0 *)
          [
            i "sw" [ r src; A.mem "t0" at ];
            i "sw" [ r dst; A.mem "t0" (at + 4) ];
          ]
          @ A.load_imm "t2" (flags bytes)
          @ [
              i "sw" [ r "t2"; A.mem "t0" (at + 8) ];
              i "sw" [ r "zero"; A.mem "t0" (at + 12) ];
            ]
        in
        if block_bytes > 0xFF_FFF0 then
          refuse "a ring block past one DMA transfer";
        let go ~src ~dst ~len =
          [
            la "t0" rd;
            i "sw" [ r src; A.mem "t0" 0 ];
            i "sw" [ r dst; A.mem "t0" 4 ];
          ]
          @ len
          @ [
              i "sw" [ r "t1"; A.mem "t0" 8 ];
              i "sw" [ r "zero"; A.mem "t0" 12 ];
              i "fence" [];
            ]
          @ A.load_imm "t4" dma_base
          @ [
              i "sw" [ r "t0"; A.mem "t4" dma_desc_addr ];
              i "li" [ r "t5"; A.imm dma_start ];
              i "sw" [ r "t5"; A.mem "t4" dma_ctrl ];
            ]
        in
        let wait () =
          if mutation = Some Mutation.No_ring_wait then []
          else
            let l = fresh () in
            A.load_imm "t4" dma_base
            @ [
                A.label l;
                i "lw" [ r "t2"; A.mem "t4" dma_status ];
                i "andi" [ r "t3"; r "t2"; A.imm dma_done_or_error ];
                i "beqz" [ r "t3"; A.sym l ];
                i "andi" [ r "t3"; r "t2"; A.imm 4 ];
                i "bnez" [ r "t3"; A.sym fail ];
              ]
        in
        let first =
          A.load_imm "t1"
            (flags (if blocks = 1 then last_bytes else block_bytes))
        in
        let middle =
          if blocks = 1 then []
          else
            let l = fresh () and full_len = fresh () in
            A.load_imm "s5" (blocks - 1)
            @ [ A.label l ]
            @ wait ()
            @ (if mutation = Some Mutation.Aliased_ring then
                 [ i "mv" [ r "s11"; r "s6" ] ]
               else
                 [
                   la "t0" h0;
                   la "t1" h1;
                   i "add" [ r "t1"; r "t0"; r "t1" ];
                   i "sub" [ r "s11"; r "t1"; r "s6" ];
                 ])
            @ add_const "s7" block_bytes
            @ A.load_imm "t1" (flags block_bytes)
            @ [
                i "li" [ r "t3"; A.imm 1 ];
                i "bne" [ r "s5"; r "t3"; A.sym full_len ];
              ]
            @ A.load_imm "t1" (flags last_bytes)
            @ [ A.label full_len ]
            @ (if not c_dma then go ~src:"s7" ~dst:"s11" ~len:[]
               else
                 let no_prev = fresh () and started = fresh () in
                 [
                   la "t0" rd;
                   i "sw" [ r "s7"; A.mem "t0" 0 ];
                   i "sw" [ r "s11"; A.mem "t0" 4 ];
                   i "sw" [ r "t1"; A.mem "t0" 8 ];
                   i "li" [ r "t3"; A.imm (blocks - 1) ];
                   i "beq" [ r "s5"; r "t3"; A.sym no_prev ];
                   i "addi" [ r "t2"; r "t0"; A.imm 32 ];
                   i "sw" [ r "t2"; A.mem "t0" 12 ];
                 ]
                 @ add_from "t3"
                     (if mutation = Some Mutation.C_current_half then "s6"
                      else "s11")
                     block_bytes
                 @ add_from "t1" "s8" (-c_block)
                 @ c_out ~at:32 ~src:"t3" ~dst:"t1" ~bytes:c_block
                 @ [
                     i "j" [ A.sym started ];
                     A.label no_prev;
                     i "sw" [ r "zero"; A.mem "t0" 12 ];
                     A.label started;
                     i "fence" [];
                   ]
                 @ A.load_imm "t4" dma_base
                 @ [
                     i "sw" [ r "t0"; A.mem "t4" dma_desc_addr ];
                     i "li" [ r "t5"; A.imm dma_start ];
                     i "sw" [ r "t5"; A.mem "t4" dma_ctrl ];
                   ])
            @ [ i "mv" [ r "a0"; r "s6" ] ]
            @ (if c_dma then add_from "a2" "s6" block_bytes
               else [ i "mv" [ r "a2"; r "s8" ] ])
            @ tiles ~m:rb
            @ add_const "s8" (rb * c_row)
            @ [
                i "mv" [ r "s6"; r "s11" ];
                i "addi" [ r "s5"; r "s5"; A.imm (-1) ];
                i "bnez" [ r "s5"; A.sym l ];
              ]
        in
        let code =
          [
            i "andi" [ r "t0"; r "a0"; A.imm 15 ];
            i "bnez" [ r "t0"; A.sym fail ];
            i "mv" [ r "s9"; r "a0" ];
            i "mv" [ r "s10"; r "a2" ];
            i "mv" [ r "s7"; r "a0" ];
            i "mv" [ r "s8"; r "a2" ];
            la "s6" h0;
          ]
          @ go ~src:"s7" ~dst:"s6" ~len:first
          @ middle @ wait ()
          @ [ i "mv" [ r "a0"; r "s6" ] ]
          @ (if c_dma then add_from "a2" "s6" block_bytes
             else [ i "mv" [ r "a2"; r "s8" ] ])
          @ tiles ~m:last
          @ (if not c_dma then []
             else
               (* the last two blocks' C out: the other half's, then this one's *)
               [ la "t0" rd ]
               @ (if blocks = 1 then []
                  else
                    [
                      la "t1" h0; la "t2" h1; i "add" [ r "t2"; r "t1"; r "t2" ];
                    ]
                    @ [ i "sub" [ r "t3"; r "t2"; r "s6" ] ]
                    @ add_const "t3" block_bytes
                    @ add_from "t1" "s8" (-c_block)
                    @ c_out ~at:0 ~src:"t3" ~dst:"t1" ~bytes:c_block
                    @ [
                        i "addi" [ r "t2"; r "t0"; A.imm 32 ];
                        i "sw" [ r "t2"; A.mem "t0" 12 ];
                      ])
               @ add_from "t3" "s6" block_bytes
               @ c_out
                   ~at:(if blocks = 1 then 0 else 32)
                   ~src:"t3" ~dst:"s8" ~bytes:(last * c_row)
               @ [ i "fence" [] ]
               @ A.load_imm "t4" dma_base
               @ [
                   i "sw" [ r "t0"; A.mem "t4" dma_desc_addr ];
                   i "li" [ r "t5"; A.imm dma_start ];
                   i "sw" [ r "t5"; A.mem "t4" dma_ctrl ];
                 ]
               @ wait ())
          @ [ i "mv" [ r "a0"; r "s9" ]; i "mv" [ r "a2"; r "s10" ] ]
        in
        ( [
            A.bss;
            A.align 64;
            A.label h0;
            A.zero half;
            A.align 64;
            A.label h1;
            A.zero half;
            A.align 32;
            A.label rd;
            A.zero 64;
          ],
          Some (code, fail) )
  in
  let per_batch =
    (match streamed with Some (code, _) -> code | None -> tiles ~m)
    @ add_const "a0" (bytes ~size d.C.a.C.Operand.batch)
    @ add_const "a1" (bytes ~size d.C.b.C.Operand.batch)
    @ add_const "a2" (bytes d.C.c.C.Output.batch)
  in
  let stage_code, stage_data = staging ?mutation d ~name stage in
  (* s4 widens A's bfloat16 scalars; s5-s11 run the ring *)
  let saved =
    [ "s0"; "s1"; "s2"; "s3" ]
    @ (if bf16 || Option.is_some int8 then [ "s4" ] else [])
    @
    if Option.is_some streamed then
      [ "s5"; "s6"; "s7"; "s8"; "s9"; "s10"; "s11" ]
    else []
  in
  let frame = ((4 * List.length saved) + 15) land lnot 15 in
  [ A.text; A.align 4; A.global name; A.label name ]
  @ [ i "addi" [ r "sp"; r "sp"; A.imm (-frame) ] ]
  @ List.mapi (fun j s -> i "sw" [ r s; A.mem "sp" (4 * j) ]) saved
  @ A.load_imm "s1" b_outer @ A.load_imm "s2" c_col @ stage_code
  @ loop batch ~counter:"t6" per_batch
  @ List.mapi (fun j s -> i "lw" [ r s; A.mem "sp" (4 * j) ]) saved
  @ [
      i "addi" [ r "sp"; r "sp"; A.imm frame ];
      i "li" [ r "a0"; A.imm 0 ];
      i "ret" [];
    ]
  @ (match streamed with
    | Some (_, fail) -> [ A.label fail; i "unimp" [] ]
    | None -> [])
  @ stage_data @ ring_data

let module_ ?mutation ?plan d =
  match items ?mutation ?plan d () with
  | items -> A.module_ (symbol d) (fun () -> items)
  | exception Refused why -> Error why
