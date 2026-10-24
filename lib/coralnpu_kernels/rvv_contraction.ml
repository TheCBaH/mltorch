module A = Machine_rivet_riscv32.Rv32_asm
module C = Contraction

module Mutation = struct
  type t = A_row_stride | No_wait | Tail_vl
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

(* Bytes an operand spans from its first element, rounded up to 16: what a
   staging copy moves. *)
let span ~size ~batch ~rows ~k (o : C.Operand.t) =
  let last count stride = Int64.mul (Int64.max 0L (Int64.pred count)) stride in
  let elements =
    Int64.succ
      (List.fold_left Int64.add 0L
         [
           last batch o.C.Operand.batch;
           last rows o.C.Operand.outer;
           last k o.C.Operand.k;
         ])
  in
  let b = bytes ~size elements in
  (b + 15) land lnot 15

(* Which operands to copy into the function's own DTCM panels first. *)
module Stage = struct
  type t = { a : bool; b : bool }
end

(* The staging the subsystem measurements support for operands in SRAM: A
   always (a little cheaper than its scalar loads from SRAM), B only when its
   columns are strided, where each [vlse32.v] element is its own SRAM access;
   a contiguous B row streams from SRAM faster than the DMA copies it. *)
let in_sram (d : C.t) =
  { Stage.a = true; b = Int64.compare d.C.b.C.Operand.outer 1L <> 0 }

(* The staged operands are first copied into the function's own DTCM panels
   by one DMA chain the core starts and waits for, a 16-byte-beat descriptor
   an operand it writes itself from its arguments; the panels then stand for
   the operands. A source not 16-byte aligned, or a chain that ends in error,
   leaves every operand where it is. *)
let staging ?mutation (d : C.t) ~name (st : Stage.t) =
  let k = Int64.max 1L d.C.k in
  let span = span ~size:(C.Element.bytes d.C.format) in
  let operands =
    List.filter_map Fun.id
      [
        (if st.Stage.a then
           Some ("a0", span ~batch:d.C.batch ~rows:d.C.m ~k d.C.a, name ^ "_a")
         else None);
        (if st.Stage.b then
           Some ("a1", span ~batch:d.C.batch ~rows:d.C.n ~k d.C.b, name ^ "_b")
         else None);
      ]
  in
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
             @ A.load_imm "t2" (flags size)
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

let items ?mutation ?(stage = { Stage.a = false; b = false }) (d : C.t) () =
  (match d.C.seed with
  | C.Seed.Zero -> ()
  | C.Seed.Tensor -> refuse "a contraction seeded from its output");
  let n = Int64.to_int d.C.n and m = Int64.to_int d.C.m in
  let k = Int64.to_int d.C.k and batch = Int64.to_int d.C.batch in
  let fused = d.C.numerics = C.Numerics.Fused_sequential in
  let bf16 = d.C.format = C.Element.Bf16 in
  let size = C.Element.bytes d.C.format in
  let rows = if fused then 3 else 2 in
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
    let ptrs = List.filteri (fun j _ -> j < nrows) [ "t0"; "t1"; "t2" ] in
    let zero = List.map (fun v -> i "vmv.v.i" [ r v; A.imm 0 ]) accs in
    let start =
      List.concat (List.mapi (fun j p -> add_from p "a5" (j * a_outer)) ptrs)
      @ [ i "mv" [ r "a7"; r "a3" ] ]
    in
    (* a bfloat16 is the high half of its binary32: A's scalar shifted into
       place, B's row loaded at EEW 16 into the high half of v0's group (where
       the widening may overlap its source) and widened into it *)
    let step =
      List.concat_map
        (fun (f, p) ->
          if bf16 then
            [
              i "lhu" [ r "s4"; A.mem p 0 ];
              i "slli" [ r "s4"; r "s4"; A.imm 16 ];
              i "fmv.w.x" [ r f; r "s4" ];
            ]
          else [ i "flw" [ r f; A.mem p 0 ] ])
        (List.combine fs ptrs)
      @ (if bf16 then
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
      @ List.concat_map (fun p -> add_const p a_k) ptrs
      @ add_const "a7" b_k
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
    zero @ (if k > 0 then start @ loop k ~counter:"t3" step else []) @ stores
  in
  (* every row group of one tile; then a5 and a6 are past the last row *)
  let tile_body vl =
    set_vl vl
    @ [ i "mv" [ r "a5"; r "a0" ]; i "mv" [ r "a6"; r "a4" ] ]
    @ loop (m / rows) ~counter:"t4"
        (group rows
        @ add_const "a5" (rows * a_outer)
        @ add_const "a6" (rows * c_row))
    @ if m mod rows > 0 then group (m mod rows) else []
  in
  let full = n / tile and rest = n mod tile in
  let tiles =
    [ i "mv" [ r "a3"; r "a1" ]; i "mv" [ r "a4"; r "a2" ] ]
    @ loop full ~counter:"t5"
        (tile_body tile
        @ add_const "a3" (tile * b_outer)
        @ add_const "a4" (tile * c_col))
    @
    if rest > 0 then
      tile_body (if mutation = Some Mutation.Tail_vl then rest - 1 else rest)
    else []
  in
  let per_batch =
    tiles
    @ add_const "a0" (bytes ~size d.C.a.C.Operand.batch)
    @ add_const "a1" (bytes ~size d.C.b.C.Operand.batch)
    @ add_const "a2" (bytes d.C.c.C.Output.batch)
  in
  let name = symbol d in
  let stage_code, stage_data = staging ?mutation d ~name stage in
  (* s4 widens A's bfloat16 scalars *)
  let saved = [ "s0"; "s1"; "s2"; "s3" ] @ if bf16 then [ "s4" ] else [] in
  let frame = if bf16 then 32 else 16 in
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
  @ stage_data

let module_ ?mutation ?stage d =
  match items ?mutation ?stage d () with
  | items -> A.module_ (symbol d) (fun () -> items)
  | exception Refused why -> Error why
