(* Matrix multiplies far larger than DTCM, A, B and C in SRAM, fed by the DMA:
   C[M x W] = A[M x K] B[K x W], through RVV or the Zvt matrix engine.

   B is resident: a prologue chain copies it into DTCM once. A streams by
   blocks of rows, each block into one half of a two-block ring in DTCM, and
   C's rows are staged in a second ring and copied out. Step i computes block
   i while one chain writes out block i - 1's rows and fetches block i + 1:
   the chain is started before the block and waited for after it, so the DMA
   costs only what the block does not hide. Each matrix crosses the bus once,
   which is the least any order moves. The host builds every chain for the
   layout. A block is the peak kernels' own: two rows of 64 lanes for RVV
   ({!Peak_kernels.Gemm}), two rows of 16 x 16 tiles for Zvt
   ({!Peak_zvt.Tile_gemm}), packed as they pack it.

   The table holds, in order: A, C and B in SRAM; B's copy, the A ring and the
   C ring in DTCM; the steps (the block count, the prologue chain, one chain
   per block, the epilogue chain; 0 for none); the chains (SRAM); and the
   timings (DTCM): the cycles spent waiting for the DMA after the blocks and
   the epilogue, and the cycles of the prologue, for B and A's first block. *)

open Peak_kernels
module D = Coral_dma
module G = Gemm
module T = Peak_zvt.Tile_gemm

module Slot = struct
  let a = 0
  let c = 1
  let b = 2
  let b_dtcm = 3
  let a_ring = 4
  let c_ring = 5
  let steps = 6
  let chains = 7
  let timings = 8
end

(* What computes a block. Alphabetical. *)
module Engine = struct
  type t =
    | Rvv_i8  (** INT8 x INT8 -> INT16, [vwmacc.vx]: two rows, W = 64 *)
    | Zvt_f32  (** FP32 [vtfmm]: 32 rows, W a multiple of 32 *)
    | Zvt_i8  (** signed INT8 -> INT32 [vtmms]: 32 rows, W a multiple of 32 *)

  let name = function
    | Rvv_i8 -> "rvv i8"
    | Zvt_f32 -> "zvt f32"
    | Zvt_i8 -> "zvt i8"

  let rows = function Rvv_i8 -> 2 | Zvt_f32 | Zvt_i8 -> 32
  let matrix = function Rvv_i8 -> false | Zvt_f32 | Zvt_i8 -> true
end

(* Who copies C's rows out to SRAM. Alphabetical. *)
module Out = struct
  type t =
    | Core  (** the block's own stores, straight to SRAM: no C ring *)
    | Dma  (** staged in the C ring, copied by the next step's chain *)

  let name = function Core -> "c-by-core" | Dma -> "c-by-dma"
end

(* Who copies B into DTCM in the prologue. Alphabetical. *)
module Prologue = struct
  type t =
    | Dma  (** the prologue chain, with the core waiting *)
    | Shared
        (** the chain takes a quarter of B, the core's RVV copy the rest beside
            it: the core copies 2.4 bytes a cycle, the DMA 0.8 while the core
            writes DTCM *)

  let name = function Dma -> "b-by-dma" | Shared -> "b-shared"
end

(* The bytes an RVV copy moves a trip: two e8 m8 groups. *)
let core_trip = 256

type t = {
  blocks : int;
  depth : int;
  engine : Engine.t;
  out : Out.t;
  prologue : Prologue.t;
  width : int;
}

let rows t = Engine.rows t.engine

let shape t =
  let src, dst =
    match t.engine with
    | Rvv_i8 -> (Elem.I8, Elem.I16)
    | Zvt_f32 -> (F32, F32)
    | Zvt_i8 -> (I8, I32)
  in
  {
    G.rows = rows t;
    width = t.width;
    depth = t.depth;
    blocks = t.blocks;
    src;
    dst;
    sa = 1;
    sb = 1;
  }

(* The Zvt plan, over all M rows at once as the peak kernel takes them. *)
let plan t =
  let s = shape t in
  let s = { s with rows = s.rows * s.blocks; blocks = 1 } in
  match t.engine with
  | Rvv_i8 -> invalid_arg "Dma_gemm.plan: not a Zvt engine"
  | Zvt_f32 -> T.plan Vtfmm (Peak_zvt.Tile.shape Vtfmm) s
  | Zvt_i8 -> T.plan Vtmms { (Peak_zvt.Tile.shape Vtmms) with vtype = 0x1C0 } s

let check t =
  match t.engine with
  | Rvv_i8 ->
      if t.width <> 64 || t.depth mod 4 <> 0 then
        invalid_arg "Dma_gemm: RVV INT8 takes W = 64 and K a multiple of 4"
  | Zvt_f32 | Zvt_i8 ->
      if t.width mod 32 <> 0 then
        invalid_arg "Dma_gemm: Zvt takes W a multiple of 32"

let c_row t = t.width * Elem.bytes (shape t).dst
let c_block t = rows t * c_row t

let a_block t =
  match t.engine with
  | Rvv_i8 -> rows t * t.depth
  | Zvt_f32 | Zvt_i8 -> T.a_block (plan t)

(* A and B as the block packs them, without padding. *)
let packed t =
  match t.engine with
  | Rvv_i8 -> (
      match G.inputs (shape t) with
      | [ a; b ] ->
          ( String.sub a 0 (t.blocks * a_block t),
            String.sub b 0 (t.depth * t.width) )
      | _ -> assert false)
  | Zvt_f32 | Zvt_i8 -> (
      match T.inputs (plan t) with
      | [ a; b ] ->
          ( String.sub a 0 (t.blocks * a_block t),
            String.sub b 0 (String.length b - 128) )
      | _ -> assert false)

(* How far past A's block and B's end the loops read ahead. *)
let a_pad t = match t.engine with Rvv_i8 -> 64 | Zvt_f32 | Zvt_i8 -> 128

let b_pad t =
  match t.engine with Rvv_i8 -> 2 * t.width | Zvt_f32 | Zvt_i8 -> 128

let slots t =
  check t;
  let _, b = packed t in
  let b_size = String.length b in
  let steps = 4 * (t.blocks + 4) in
  let chains = D.Descriptor.size * ((2 * t.blocks) + 2) in
  L.Placement.
    [
      L.Slot.make ~home:Ram ~align:64L In
        (Int64.of_int ((t.blocks * a_block t) + a_pad t));
      L.Slot.make ~home:Ram ~align:64L Out (Int64.of_int (t.blocks * c_block t));
      L.Slot.make ~home:Ram ~align:64L In (Int64.of_int b_size);
      L.Slot.make ~home:Dtcm ~align:64L Scratch
        (Int64.of_int (b_size + b_pad t));
      L.Slot.make ~home:Dtcm ~align:64L Scratch
        (Int64.of_int ((2 * a_block t) + a_pad t));
      L.Slot.make ~home:Dtcm ~align:64L Scratch
        (Int64.of_int (match t.out with Dma -> 2 * c_block t | Core -> 16));
      L.Slot.make ~home:Dtcm ~align:16L In (Int64.of_int steps);
      L.Slot.make ~home:Ram ~align:32L In (Int64.of_int chains);
      L.Slot.make ~home:Dtcm ~align:16L Out 16L;
    ]

let expected t =
  G.product { (shape t) with rows = rows t * t.blocks; blocks = 1 }

(* The longest piece of a descriptor: whole beats, under 16 MiB. *)
let pieces ~src ~dst n =
  let max = 0x80_0000 in
  List.init
    ((n + max - 1) / max)
    (fun k ->
      let off = Int64.of_int (k * max) in
      D.Descriptor.copy ~src:(Int64.add src off) ~dst:(Int64.add dst off)
        (Int64.of_int (min max (n - (k * max)))))

(* The chains for a layout: the prologue (B, then block 0), one per block
   (block i - 1's rows out, block i + 1 in), the epilogue (the last rows out). *)
let inputs p t (l : L.t) =
  let address k = (List.nth l.slots k).L.Placed_slot.address in
  let off a n = Int64.add a (Int64.of_int n) in
  let a = address Slot.a and c = address Slot.c and b = address Slot.b in
  let ring = address Slot.a_ring and c_ring = address Slot.c_ring in
  let a_bytes, b_bytes = packed t in
  let a_in i =
    D.Descriptor.copy
      ~src:(off a (i * a_block t))
      ~dst:(off ring (i mod 2 * a_block t))
      (Int64.of_int (a_block t))
  in
  let c_out i =
    D.Descriptor.copy
      ~src:(off c_ring (i mod 2 * c_block t))
      ~dst:(off c (i * c_block t))
      (Int64.of_int (c_block t))
  in
  (* The core's share of B, from its start, in whole 256-byte trips. *)
  let b_size = String.length b_bytes in
  let by_core =
    match t.prologue with
    | Dma -> 0
    | Shared -> b_size * 3 / 4 / core_trip * core_trip
  in
  let pre =
    (if by_core < b_size then
       pieces ~src:(off b by_core)
         ~dst:(off (address Slot.b_dtcm) by_core)
         (b_size - by_core)
     else [])
    @ [ a_in 0 ]
  in
  let by_dma = match t.out with Dma -> true | Core -> false in
  let per i =
    (if i > 0 && by_dma then [ c_out (i - 1) ] else [])
    @ if i + 1 < t.blocks then [ a_in (i + 1) ] else []
  in
  let post = if by_dma then [ c_out (t.blocks - 1) ] else [] in
  let chains = (pre :: List.init t.blocks per) @ [ post ] in
  let open Err.Syntax in
  let* placed, _ =
    Err.List.fold_left
      (fun (acc, at) chain ->
        match chain with
        | [] -> Err.return ((0L, "") :: acc, at)
        | _ ->
            let* bytes = D.chain p ~at chain in
            Err.return ((at, bytes) :: acc, off at (String.length bytes)))
      ([], address Slot.chains)
      chains
  in
  let placed = List.rev placed in
  Err.return
    [
      a_bytes;
      "";
      b_bytes;
      "";
      "";
      "";
      Dma_copy.words
        (match List.map (fun (a, _) -> Int64.to_int a) placed with
        | pre :: rest -> t.blocks :: pre :: by_core :: rest
        | [] -> assert false);
      String.concat "" (List.map snd placed);
    ]

(* {1 The kernel} *)

let andi = Dma_copy.andi
let beqz = Dma_copy.beqz
let fence = Dma_copy.fence
let mcycle = Dma_copy.mcycle
let sub = Dma_copy.sub
let saved = Gpr.[ s0; s1; s2; s3; s4; s5; s6; s7; s8; s9 ]

(* The chain at [s9] started, unless it is 0. Every earlier store is made
   visible first: the chain may copy out rows just stored. *)
let start ~skip =
  [
    beqz Gpr.s9 skip;
    fence;
    sw Gpr.s9 Gpr.s3 (D.Register.offset Desc_addr);
    li Gpr.t2 D.ctrl_start;
    sw Gpr.t2 Gpr.s3 (D.Register.offset Ctrl);
    Item.label skip;
  ]

(* The chain at [s9] waited for, its cycles added to [into]. *)
let wait ~into ~skip ~poll =
  [
    beqz Gpr.s9 skip;
    mcycle Gpr.t2;
    Item.label poll;
    lw Gpr.t4 Gpr.s3 (D.Register.offset Status);
    andi Gpr.t4 Gpr.t4 D.status_done;
    beqz Gpr.t4 poll;
    mcycle Gpr.t3;
    sub Gpr.t3 Gpr.t3 Gpr.t2;
    add into into Gpr.t3;
    Item.label skip;
  ]

(* The core's share of B, [bytes_at] bytes into the steps ([s2]), copied by
   RVV from the table's slot at [src] to [dst], 256 bytes a trip, while the
   prologue chain runs. Writes [a1], [a5], [t6]. *)
let core_copy ~src ~dst ~bytes_at =
  let open Gpr in
  let trip = Label.named "b_trip" and none = Label.named "b_none" in
  let e8 = Vtype.Sew.E8 in
  [
    lw t6 s2 bytes_at;
    beqz t6 none;
    lw a1 a0 src;
    mv a5 dst;
    srli t6 t6 8;
    vset_item 8 8;
    Item.label trip;
    vle e8 Vreg.v0 a1;
    addi a1 a1 128;
    vle e8 Vreg.v8 a1;
    addi a1 a1 128;
    vse e8 Vreg.v0 a5;
    addi a5 a5 128;
    vse e8 Vreg.v8 a5;
    addi a5 a5 128;
    addi t6 t6 (-1);
    bnez t6 trip;
    Item.label none;
  ]

(* The peak kernel's INT8 block: A from [a1], B from [s1], C's two rows to
   [a6]. Writes [a0]-[a6], [t0], [t3]-[t5]. *)
let rvv_i8_block t =
  let open Gpr in
  let v0 = Vreg.v0 and v4 = Vreg.v4 and v16 = Vreg.v16 and v24 = Vreg.v24 in
  let mac acc scalar src = Item.insn (Vwmacc_vx { acc; scalar; src }) in
  let b_row = t.width and rows2 = 2 * t.width in
  [
    li t0 t.width;
    mv a2 s1;
    addi a3 a2 b_row;
    li a0 (t.depth / 4);
    vset_item ~avl:t0 8 4;
    vle E8 v0 a2;
    lw a4 a1 0;
    vset_item ~avl:t0 16 8;
    vmv_v_i v16;
    vmv_v_i v24;
    vset_item ~avl:t0 8 4;
  ]
  @ body_items
      [
        lw a5 a1 4;
        vle E8 v4 a3;
        srli t3 a4 8;
        mac v16 a4 v0;
        mac v24 t3 v0;
        addi a2 a2 rows2;
        vle E8 v0 a2;
        srli t4 a4 16;
        srli t5 a4 24;
        mac v16 t4 v4;
        mac v24 t5 v4;
        addi a3 a3 rows2;
        vle E8 v4 a3;
        srli t3 a5 8;
        mac v16 a5 v0;
        mac v24 t3 v0;
        addi a2 a2 rows2;
        addi a1 a1 8;
        lw a4 a1 0;
        vle E8 v0 a2;
        srli t4 a5 16;
        srli t5 a5 24;
        mac v16 t4 v4;
        mac v24 t5 v4;
        addi a3 a3 rows2;
        addi a0 a0 (-1);
        bnez a0 body_label;
      ]
  @ [
      vset_item ~avl:t0 16 8;
      vse E16 v16 a6;
      addi a6 a6 (c_row t);
      vse E16 v24 a6;
    ]

let items t () =
  let open Err.Syntax in
  let open Gpr in
  check t;
  let block = Label.named "block" and even = Label.named "even" in
  let label n = Label.named n in
  (* The registers a block reads its halves of the rings from. *)
  let a_at, c_at =
    match t.engine with Rvv_i8 -> (a1, a6) | Zvt_f32 | Zvt_i8 -> (a3, a4)
  in
  let* dma = Insn.load_imm s3 (Int64.to_int D.base) in
  let* to_odd_a = G.add_imm a_at (a_block t) in
  let* to_odd_c = G.add_imm c_at (c_block t) in
  let* next_c = G.add_imm s5 (c_block t) in
  let* body =
    match t.engine with
    | Rvv_i8 -> Ok (rvv_i8_block t)
    | Zvt_f32 | Zvt_i8 -> T.block_items (plan t)
  in
  Ok
    (start_items
    @ [ addi sp sp (-48) ]
    @ List.mapi (fun k r -> sw r sp (4 * k)) saved
    @ [
        lw s1 a0 (4 * Slot.b_dtcm);
        lw s2 a0 (4 * Slot.steps);
        lw s4 a0 (4 * Slot.a_ring);
        lw s5 a0 (4 * match t.out with Dma -> Slot.c_ring | Core -> Slot.c);
        lw s8 a0 (4 * Slot.timings);
        mv a7 s1;
      ]
    @ List.map Item.insn dma
    @ [ li s7 0; lw s9 s2 4 ]
    @ start ~skip:(label "pre_started")
    @ core_copy ~src:(4 * Slot.b) ~dst:s1 ~bytes_at:8
    @ wait ~into:s7 ~skip:(label "pre_waited") ~poll:(label "pre_poll")
    @ [
        sw s7 s8 4;
        li s7 0;
        lw s0 s2 0;
        addi s2 s2 12;
        li s6 0;
        Item.label block;
        lw s9 s2 0;
        addi s2 s2 4;
      ]
    @ start ~skip:(label "started")
    @ [ mv a_at s4; mv c_at s5; beqz s6 even ]
    @ to_odd_a
    @ (match t.out with Dma -> to_odd_c | Core -> [])
    @ [ Item.label even ]
    @ body
    (* With the core's stores, C's cursor moves on a block in SRAM. *)
    @ (match t.out with Dma -> [] | Core -> next_c)
    @ wait ~into:s7 ~skip:(label "waited") ~poll:(label "poll")
    @ [ li t2 1; sub s6 t2 s6; addi s0 s0 (-1); bnez s0 block; lw s9 s2 0 ]
    @ start ~skip:(label "post_started")
    @ wait ~into:s7 ~skip:(label "post_waited") ~poll:(label "post_poll")
    @ [ sw s7 s8 0 ]
    @ List.mapi (fun k r -> lw r sp (4 * k)) saved
    @ [ addi sp sp 48; li a0 0; Item.insn Ret ])

let ops t = 2 * rows t * t.blocks * t.width * t.depth

let name t =
  Printf.sprintf "%s resident-b %s %s M=%d N=%d K=%d" (Engine.name t.engine)
    (Prologue.name t.prologue) (Out.name t.out)
    (rows t * t.blocks)
    t.width t.depth
