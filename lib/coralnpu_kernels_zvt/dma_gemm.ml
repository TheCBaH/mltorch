(* Matrix multiplies far larger than DTCM, A, B and C in SRAM, fed by the DMA:
   C[M x W] = A[M x K] B[K x W], through RVV or the Zvt matrix engine.

   One operand is resident: a prologue chain copies it into DTCM once. The
   other streams, a piece a step into one half of a two-piece ring in DTCM,
   and C's rows are staged in a second ring and copied out. Step i computes
   piece i while one chain writes out step i - 1's rows and fetches piece i +
   1: the chain is started before the step and waited for after it, so the
   DMA costs only what the step does not hide. Each matrix crosses the bus
   once, which is the least any order moves. The host builds every chain for
   the layout. A block is the peak kernels' own: two rows of 64 lanes for RVV
   ({!Peak_kernels.Gemm}), two rows of 16 x 16 tiles for Zvt
   ({!Peak_zvt.Tile_gemm}), packed as they pack it.

   With B resident, a piece is a block of A's rows and a step one block. With
   A resident, a piece is a panel of B's columns, packed contiguous, and a
   step every block of rows against it, C's rows a panel wide.

   The table holds, in order: the streamed operand, C and the resident one in
   SRAM; the resident's copy, the ring and the C ring in DTCM; the steps (the
   step count, the prologue chain, one chain per step, the epilogue chain; 0
   for none); the chains (SRAM); and the timings (DTCM): the cycles spent
   waiting for the DMA after the steps and the epilogue, and the cycles of the
   prologue, for the resident operand and the first piece. *)

open Coralnpu_kernels.Peak_kernels
module D = Coral_dma
module G = Gemm
module T = Peak_zvt.Tile_gemm

module Slot = struct
  let streamed = 0
  let c = 1
  let resident = 2
  let resident_dtcm = 3
  let ring = 4
  let c_ring = 5
  let steps = 6
  let chains = 7
  let timings = 8

  (* The geometry's own: split K's zeros, row-major A's packed block. *)
  let zeros = 9
  let packed = 9
end

(* 32-bit little-endian words *)
let words l =
  let b = Bytes.create (4 * List.length l) in
  List.iteri (fun k w -> Bytes.set_int32_le b (4 * k) (Int32.of_int w)) l;
  Bytes.to_string b

module Engine = Coral_transfer.Dma_geometry.Engine
module Out = Coral_transfer.Dma_geometry.Out
module Prologue = Coral_transfer.Dma_geometry.Prologue
module Geometry = Coral_transfer.Dma_geometry.Geometry

(* The bytes an RVV copy moves a trip: two e8 m8 groups. *)
let core_trip = 256

type t = Coral_transfer.Dma_geometry.t = {
  blocks : int;
  depth : int;
  engine : Engine.t;
  geometry : Geometry.t;
  out : Out.t;
  prologue : Prologue.t;
  width : int;
}

let rows t = Engine.rows t.engine

(* The columns a block computes. *)
let lanes t =
  match t.geometry with
  | Resident_a { panel } -> panel
  | Resident_b | Row_major_a | Split_k _ -> t.width

(* The depth a block reduces over. *)
let block_depth t =
  match t.geometry with
  | Split_k { chunk } -> chunk
  | Resident_a _ | Resident_b | Row_major_a -> t.depth

let chunks t = t.depth / block_depth t

let steps t =
  match t.geometry with
  | Resident_a { panel } -> t.width / panel
  | Resident_b | Row_major_a -> t.blocks
  | Split_k _ -> chunks t * t.blocks

let shape t =
  let src, dst =
    match t.engine with
    | Rvv_bf16 -> (Elem.Bf16, Elem.F32)
    | Rvv_f32 -> (F32, F32)
    | Rvv_i8 -> (I8, I16)
    | Zvt_bf16 -> (Bf16, F32)
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
  | Rvv_bf16 | Rvv_f32 | Rvv_i8 -> invalid_arg "Dma_gemm.plan: not Zvt"
  | Zvt_bf16 -> T.plan Vtfmm_alt (Peak_zvt.Tile.shape Vtfmm_alt) s
  | Zvt_f32 -> T.plan Vtfmm (Peak_zvt.Tile.shape Vtfmm) s
  | Zvt_i8 -> T.plan Vtmms { (Peak_zvt.Tile.shape Vtmms) with vtype = 0x1C0 } s

(* The plan of one block's columns and depth. *)
let block_plan t = plan { t with width = lanes t; depth = block_depth t }

let check t =
  (match t.engine with
  | Rvv_bf16 | Rvv_f32 ->
      if lanes t <> 32 || block_depth t mod 2 <> 0 then
        invalid_arg "Dma_gemm: RVV BF16 and FP32 take 32 lanes and K even"
  | Rvv_i8 ->
      if lanes t <> 64 || block_depth t mod 4 <> 0 then
        invalid_arg "Dma_gemm: RVV INT8 takes 64 lanes and K a multiple of 4"
  | Zvt_bf16 | Zvt_f32 | Zvt_i8 ->
      if lanes t mod 32 <> 0 then
        invalid_arg "Dma_gemm: Zvt takes lanes a multiple of 32");
  if t.width mod lanes t <> 0 then
    invalid_arg "Dma_gemm: W is not a multiple of the panel";
  match (t.out, t.geometry) with
  | Split n, (Resident_b | Row_major_a)
    when (not (Engine.matrix t.engine)) || n <= 0 || n >= rows t ->
      invalid_arg "Dma_gemm: a split of C is Zvt's, within a block"
  | Split _, (Resident_a _ | Split_k _) ->
      invalid_arg "Dma_gemm: a split of C is resident B's"
  | Core, Split_k _ ->
      invalid_arg "Dma_gemm: split K's partial sums go by the DMA"
  (* A block's partial sums go out in the chain after its step, and come back
     in the chain before its next: a step apart only with two blocks. *)
  | Dma, Split_k { chunk } when t.depth mod chunk <> 0 || t.blocks < 2 ->
      invalid_arg "Dma_gemm: split K takes chunks of K, and two blocks"
  | (Core | Dma | Split _), (Resident_a _ | Resident_b | Row_major_a | Split_k _)
    ->
      ()

(* Bytes from a row of B to the next in DTCM, and of C in SRAM. *)
let b_row t = lanes t * Elem.bytes (shape t).src
let c_row t = t.width * Elem.bytes (shape t).dst
let c_block t = rows t * c_row t

(* Bytes from a row of C to the next where the block stores it: a panel's
   staging in the C ring is its own width. *)
let c_stride t =
  match (t.geometry, t.out) with
  | Resident_a _, Dma -> lanes t * Elem.bytes (shape t).dst
  | Resident_a _, (Core | Split _) | (Resident_b | Row_major_a | Split_k _), _
    ->
      c_row t

(* Bytes of C a step leaves: a block's rows, or every row a panel wide. *)
let c_step t =
  match t.geometry with
  | Resident_a _ -> rows t * t.blocks * lanes t * Elem.bytes (shape t).dst
  | Resident_b | Row_major_a | Split_k _ -> c_block t

let a_block t =
  match t.engine with
  | Rvv_bf16 | Rvv_f32 | Rvv_i8 -> rows t * t.depth * Elem.bytes (shape t).src
  | Zvt_bf16 | Zvt_f32 | Zvt_i8 -> T.a_block (plan t)

(* B's columns by panels, each K rows of [lanes] contiguous: for RVV, whose B
   is otherwise row-major. *)
let panels t =
  let s = shape t and n = lanes t in
  let e = Elem.bytes s.src in
  let b = Bytes.make (t.depth * t.width * e) '\000' in
  for j = 0 to (t.width / n) - 1 do
    for k = 0 to t.depth - 1 do
      for c = 0 to n - 1 do
        Elem.set s.src b
          (((((j * t.depth) + k) * n) + c) * e)
          (G.b s k ((j * n) + c))
      done
    done
  done;
  Bytes.to_string b

(* Zvt's B by chunks of K, each packed as for that depth: a pass's rows of the
   chunk, pass by pass. RVV's row-major B is by chunks already. *)
let by_chunks t b =
  let e = Elem.bytes (shape t).src and chunk = block_depth t in
  let pass = t.depth * 2 * T.tile * e and part = chunk * 2 * T.tile * e in
  String.concat ""
    (List.concat
       (List.init (chunks t) (fun c ->
            List.init
              (String.length b / pass)
              (fun p -> String.sub b ((p * pass) + (c * part)) part))))

(* A row-major, M x K, as the host has it. *)
let row_major t =
  let s = shape t in
  let e = Elem.bytes s.src and m = rows t * t.blocks in
  let a = Bytes.make (m * t.depth * e) '\000' in
  for i = 0 to m - 1 do
    for k = 0 to t.depth - 1 do
      Elem.set s.src a (((i * t.depth) + k) * e) (G.a s i k)
    done
  done;
  Bytes.to_string a

(* A and B as the block packs them, without padding; A row-major for
   [Row_major_a]. Zvt's B is already by panels of two tile columns. *)
let packed t =
  let a_of a =
    match t.geometry with
    | Row_major_a -> row_major t
    | Resident_a _ | Resident_b | Split_k _ ->
        String.sub a 0 (t.blocks * a_block t)
  in
  match t.engine with
  | Rvv_bf16 | Rvv_f32 | Rvv_i8 -> (
      match G.inputs (shape t) with
      | [ a; b ] ->
          ( a_of a,
            match t.geometry with
            | Resident_a _ -> panels t
            | Resident_b | Row_major_a | Split_k _ ->
                String.sub b 0 (t.depth * b_row t) )
      | _ -> assert false)
  | Zvt_bf16 | Zvt_f32 | Zvt_i8 -> (
      match T.inputs (plan t) with
      | [ a; b ] -> (
          let b = String.sub b 0 (String.length b - 128) in
          ( a_of a,
            match t.geometry with
            | Split_k _ -> by_chunks t b
            | Resident_a _ | Resident_b | Row_major_a -> b ))
      | _ -> assert false)

(* How far past A's block and B's end the loops read ahead. *)
let a_pad t = if Engine.rvv t.engine then 64 else 128
let b_pad t = if Engine.rvv t.engine then 2 * b_row t else 128

(* Bytes of the streamed operand a step: a block of A's rows, or a panel of
   B's columns. *)
let piece t =
  match t.geometry with
  | Resident_a _ -> t.depth * b_row t
  | Resident_b | Row_major_a -> a_block t
  | Split_k { chunk } -> a_block t / t.depth * chunk

(* Bytes of B's chunk. *)
let chunk_bytes t = block_depth t * b_row t

(* The descriptors that copy a step's C out. *)
let c_descriptors t =
  match (t.geometry, t.out) with
  | _, Core -> 0
  | Resident_a _, (Dma | Split _) -> rows t * t.blocks
  | (Resident_b | Row_major_a | Split_k _), (Dma | Split _) -> 1

let slots t =
  check t;
  let a, b = packed t in
  let a_size = String.length a and b_size = String.length b in
  let table = 4 * (steps t + 4) in
  let chains =
    D.Descriptor.size
    *
    match t.geometry with
    | Resident_a _ -> (t.width / lanes t * (1 + c_descriptors t)) + 3
    | Resident_b | Row_major_a -> (2 * t.blocks) + 2
    | Split_k _ -> (4 * steps t) + 4
  in
  (* With K split, B's copy is two chunks. *)
  let streamed, resident, resident_dtcm, ring_pad =
    match t.geometry with
    | Resident_a _ -> (b_size, a_size, a_size + a_pad t, b_pad t)
    | Resident_b | Row_major_a ->
        (a_size + a_pad t, b_size, b_size + b_pad t, a_pad t)
    | Split_k _ ->
        (a_size + a_pad t, b_size, (2 * chunk_bytes t) + b_pad t, a_pad t)
  in
  L.Placement.
    [
      L.Slot.make ~home:Ram ~align:64L In (Int64.of_int streamed);
      L.Slot.make ~home:Ram ~align:64L Out (Int64.of_int (t.blocks * c_block t));
      L.Slot.make ~home:Ram ~align:64L In (Int64.of_int resident);
      L.Slot.make ~home:Dtcm ~align:64L Scratch (Int64.of_int resident_dtcm);
      L.Slot.make ~home:Dtcm ~align:64L Scratch
        (Int64.of_int ((2 * piece t) + ring_pad));
      L.Slot.make ~home:Dtcm ~align:64L Scratch
        (Int64.of_int
           (match t.out with Dma | Split _ -> 2 * c_step t | Core -> 16));
      L.Slot.make ~home:Dtcm ~align:16L In (Int64.of_int table);
      L.Slot.make ~home:Ram ~align:32L In (Int64.of_int chains);
      L.Slot.make ~home:Dtcm ~align:16L Out 16L;
    ]
  (* With K split, the zeros the first chunk's partial sums start from. *)
  @
  match t.geometry with
  | Split_k _ ->
      [ L.Slot.make ~home:Ram ~align:64L In (Int64.of_int (c_block t)) ]
  (* With A row-major, the block packed from its rows. *)
  | Row_major_a ->
      [
        L.Slot.make ~home:Dtcm ~align:64L Scratch
          (Int64.of_int (a_block t + a_pad t));
      ]
  | Resident_a _ | Resident_b -> []

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

(* The chains for a layout: the prologue (the resident operand, then piece
   0), one per step (step i - 1's rows out, piece i + 1 in), the epilogue
   (the last rows out). *)
let inputs p t (l : L.t) =
  let address k = (List.nth l.slots k).L.Placed_slot.address in
  let off a n = Int64.add a (Int64.of_int n) in
  let streamed = address Slot.streamed and c = address Slot.c in
  let ring = address Slot.ring and c_ring = address Slot.c_ring in
  let a_bytes, b_bytes = packed t in
  let resident_bytes, streamed_bytes =
    match t.geometry with
    | Resident_a _ -> (a_bytes, b_bytes)
    | Resident_b | Row_major_a | Split_k _ -> (b_bytes, a_bytes)
  in
  (* With K split, step i is block [i mod blocks] of chunk [i / blocks]. *)
  let block i = i mod t.blocks and chunk i = i / t.blocks in
  let piece_in i =
    D.Descriptor.copy
      ~src:
        (off streamed
           (match t.geometry with
           | Resident_a _ | Resident_b | Row_major_a -> i * piece t
           | Split_k _ -> (block i * a_block t) + (chunk i * piece t)))
      ~dst:(off ring (i mod 2 * piece t))
      (Int64.of_int (piece t))
  in
  (* The rows the core does not store itself; a panel's, a row at a time. *)
  let skip = Out.by_core ~rows:(rows t) t.out * c_row t in
  let c_out i =
    match t.geometry with
    | Resident_b | Row_major_a ->
        [
          D.Descriptor.copy
            ~src:(off c_ring ((i mod 2 * c_block t) + skip))
            ~dst:(off c ((i * c_block t) + skip))
            (Int64.of_int (c_block t - skip));
        ]
    | Resident_a _ ->
        List.init
          (rows t * t.blocks)
          (fun r ->
            D.Descriptor.copy
              ~src:(off c_ring ((i mod 2 * c_step t) + (r * c_stride t)))
              ~dst:(off c ((r * c_row t) + (i * c_stride t)))
              (Int64.of_int (c_stride t)))
    | Split_k _ ->
        [
          D.Descriptor.copy
            ~src:(off c_ring (i mod 2 * c_block t))
            ~dst:(off c (block i * c_block t))
            (Int64.of_int (c_block t));
        ]
  in
  (* With K split, a block's partial sums back in, from zero for the first
     chunk; and step i's share of the next chunk of B, whole beats. *)
  let c_in i =
    match t.geometry with
    | Resident_a _ | Resident_b | Row_major_a -> []
    | Split_k _ ->
        [
          D.Descriptor.copy
            ~src:
              (if chunk i = 0 then address Slot.zeros
               else off c (block i * c_block t))
            ~dst:(off c_ring (i mod 2 * c_block t))
            (Int64.of_int (c_block t));
        ]
  in
  let b_share i =
    match t.geometry with
    | Resident_a _ | Resident_b | Row_major_a -> []
    | Split_k _ ->
        let n = chunk_bytes t in
        let share = (((n + t.blocks - 1) / t.blocks) + 15) land lnot 15 in
        let from = block i * share in
        if chunk i + 1 < chunks t && from < n then
          [
            D.Descriptor.copy
              ~src:(off (address Slot.resident) (((chunk i + 1) * n) + from))
              ~dst:
                (off
                   (address Slot.resident_dtcm)
                   (((chunk i + 1) mod 2 * n) + from))
              (Int64.of_int (min share (n - from)));
          ]
        else []
  in
  (* The core's share of the resident operand, from its start, in whole
     256-byte trips. *)
  let size =
    match t.geometry with
    | Split_k _ -> chunk_bytes t
    | Resident_a _ | Resident_b | Row_major_a -> String.length resident_bytes
  in
  let by_core =
    match t.prologue with
    | Dma -> 0
    | Shared -> size * 3 / 4 / core_trip * core_trip
  in
  let pre =
    (if by_core < size then
       pieces
         ~src:(off (address Slot.resident) by_core)
         ~dst:(off (address Slot.resident_dtcm) by_core)
         (size - by_core)
     else [])
    @ [ piece_in 0 ]
    @ c_in 0
  in
  let n = steps t in
  let by_dma = c_descriptors t > 0 in
  let per i =
    (if i > 0 && by_dma then c_out (i - 1) else [])
    @ (if i + 1 < n then piece_in (i + 1) :: c_in (i + 1) else [])
    @ b_share i
  in
  let post = if by_dma then c_out (n - 1) else [] in
  let chains = (pre :: List.init n per) @ [ post ] in
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
    ([
       streamed_bytes;
       "";
       resident_bytes;
       "";
       "";
       "";
       words
         (match List.map (fun (a, _) -> Int64.to_int a) placed with
         | pre :: rest -> n :: pre :: by_core :: rest
         | [] -> assert false);
       String.concat "" (List.map snd placed);
     ]
    @
    match t.geometry with
    | Split_k _ -> [ ""; String.make (c_block t) '\000' ]
    | Resident_a _ | Resident_b | Row_major_a -> [])

(* {1 The kernel} *)

let andi dst src n =
  Item.insn (Insn.Andi { dst; src; imm = Imm.Imm12.of_int n })

let beqz src target = Item.insn (Insn.Beqz { src; target })
let fence = Item.insn Insn.Fence
let mcycle dst = Item.insn (Insn.Csrr { dst; csr = Mcycle })
let sub dst lhs rhs = Item.insn (Insn.Sub { dst; lhs; rhs })

(* [s10] holds the C ring when the core stores a share of C's rows; with A
   resident, [s10] is C's cursor and [s11] counts the step's blocks; with K
   split, [s10] is the chunk of B and [s11] counts the chunk's blocks left. *)
let saved t =
  Gpr.[ s0; s1; s2; s3; s4; s5; s6; s7; s8; s9 ]
  @
  match (t.geometry, t.out) with
  | (Resident_a _ | Split_k _), _ -> [ Gpr.s10; Gpr.s11 ]
  | Resident_b, Split _ -> [ Gpr.s10 ]
  | Resident_b, (Core | Dma) -> []
  | Row_major_a, Split _ -> [ Gpr.s10; Gpr.s11 ]
  | Row_major_a, (Core | Dma) -> [ Gpr.s11 ]

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

let items t () =
  let open Err.Syntax in
  let open Gpr in
  check t;
  let block = Label.named "block" and even = Label.named "even" in
  let label n = Label.named n in
  (* The registers a block reads its halves of the rings from. *)
  let a_at, c_at = if Engine.rvv t.engine then (a1, a6) else (a3, a4) in
  let* dma = Insn.load_imm s3 (Int64.to_int D.base) in
  let* to_odd_a = G.add_imm a_at (piece t) in
  let* to_odd_c = G.add_imm c_at (c_block t) in
  let* next_c = G.add_imm s5 (c_block t) in
  let* to_odd_ring = G.add_imm a5 (c_block t) in
  let b =
    match t.geometry with
    | Resident_a _ -> a7
    | Resident_b | Row_major_a -> s1
    | Split_k _ -> s10
  in
  let c_row = c_stride t in
  let accumulate =
    match t.geometry with
    | Split_k _ -> true
    | Resident_a _ | Resident_b | Row_major_a -> false
  in
  let block_shape =
    {
      Coralnpu_kernels.Dma_block.Shape.accumulate;
      b;
      b_row = b_row t;
      c_row;
      depth = block_depth t;
      lanes = lanes t;
    }
  in
  let* body =
    match t.engine with
    | Rvv_bf16 -> Ok (Coralnpu_kernels.Dma_block.bf16 block_shape)
    | Rvv_f32 -> Ok (Coralnpu_kernels.Dma_block.f32 block_shape)
    | Rvv_i8 -> Ok (Coralnpu_kernels.Dma_block.i8 block_shape)
    | Zvt_bf16 | Zvt_f32 | Zvt_i8 -> (
        match t.out with
        | Split n -> T.block_items ~split:(n, a5) (block_plan t)
        | Core | Dma -> T.block_items ~c_row ~accumulate (block_plan t))
  in
  let* step =
    match t.geometry with
    | (Resident_b | Row_major_a) as g ->
        let* repack =
          match g with
          | Row_major_a ->
              Coralnpu_kernels.Dma_block.repack
                ~bytes:(Elem.bytes (shape t).src) ~rows:(rows t) ~depth:t.depth
                ~src:a_at
          | Resident_a _ | Resident_b | Split_k _ -> Ok []
        in
        Ok
          ([ mv a_at s4; mv c_at s5 ]
          @ (match t.out with Split _ -> [ mv a5 s10 ] | Core | Dma -> [])
          @ [ beqz s6 even ]
          @ to_odd_a
          @ (match t.out with
            | Dma -> to_odd_c
            | Split _ -> to_odd_ring
            | Core -> [])
          @ [ Item.label even ]
          @ repack @ body
          (* With the core's stores, C's cursor moves on a block in SRAM. *)
          @ match t.out with Dma -> [] | Core | Split _ -> next_c)
    | Resident_a _ ->
        (* Every block of A against the panel at [a7], C's rows from [s10]:
           in the C ring's half, or a panel's columns on in SRAM. *)
        let inner = label "inner" in
        let* to_odd_b = G.add_imm a7 (piece t) in
        let* to_odd_c = G.add_imm s10 (c_step t) in
        let* next_a =
          if Engine.rvv t.engine then Ok [] else G.add_imm a_at (a_block t)
        in
        let* next_rows = G.add_imm s10 (rows t * c_row) in
        let* next_panel = G.add_imm s5 (lanes t * Elem.bytes (shape t).dst) in
        Ok
          ([ mv a7 s4; mv s10 s5; beqz s6 even ]
          @ to_odd_b
          @ (match t.out with Dma | Split _ -> to_odd_c | Core -> [])
          @ [ Item.label even; mv a_at s1; li s11 t.blocks; Item.label inner ]
          @ [ mv c_at s10 ]
          @ body @ next_a @ next_rows
          @ [
              addi s11 s11 (-1);
              beqz s11 (label "inner_done");
              Item.insn (J { target = inner });
              Item.label (label "inner_done");
            ]
          @ match t.out with Dma | Split _ -> [] | Core -> next_panel)
    | Split_k _ ->
        (* A block against the chunk of B at [s10], its partial sums in the C
           ring's half; after a chunk's last block, [s10] to the other chunk,
           [t6]. *)
        Ok
          ([ mv a_at s4; mv c_at s5 ]
          @ (if Engine.matrix t.engine then [ mv a7 s10 ] else [])
          @ [ beqz s6 even ]
          @ to_odd_a @ to_odd_c
          @ [ Item.label even ]
          @ body
          @ [
              addi s11 s11 (-1);
              bnez s11 (label "same_chunk");
              li s11 t.blocks;
              add t2 s1 t6;
              sub s10 t2 s10;
              Item.label (label "same_chunk");
            ])
  in
  let* chunk_setup =
    match t.geometry with
    | Split_k _ ->
        let* other = G.add_imm t6 (chunk_bytes t) in
        Ok ([ mv s10 s1; li s11 t.blocks; mv t6 s1 ] @ other)
    | Resident_a _ | Resident_b | Row_major_a -> Ok []
  in
  Ok
    (start_items
    @ [ addi sp sp (-48) ]
    @ List.mapi (fun k r -> sw r sp (4 * k)) (saved t)
    @ [
        lw s1 a0 (4 * Slot.resident_dtcm);
        lw s2 a0 (4 * Slot.steps);
        lw s4 a0 (4 * Slot.ring);
        lw s5 a0
          (4 * match t.out with Dma -> Slot.c_ring | Core | Split _ -> Slot.c);
        lw s8 a0 (4 * Slot.timings);
        mv a7 s1;
      ]
    @ (match t.out with
      | Split _ -> [ lw s10 a0 (4 * Slot.c_ring) ]
      | Core | Dma -> [])
    @ (match t.geometry with
      | Row_major_a -> [ lw s11 a0 (4 * Slot.packed) ]
      | Resident_a _ | Resident_b | Split_k _ -> [])
    @ List.map Item.insn dma
    @ [ li s7 0; lw s9 s2 4 ]
    @ start ~skip:(label "pre_started")
    @ core_copy ~src:(4 * Slot.resident) ~dst:s1 ~bytes_at:8
    @ wait ~into:s7 ~skip:(label "pre_waited") ~poll:(label "pre_poll")
    @ chunk_setup
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
    @ step
    @ wait ~into:s7 ~skip:(label "waited") ~poll:(label "poll")
    @ [
        li t2 1;
        sub s6 t2 s6;
        addi s0 s0 (-1);
        (* A wide block is more than a branch reaches. *)
        beqz s0 (label "blocks_done");
        Item.insn (J { target = block });
        Item.label (label "blocks_done");
        lw s9 s2 0;
      ]
    @ start ~skip:(label "post_started")
    @ wait ~into:s7 ~skip:(label "post_waited") ~poll:(label "post_poll")
    @ [ sw s7 s8 0 ]
    @ List.mapi (fun k r -> lw r sp (4 * k)) (saved t)
    @ [ addi sp sp 48; li a0 0; Item.insn Ret ])

let ops t = 2 * rows t * t.blocks * t.width * t.depth

let name t =
  let operand =
    match t.geometry with
    | Resident_a _ -> "a"
    | Resident_b | Row_major_a | Split_k _ -> "b"
  in
  Printf.sprintf "%s %s %s %s M=%d N=%d K=%d" (Engine.name t.engine)
    (Geometry.name t.geometry)
    (Prologue.name ~operand t.prologue)
    (Out.name t.out)
    (rows t * t.blocks)
    t.width t.depth

(* {1 The control: no DMA}

   The peak kernel itself at the same shape, with A, B and C left in SRAM: the
   core's loads and stores reach them there directly. *)
let direct t =
  check t;
  let k =
    match t.engine with
    | Rvv_bf16 -> G.make (shape t) G.bf16
    | Rvv_f32 -> G.make (shape t) (G.word G.Scalar_ops.float)
    | Rvv_i8 -> G.make (shape t) G.i8
    | Zvt_bf16 | Zvt_f32 | Zvt_i8 -> T.make (plan t)
  in
  {
    k with
    slots = List.map (fun (s : L.Slot.t) -> { s with home = Ram }) k.slots;
  }

(* {1 The reference: the peak kernel in DTCM}

   The same kernel with A, B and C all in DTCM, which must hold them: the
   rate the DMA-fed multiply is measured against. *)
let peak t =
  let k = direct t in
  {
    k with
    slots = List.map (fun (s : L.Slot.t) -> { s with home = Dtcm }) k.slots;
  }
