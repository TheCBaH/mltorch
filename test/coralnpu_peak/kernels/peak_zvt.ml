(* The peak-throughput kernels of Coral NPU's matrix engine, through Zvt: one
   tile multiply repeated, and matrix multiplies in the four accumulator tiles.
   Built like {!Peak_kernels}' own, whose helpers they use. *)

open Peak_kernels

(* {1 One tile multiply}

   [op] repeated on the four independent 32-bit accumulator tiles [mt0],
   [mt4], [mt8] and [mt12], each 16 x 16 elements spanning four tile registers.
   An instruction is 16 x 16 x [tk] multiply-adds: A's [tk] slices of 16
   elements from v8, B's from v16, a slice every two register groups, set
   whole by [vmv.v.x]. Afterwards each tile is stored a row at a time to the
   [Out] slot, which must hold the sum in every element. *)

module Tile = struct
  type shape = {
    mtwiden : int;  (** the accumulator's width over the inputs', as a code *)
    tk : int;  (** slices of A and B an instruction reduces over *)
    vtype : int;  (** with [altfmt] (bit 8) for BF16 *)
    lmul : int;
    a : int;  (** every element of A, as bits *)
    b : int;
    acc : Elem.t;
    sum : int;  (** every accumulator element after one instruction *)
  }

  (* The configuration the VME RTL requires per type: INT8 at e8 m1, BF16 at
     e16 m2 with [altfmt], FP32 at e32 m4, so a 16-element slice is one group.
     A is all-ones bits for INT8: -1 signed, 255 unsigned. *)
  let shape : Insn.Tile_op.t -> shape = function
    | Vtfmm ->
        {
          mtwiden = 1;
          tk = 1;
          vtype = 0xD2;
          lmul = 4;
          a = 0x3f80_0000;
          b = 0x3f80_0000;
          acc = F32;
          sum = 1;
        }
    | Vtfmm_alt ->
        {
          mtwiden = 2;
          tk = 2;
          vtype = 0x1C9;
          lmul = 2;
          a = 0x3f80;
          b = 0x3f80;
          acc = F32;
          sum = 2;
        }
    | Vtmms ->
        {
          mtwiden = 3;
          tk = 4;
          vtype = 0xC0;
          lmul = 1;
          a = 0xFF;
          b = 1;
          acc = I32;
          sum = -4;
        }
    | Vtmmu ->
        {
          mtwiden = 3;
          tk = 4;
          vtype = 0xC0;
          lmul = 1;
          a = 0xFF;
          b = 1;
          acc = I32;
          sum = 4 * 255;
        }

  let elements = 16
  let tiles = Reg.Mtreg.[ mt0; mt4; mt8; mt12 ]

  (* [mtype]: the tile's rows [tm], [tk] and [mtwiden]. *)
  let mtype ~tk ~mtwiden = (elements lsl 10) lor (tk lsl 5) lor mtwiden

  (* [msetmtype zero, mtype, vtype], then [msettn] for 16 columns, through
     [t1], [t2] and [t3]. *)
  let configure ~mtype ~vtype =
    let open Err.Syntax in
    let* mt = Insn.load_imm Gpr.t1 mtype in
    let* vt = Insn.load_imm Gpr.t2 vtype in
    Ok
      (List.map Item.insn (mt @ vt)
      @ [
          Item.insn
            (Msetmtype { dst = Gpr.zero; mtype = Gpr.t1; vtype = Gpr.t2 });
          li Gpr.t3 elements;
          Item.insn (Msettn { dst = Gpr.zero; avl = Gpr.t3 });
        ])

  (* The first [rows] rows of tile [mt] to [base], [stride] bytes apart, [vl]
     elements each, which leaves [base] past the last row: a slice is the tile
     number at bit 27 and the row below it. *)
  let store ~base ~stride ~rows k mt =
    let open Err.Syntax in
    let row = Label.named (Printf.sprintf "peak_store%d" k) in
    let* slice = Insn.load_imm Gpr.t3 (Reg.Mtreg.index mt lsl 27) in
    Ok
      (List.map Item.insn slice
      @ [
          li Gpr.t4 rows;
          Item.label row;
          Item.insn (Vtse { eew = Vtype.Sew.E32; slice = Gpr.t3; base });
          addi base base stride;
          addi Gpr.t3 Gpr.t3 1;
          addi Gpr.t4 Gpr.t4 (-1);
          bnez Gpr.t4 row;
        ])

  let kernel op =
    let s = shape op in
    let slices first =
      Err.List.map
        (fun k -> Reg.Vreg.of_index (first + (2 * k * s.lmul)))
        (List.init s.tk Fun.id)
    in
    let items () =
      let open Err.Syntax in
      let* a = slices 8 in
      let* b = slices 16 in
      let* config =
        configure ~mtype:(mtype ~tk:s.tk ~mtwiden:s.mtwiden) ~vtype:s.vtype
      in
      let* moves = configure ~mtype:(mtype ~tk:1 ~mtwiden:1) ~vtype:0xD2 in
      let* set_a = Insn.load_imm Gpr.t1 s.a in
      let* set_b = Insn.load_imm Gpr.t2 s.b in
      let* stores =
        Err.List.map
          (fun (k, mt) ->
            store ~base:Gpr.a6 ~stride:(4 * elements) ~rows:elements k mt)
          (List.mapi (fun k mt -> (k, mt)) tiles)
      in
      let fill r v = Item.insn (Vmv_v_x { dst = v; src = r }) in
      Ok
        (start_items
        @ [ lw Gpr.a6 Gpr.a0 0 ]
        @ config
        @ List.map Item.insn (set_a @ set_b)
        @ List.map (fill Gpr.t1) a
        @ List.map (fill Gpr.t2) b
        @ List.map (fun dst -> Item.insn (Vtzero { dst })) tiles
        @ [ li Gpr.a1 iterations ]
        @ body_items
            (List.map
               (fun acc ->
                 Item.insn (Vtmm { op; acc; a = Vreg.v8; b = Vreg.v16 }))
               tiles
            @ [ addi Gpr.a1 Gpr.a1 (-1); bnez Gpr.a1 body_label ])
        @ moves @ List.concat stores
        @ [ li Gpr.a0 0; Item.insn Ret ])
    in
    let size = List.length tiles * elements * elements * 4 in
    let expected = Bytes.create size in
    for i = 0 to (size / 4) - 1 do
      Elem.set s.acc expected (4 * i) (iterations * s.sum)
    done;
    {
      items;
      slots = [ slot L.Placement.Out size ];
      inputs = [];
      expected = Bytes.to_string expected;
      shape = Printf.sprintf "%d iterations of:" iterations;
      matrix = true;
    }
end

let tile = Tile.kernel

(* {1 Matrix multiplies through Zvt}

   C[M x W] = A[M x K] B[K x W] in 16 x 16 tiles: M up to 16 is one row of tiles
   [tm] = M deep, a multiple of 32 is blocks of two full rows. W runs in passes
   of two tile columns, the last [tn] = what is left of W wide; a pass keeps its
   tiles in the four accumulators ([mt0], [mt4] the first row, [mt8], [mt12] the
   second), each loaded slice of A or B serving every tile of its row or
   column. [tm] and [tn] are set as they are, so a partial tile is counted for
   what it computes. Both operands are packed per tile, k-major, zero beyond M
   or W: per k, the 16 rows of each tile row of a block (A) or the 16 columns of
   each tile column of a pass (B). A k step (the [tk] values a tile multiply
   reduces) of two tiles is 128 contiguous bytes, loaded by two [vle32.v] of 64:
   at the tile configuration's SEW a 32-bit element spans four registers, so a
   load fills both tiles' slices, which interleave, without a change of
   [vtype]. One tile's slices load one at a time, at their own width. The next
   step's operands load into a second buffer (v0-v7 for A, v24-v31 for B)
   while the tile multiplies read the first (v8-v15, v16-v23). INT8 is signed
   on both sides: [vtmms] with [altfmt], which makes B signed too. *)

module Tile_gemm = struct
  (* The input types the matrix engine multiplies. Alphabetical. *)
  module Input = struct
    type t = Bf16 | F32 | I8

    let all = [ Bf16; F32; I8 ]
    let name = function Bf16 -> "bf16" | F32 -> "f32" | I8 -> "i8"
    let of_name n = List.find_opt (fun t -> String.equal (name t) n) all

    let to_gemm : t -> Gemm.Input.t = function
      | Bf16 -> Bf16
      | F32 -> F32
      | I8 -> I8
  end

  (* The engine's own shape, two full tile rows by two full tile columns with
     A, B and C in DTCM; or the RVV kernel's, with its width if given.
     Alphabetical. *)
  module Geometry = struct
    type t = Engine | Rvv of { width : int option }
  end

  let tile = Tile.elements

  type plan = {
    s : Gemm.shape;  (** M = [s.rows], one block *)
    op : Insn.Tile_op.t;
    t : Tile.shape;
    tile_rows : int;  (** of a block: 1 or 2 *)
    blocks : int;
    tm : int;
    columns : int;  (** tile columns *)
    last : int;  (** the last tile column's width *)
  }

  let plan op (t : Tile.shape) (s : Gemm.shape) =
    let m = s.rows in
    let tile_rows, blocks, tm =
      if m <= tile then (1, 1, m)
      else if m mod (2 * tile) = 0 then (2, m / (2 * tile), tile)
      else
        invalid_arg
          (Printf.sprintf
             "Peak_zvt.Tile_gemm: M = %d is not up to 16 or a multiple of 32" m)
    in
    if s.depth mod (2 * t.tk) <> 0 then
      invalid_arg
        (Printf.sprintf "Peak_zvt.Tile_gemm: K = %d is not a multiple of %d"
           s.depth (2 * t.tk));
    let columns = (s.width + tile - 1) / tile in
    {
      s;
      op;
      t;
      tile_rows;
      blocks;
      tm;
      columns;
      last = s.width - ((columns - 1) * tile);
    }

  (* The tile columns of each pass, two at a time. *)
  let passes p =
    List.init
      ((p.columns + 1) / 2)
      (fun k -> List.filter (fun c -> c < p.columns) [ 2 * k; (2 * k) + 1 ])

  let accumulator r c = List.nth Tile.tiles ((2 * r) + c)

  (* One k step of [n] tiles' slices into the buffer at [base] from [ptr]:
     both tiles' in two 64-byte loads, one tile's a slice at a time. *)
  let loads p ~n ~base ptr =
    let open Err.Syntax in
    let e = Elem.bytes p.s.src in
    let load eew v =
      [ vle eew v ptr; addi ptr ptr (if n = 2 then 64 else tile * e) ]
    in
    let regs =
      if n = 2 then [ base; base + 4 ]
      else List.init p.t.tk (fun j -> base + (2 * j * p.t.lmul))
    in
    let eew =
      if n = 2 then Vtype.Sew.E32
      else
        match p.s.src with
        | I8 -> Vtype.Sew.E8
        | Bf16 | I16 -> E16
        | F32 | I32 -> E32
    in
    let* regs = Err.List.map Reg.Vreg.of_index regs in
    Ok (List.concat_map (load eew) regs)

  (* The tile multiplies of one step, a column at a time: [tn] is narrowed
     ([t1]) for a partial last column and restored ([t3], 16) for the loads. *)
  let multiplies p cols ~a ~b =
    let open Err.Syntax in
    let* per_column =
      Err.List.map
        (fun (k, c) ->
          let* bv = Reg.Vreg.of_index (b + (k * p.t.lmul)) in
          let* mults =
            Err.List.map
              (fun r ->
                let* av = Reg.Vreg.of_index (a + (r * p.t.lmul)) in
                Ok
                  (Item.insn
                     (Vtmm { op = p.op; acc = accumulator r k; a = av; b = bv })))
              (List.init p.tile_rows Fun.id)
          in
          let msettn avl = Item.insn (Msettn { dst = Gpr.zero; avl }) in
          Ok
            (if c = p.columns - 1 && p.last < tile then
               (msettn Gpr.t1 :: mults) @ [ msettn Gpr.t3 ]
             else mults))
        (List.mapi (fun k c -> (k, c)) cols)
    in
    Ok (List.concat per_column)

  (* Bytes of A's block of rows, and of a row of C. *)
  let a_block p =
    p.s.depth / p.t.tk * p.tile_rows * tile * p.t.tk * Elem.bytes p.s.src

  let c_row p = p.s.width * Elem.bytes p.s.dst

  (* One block of rows: every pass over A's block at [a3] and B at [a7], C's
     rows stored from [a4], [c_row p] apart; [a3] and [a4] are left as they
     were. Writes [a1], [a2], [a6], [t1]-[t5]. *)
  let block_items p =
    let open Err.Syntax in
    let open Gpr in
    let src = Elem.bytes p.s.src and dst = Elem.bytes p.s.dst in
    let k_steps = p.s.depth / p.t.tk in
    let c_row = c_row p in
    let* config =
      Tile.configure
        ~mtype:((p.tm lsl 10) lor (p.t.tk lsl 5) lor p.t.mtwiden)
        ~vtype:p.t.vtype
    in
    let* moves =
      Tile.configure ~mtype:(Tile.mtype ~tk:1 ~mtwiden:1) ~vtype:0xD2
    in
    let partial = if p.last < tile then [ li t1 p.last ] else [] in
    let stores_seen = ref 0 in
    let* pass_items =
      Err.List.map
        (fun (index, cols) ->
          let n = List.length cols in
          let b_offset = index * 2 * k_steps * tile * p.t.tk * src in
          (* column by column, so a partial last column's [tn] comes last *)
          let tiles =
            List.concat
              (List.mapi
                 (fun k c -> List.init p.tile_rows (fun r -> (r, k, c)))
                 cols)
          in
          let* first_a = loads p ~n:p.tile_rows ~base:8 a1 in
          let* first_b = loads p ~n ~base:16 a2 in
          let* second_a = loads p ~n:p.tile_rows ~base:0 a1 in
          let* second_b = loads p ~n ~base:24 a2 in
          let* current = multiplies p cols ~a:8 ~b:16 in
          let* next = multiplies p cols ~a:0 ~b:24 in
          let* b_at = Insn.load_imm t2 b_offset in
          let* stores =
            Err.List.map
              (fun (r, k, c) ->
                let* at =
                  Insn.load_imm t5 ((r * tile * c_row) + (c * tile * dst))
                in
                let k' = !stores_seen in
                incr stores_seen;
                let* store =
                  Tile.store ~base:t5 ~stride:c_row ~rows:p.tm k'
                    (accumulator r k)
                in
                let narrow =
                  if c = p.columns - 1 && p.last < tile then
                    [
                      li t1 p.last; Item.insn (Msettn { dst = zero; avl = t1 });
                    ]
                  else []
                in
                Ok (narrow @ List.map Item.insn at @ [ add t5 t5 a4 ] @ store))
              tiles
          in
          let loop =
            second_a @ second_b @ current @ first_a @ first_b @ next
            @ [ addi a6 a6 (-1) ]
          in
          let body =
            if index = 0 then body_items (loop @ [ bnez a6 body_label ])
            else
              let l = Label.named (Printf.sprintf "peak_pass%d" index) in
              (Item.label l :: loop) @ [ bnez a6 l ]
          in
          Ok
            (config @ partial
            @ List.map
                (fun (r, k, _) -> Item.insn (Vtzero { dst = accumulator r k }))
                tiles
            @ List.map Item.insn b_at
            @ [ mv a1 a3; add a2 a7 t2; li a6 (k_steps / 2) ]
            @ first_a @ first_b @ body @ moves @ List.concat stores))
        (List.mapi (fun k cols -> (k, cols)) (passes p))
    in
    Ok (List.concat pass_items)

  let items p () =
    let open Err.Syntax in
    let open Gpr in
    let block = Label.named "peak_block" in
    let* blocks = block_items p in
    let* next_a = Insn.load_imm t2 (a_block p) in
    let* next_c = Insn.load_imm t2 (2 * tile * c_row p) in
    Ok
      (start_items
      @ [ lw a3 a0 0; lw a7 a0 4; lw a4 a0 8; li a5 p.blocks; Item.label block ]
      @ blocks @ List.map Item.insn next_a
      @ [ add a3 a3 t2 ]
      @ List.map Item.insn next_c
      @ [ add a4 a4 t2 ]
      @ [ addi a5 a5 (-1); bnez a5 block; li a0 0; Item.insn Ret ])

  (* A and B packed per tile, k-major, zero beyond M and W, each with a step
     past its end for the loop's load ahead. *)
  let inputs p =
    let s = p.s and e = Elem.bytes p.s.src in
    let pack ~outer ~inner value =
      let b = Bytes.make ((outer * s.depth * inner * tile * e) + 128) '\000' in
      let n = ref 0 in
      for o = 0 to outer - 1 do
        for k = 0 to s.depth - 1 do
          for i = 0 to (inner * tile) - 1 do
            (match value o k i with
            | Some x -> Elem.set s.src b (!n * e) x
            | None -> ());
            incr n
          done
        done
      done;
      Bytes.to_string b
    in
    let a =
      pack ~outer:p.blocks ~inner:p.tile_rows (fun blk k i ->
          let row = (blk * 2 * tile) + i in
          if row < s.rows then Some (Gemm.a s row k) else None)
    in
    let passes = passes p in
    let b =
      (* each pass two tile columns wide, the last possibly one *)
      let buf = Buffer.create 4096 in
      List.iteri
        (fun index cols ->
          let n = List.length cols in
          let one =
            pack ~outer:1 ~inner:n (fun _ k i ->
                let col = (2 * index * tile) + i in
                if col < s.width then Some (Gemm.b s k col) else None)
          in
          Buffer.add_string buf (String.sub one 0 (String.length one - 128)))
        passes;
      Buffer.add_string buf (String.make 128 '\000');
      Buffer.contents buf
    in
    [ a; b ]

  let make p =
    let inputs = inputs p in
    let expected = Gemm.product p.s in
    let s = p.s in
    {
      items = items p;
      slots =
        [
          slot L.Placement.In (String.length (List.nth inputs 0));
          slot L.Placement.In (String.length (List.nth inputs 1));
          slot L.Placement.Out (String.length expected);
        ];
      inputs;
      expected;
      shape =
        Printf.sprintf
          "C[%dx%d] %s = A[%dx%d] %s x B[%dx%d] %s, tiles %d deep, %d columns \
           (the last %d wide) in %d pass%s; inner loop:"
          s.rows s.width (Elem.name s.dst) s.rows s.depth (Elem.name s.src)
          s.depth s.width (Elem.name s.src) p.tm p.columns p.last
          (List.length (passes p))
          (if List.length (passes p) = 1 then "" else "es");
      matrix = true;
    }
end

(* The engine's shapes have 32 k steps a block, and A, B and C fit DTCM
   together: in testbench RAM, B's loads take three times as long. *)
let tile_gemm ?(geometry = Tile_gemm.Geometry.Engine)
    (input : Tile_gemm.Input.t) =
  let open Tile_gemm in
  let op, t, dst =
    match input with
    | Bf16 -> (Insn.Tile_op.Vtfmm_alt, Tile.shape Vtfmm_alt, Elem.F32)
    | F32 -> (Vtfmm, Tile.shape Vtfmm, F32)
    | I8 -> (Vtmms, { (Tile.shape Vtmms) with vtype = 0x1C0 }, I32)
  in
  let s =
    match geometry with
    | Engine ->
        let depth = match input with Bf16 -> 64 | F32 -> 32 | I8 -> 128 in
        let src : Elem.t =
          match input with Bf16 -> Bf16 | F32 -> F32 | I8 -> I8
        in
        {
          Gemm.rows = 64;
          width = 32;
          depth;
          blocks = 1;
          src;
          dst;
          sa = 1;
          sb = 1;
        }
    | Rvv { width } ->
        let r = gemm_shape ?width (Input.to_gemm input) in
        { r with rows = r.rows * r.blocks; blocks = 1; dst }
  in
  make (plan op t s)
