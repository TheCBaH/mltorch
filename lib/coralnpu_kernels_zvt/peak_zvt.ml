(* The peak-throughput kernels of Coral NPU's matrix engine, through Zvt: one
   tile multiply repeated, and matrix multiplies in the four accumulator tiles.
   Built like {!Peak_kernels}' own, whose helpers they use. *)

open Coralnpu_kernels.Peak_kernels
module Units = Coral_transfer.Transfer_units

(* The first register of the group [n] groups above [base]'s. *)
let groups_above lmul base (n : Core.Dim.index Core.Dim.t) =
  let open Err.Syntax in
  let rec go r n =
    if n = 0 then Ok r
    else
      let* next = Vtype.Lmul.next lmul r in
      go next (n - 1)
  in
  go base (n :> int)

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
    tk : Core.Dim.extent Core.Dim.t;
        (** slices of A and B an instruction reduces over *)
    vtype : int;  (** with [altfmt] (bit 8) for BF16 *)
    lmul : Vtype.Lmul.t;
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
          tk = Core.Dim.extent 1;
          vtype = 0xD2;
          lmul = M4;
          a = 0x3f80_0000;
          b = 0x3f80_0000;
          acc = F32;
          sum = 1;
        }
    | Vtfmm_alt ->
        {
          mtwiden = 2;
          tk = Core.Dim.extent 2;
          vtype = 0x1C9;
          lmul = M2;
          a = 0x3f80;
          b = 0x3f80;
          acc = F32;
          sum = 2;
        }
    | Vtmms ->
        {
          mtwiden = 3;
          tk = Core.Dim.extent 4;
          vtype = 0xC0;
          lmul = M1;
          a = 0xFF;
          b = 1;
          acc = I32;
          sum = -4;
        }
    | Vtmmu ->
        {
          mtwiden = 3;
          tk = Core.Dim.extent 4;
          vtype = 0xC0;
          lmul = M1;
          a = 0xFF;
          b = 1;
          acc = I32;
          sum = 4 * 255;
        }

  let elements = Core.Dim.extent 16
  let tiles = Reg.Mtreg.[ mt0; mt4; mt8; mt12 ]

  (* [mtype]: the tile's rows [tm], [tk] and [mtwiden]. *)
  let mtype ~tk ~mtwiden = ((elements :> int) lsl 10) lor (tk lsl 5) lor mtwiden

  (* [msetmtype mtype, vtype], then [msettn] for 16 columns, through
     [t1], [t2] and [t3]. *)
  let configure ~mtype ~vtype =
    let open Err.Syntax in
    let* mt = Insn.load_imm Gpr.t1 mtype in
    let* vt = Insn.load_imm Gpr.t2 vtype in
    Ok
      (List.map Item.insn (mt @ vt)
      @ [
          Item.insn (Msetmtype { mtype = Gpr.t1; vtype = Gpr.t2 });
          li Gpr.t3 (elements :> int);
          Item.insn (Msettn { dst = Gpr.zero; avl = Gpr.t3 });
        ])

  (* The first [rows] rows of tile [mt] to [base], [stride] bytes apart, [vl]
     elements each, which leaves [base] past the last row: a slice is the tile
     number at bit 27 and the row below it. A stride too far for an [addi]
     goes through [t2]. [from] rows are skipped first. *)
  let store ?(from = Units.Rows.of_int 0) ~base ~(stride : Units.Bytes.t)
      ~(rows : Units.Rows.t) ~label:row mt =
    let open Err.Syntax in
    let stride = (stride :> int) in
    let* slice =
      Insn.load_imm Gpr.t3 ((Reg.Mtreg.index mt lsl 27) + (from :> int))
    in
    let* set_stride, step =
      if Imm.Imm12.(in_range (of_int stride)) then Ok ([], addi base base stride)
      else
        Result.map
          (fun set -> (List.map Item.insn set, add base base Gpr.t2))
          (Insn.load_imm Gpr.t2 stride)
    in
    Ok
      (List.map Item.insn slice @ set_stride
      @ [
          li Gpr.t4 (rows :> int);
          Item.label row;
          Item.insn (Vtse { eew = Vtype.Sew.E32; slice = Gpr.t3; base });
          step;
          addi Gpr.t3 Gpr.t3 1;
          addi Gpr.t4 Gpr.t4 (-1);
          bnez Gpr.t4 row;
        ])

  (* Rows [from, from + rows) of tile [mt] loaded from [base], as {!store}
     stores them. *)
  let load ~(from : Units.Rows.t) ~base ~(stride : Units.Bytes.t)
      ~(rows : Units.Rows.t) ~label:row mt =
    let open Err.Syntax in
    let stride = (stride :> int) in
    let* slice =
      Insn.load_imm Gpr.t3 ((Reg.Mtreg.index mt lsl 27) + (from :> int))
    in
    let* set_stride, step =
      if Imm.Imm12.(in_range (of_int stride)) then Ok ([], addi base base stride)
      else
        Result.map
          (fun set -> (List.map Item.insn set, add base base Gpr.t2))
          (Insn.load_imm Gpr.t2 stride)
    in
    Ok
      (List.map Item.insn slice @ set_stride
      @ [
          li Gpr.t4 (rows :> int);
          Item.label row;
          Item.insn (Vtle { eew = Vtype.Sew.E32; slice = Gpr.t3; base });
          step;
          addi Gpr.t3 Gpr.t3 1;
          addi Gpr.t4 Gpr.t4 (-1);
          bnez Gpr.t4 row;
        ])

  let kernel op =
    let s = shape op in
    let slices first =
      Err.List.map
        (fun k -> groups_above s.lmul first (Core.Dim.index (2 * k)))
        (List.init (s.tk :> int) Fun.id)
    in
    let items () =
      let open Err.Syntax in
      let* a = slices Vreg.v8 in
      let* b = slices Vreg.v16 in
      let* config =
        configure
          ~mtype:(mtype ~tk:(s.tk :> int) ~mtwiden:s.mtwiden)
          ~vtype:s.vtype
      in
      let* moves = configure ~mtype:(mtype ~tk:1 ~mtwiden:1) ~vtype:0xD2 in
      let* set_a = Insn.load_imm Gpr.t1 s.a in
      let* set_b = Insn.load_imm Gpr.t2 s.b in
      let* stores =
        Err.List.map
          (fun (k, mt) ->
            store ~base:Gpr.a6
              ~stride:(Units.Bytes.times_extent elements (Elem.size F32))
              ~rows:(Units.Rows.of_int (elements :> int))
              ~label:(Label.named (Printf.sprintf "peak_store%d" k))
              mt)
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
    let size = List.length tiles * (elements :> int) * (elements :> int) * 4 in
    let expected = Bytes.create size in
    for i = 0 to (size / 4) - 1 do
      Elem.set s.acc expected i (iterations * s.sum)
    done;
    {
      items;
      slots =
        [
          slot L.Placement.Out (Coral_transfer.Transfer_units.Bytes.of_int size);
        ];
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
    type t = Engine | Rvv of { width : Core.Dim.extent Core.Dim.t option }
  end

  let tile = Tile.elements

  type plan = {
    s : Gemm.shape;  (** M = [s.rows], one block *)
    op : Insn.Tile_op.t;
    t : Tile.shape;
    tile_rows : Core.Dim.extent Core.Dim.t;  (** of a block: 1 or 2 *)
    blocks : Core.Dim.extent Core.Dim.t;
    tm : Core.Dim.extent Core.Dim.t;
    columns : Core.Dim.extent Core.Dim.t;  (** tile columns *)
    last : Core.Dim.extent Core.Dim.t;  (** the last tile column's width *)
  }

  let plan op (t : Tile.shape) (s : Gemm.shape) =
    let ext = Core.Dim.extent in
    let m = (s.rows :> int) and tl = (tile :> int) in
    let tile_rows, blocks, tm =
      if m <= tl then (1, 1, m)
      else if m mod (2 * tl) = 0 then (2, m / (2 * tl), tl)
      else
        invalid_arg
          (Printf.sprintf
             "Peak_zvt.Tile_gemm: M = %d is not up to 16 or a multiple of 32" m)
    in
    let tk = (t.tk :> int) in
    if (s.depth :> int) mod (2 * tk) <> 0 then
      invalid_arg
        (Printf.sprintf "Peak_zvt.Tile_gemm: K = %d is not a multiple of %d"
           (s.depth :> int)
           (2 * tk));
    let width = (s.width :> int) in
    let columns = (width + tl - 1) / tl in
    {
      s;
      op;
      t;
      tile_rows = ext tile_rows;
      blocks = ext blocks;
      tm = ext tm;
      columns = ext columns;
      last = ext (width - ((columns - 1) * tl));
    }

  (* The tile columns of each pass, two at a time. *)
  let passes p =
    let columns = (p.columns :> int) in
    List.init
      ((columns + 1) / 2)
      (fun k ->
        List.filter_map
          (fun c -> if c < columns then Some (Core.Dim.index c) else None)
          [ 2 * k; (2 * k) + 1 ])

  let accumulator (r : Core.Dim.index Core.Dim.t)
      (c : Core.Dim.index Core.Dim.t) =
    List.nth Tile.tiles ((2 * (r :> int)) + (c :> int))

  (* The last tile column of the plan. *)
  let last_column p = Core.Dim.index ((p.columns :> int) - 1)

  (* One k step of [n] tiles' slices into the buffer at [base] from [ptr]:
     both tiles' in two 64-byte loads, one tile's a slice at a time. [pitch] is
     the bytes from one load to the next. *)
  let loads ?(pitch : Units.Bytes.t option) p ~(n : Core.Dim.extent Core.Dim.t)
      ~(base : Reg.Vreg.t) ptr =
    let open Err.Syntax in
    let e = (Elem.size p.s.src :> int) in
    let two = Core.Dim.equal n (Core.Dim.extent 2) in
    let load eew v =
      [
        vle eew v ptr;
        addi ptr ptr
          (match pitch with
          | Some x -> (x :> int)
          | None -> if two then 64 else (tile :> int) * e);
      ]
    in
    let* regs =
      if two then
        let* second = Reg.Vreg.of_index (Reg.Vreg.index base + 4) in
        Ok [ base; second ]
      else
        Err.List.map
          (fun j -> groups_above p.t.lmul base (Core.Dim.index (2 * j)))
          (List.init (p.t.tk :> int) Fun.id)
    in
    let eew =
      if two then Vtype.Sew.E32
      else
        match p.s.src with
        | I8 -> Vtype.Sew.E8
        | Bf16 | I16 -> E16
        | F32 | I32 -> E32
    in
    Ok (List.concat_map (load eew) regs)

  (* The tile multiplies of one step, a column at a time: [tn] is narrowed
     ([t1]) for a partial last column and restored ([t3], 16) for the loads. *)
  let multiplies p cols ~(a : Reg.Vreg.t) ~(b : Reg.Vreg.t) =
    let open Err.Syntax in
    let* per_column =
      Err.List.map
        (fun (k, c) ->
          let* bv = groups_above p.t.lmul b (Core.Dim.index k) in
          let* mults =
            Err.List.map
              (fun r ->
                let* av = groups_above p.t.lmul a (Core.Dim.index r) in
                Ok
                  (Item.insn
                     (Vtmm
                        {
                          op = p.op;
                          acc =
                            accumulator (Core.Dim.index r) (Core.Dim.index k);
                          a = av;
                          b = bv;
                        })))
              (List.init (p.tile_rows :> int) Fun.id)
          in
          let msettn avl = Item.insn (Msettn { dst = Gpr.zero; avl }) in
          Ok
            (if Core.Dim.equal c (last_column p) && p.last < tile then
               (msettn Gpr.t1 :: mults) @ [ msettn Gpr.t3 ]
             else mults))
        (List.mapi (fun k c -> (k, c)) cols)
    in
    Ok (List.concat per_column)

  (* Bytes of A's block of rows, and of a row of C. *)
  let a_block p =
    let tk = (p.t.tk :> int) in
    Units.Bytes.of_int
      ((p.s.depth :> int)
      / tk
      * (p.tile_rows :> int)
      * (tile :> int)
      * tk
      * (Elem.size p.s.src :> int))

  let c_row p = Units.Bytes.times_extent p.s.width (Elem.size p.s.dst)

  (* One block of rows: every pass over A's block at [a3] and B at [a7], C's
     rows stored from [a4], [c_row p] apart; [a3] and [a4] are left as they
     were. Writes [a1], [a2], [a6], [t1]-[t5]. With [split] = [(n, ring)], a
     block's first [n] rows go to [a4] and the rest to [ring], at the same
     offsets; [c_row] overrides the bytes from a row of C to the next. With
     [accumulate], each pass's accumulators start from C at [a4], not zero.
     [prefix] names the labels (the default keeps the peak suite's exported
     body); [a_pitch] and [b_pitch] are the bytes from one k row of A or B to
     the next, 64 packed (32 for a single tile), and [b_pass] the bytes from a
     pass's B to the next, so that operands k-major in memory are read in
     place. *)
  let block_items ?(prefix = "peak") ?(a_pitch : Units.Bytes.t option)
      ?(b_pitch : Units.Bytes.t option) ?(b_pass : Units.Bytes.t option)
      ?(split : (Units.Rows.t * Gpr.t) option)
      ?c_row:(row : Units.Bytes.t option) ?(accumulate = false) p =
    let open Err.Syntax in
    let open Gpr in
    let src = (Elem.size p.s.src :> int) and dst = (Elem.size p.s.dst :> int) in
    let tl = (tile :> int) and tk = (p.t.tk :> int) and tm = (p.tm :> int) in
    let rows_of n = Units.Rows.of_int n in
    let k_steps = (p.s.depth :> int) / tk in
    let c_row = (Option.value row ~default:(c_row p) :> int) in
    let* config =
      Tile.configure
        ~mtype:((tm lsl 10) lor (tk lsl 5) lor p.t.mtwiden)
        ~vtype:p.t.vtype
    in
    let* moves =
      Tile.configure ~mtype:(Tile.mtype ~tk:1 ~mtwiden:1) ~vtype:0xD2
    in
    let partial = if p.last < tile then [ li t1 (p.last :> int) ] else [] in
    let stores_seen = ref 0 and loads_seen = ref 0 in
    let* pass_items =
      Err.List.map
        (fun (index, cols) ->
          let n = Core.Dim.extent (List.length cols) in
          let b_offset =
            match b_pass with
            | Some bytes -> index * (bytes :> int)
            | None -> index * 2 * k_steps * tl * tk * src
          in
          (* column by column, so a partial last column's [tn] comes last *)
          let tiles =
            List.concat
              (List.mapi
                 (fun k c ->
                   List.init
                     (p.tile_rows :> int)
                     (fun r -> (Core.Dim.index r, Core.Dim.index k, c)))
                 cols)
          in
          let* first_a =
            loads ?pitch:a_pitch p ~n:p.tile_rows ~base:Reg.Vreg.v8 a1
          in
          let* first_b = loads ?pitch:b_pitch p ~n ~base:Reg.Vreg.v16 a2 in
          let* second_a =
            loads ?pitch:a_pitch p ~n:p.tile_rows ~base:Reg.Vreg.v0 a1
          in
          let* second_b = loads ?pitch:b_pitch p ~n ~base:Reg.Vreg.v24 a2 in
          let* current = multiplies p cols ~a:Reg.Vreg.v8 ~b:Reg.Vreg.v16 in
          let* next = multiplies p cols ~a:Reg.Vreg.v0 ~b:Reg.Vreg.v24 in
          let* b_at = Insn.load_imm t2 b_offset in
          (* Rows [from, from + rows) of a tile to or from [base]'s C. *)
          let part ?(load = false) (r, k, c) ~base ~(from : Units.Rows.t)
              ~(rows : Units.Rows.t) =
            if (rows :> int) = 0 then Ok []
            else
              let* at =
                Insn.load_imm t5
                  (((((r : Core.Dim.index Core.Dim.t) :> int) * tl)
                   + (from :> int))
                   * c_row
                  + ((c : Core.Dim.index Core.Dim.t :> int) * tl * dst))
              in
              let seen = if load then loads_seen else stores_seen in
              let k' = !seen in
              incr seen;
              let* move =
                if load then
                  Tile.load ~from ~base:t5 ~stride:(Units.Bytes.of_int c_row)
                    ~rows
                    ~label:(Label.named (Printf.sprintf "%s_load%d" prefix k'))
                    (accumulator r k)
                else
                  Tile.store ~from ~base:t5 ~stride:(Units.Bytes.of_int c_row)
                    ~rows
                    ~label:(Label.named (Printf.sprintf "%s_store%d" prefix k'))
                    (accumulator r k)
              in
              Ok (List.map Item.insn at @ [ add t5 t5 base ] @ move)
          in
          let narrow c =
            if Core.Dim.equal c (last_column p) && p.last < tile then
              [
                li t1 (p.last :> int);
                Item.insn (Msettn { dst = zero; avl = t1 });
              ]
            else []
          in
          let* starts =
            if accumulate then
              let* loads =
                Err.List.map
                  (fun (r, k, c) ->
                    let* load =
                      part ~load:true (r, k, c) ~base:a4 ~from:(rows_of 0)
                        ~rows:(rows_of tm)
                    in
                    Ok (narrow c @ load))
                  tiles
              in
              Ok (moves @ List.concat loads @ config @ partial)
            else
              Ok
                (config @ partial
                @ List.map
                    (fun (r, k, _) ->
                      Item.insn (Vtzero { dst = accumulator r k }))
                    tiles)
          in
          let* stores =
            Err.List.map
              (fun ((r : Core.Dim.index Core.Dim.t), k, c) ->
                let first =
                  match split with
                  | None -> tm
                  | Some (n, _) ->
                      max 0 (min tm ((n :> int) - ((r :> int) * tl)))
                in
                let* core =
                  part (r, k, c) ~base:a4 ~from:(rows_of 0)
                    ~rows:(rows_of first)
                in
                let* rest =
                  match split with
                  | None -> Ok []
                  | Some (_, ring) ->
                      part (r, k, c) ~base:ring ~from:(rows_of first)
                        ~rows:(rows_of (tm - first))
                in
                Ok (narrow c @ core @ rest))
              tiles
          in
          let loop =
            second_a @ second_b @ current @ first_a @ first_b @ next
            @ [ addi a6 a6 (-1) ]
          in
          let body =
            if index = 0 && prefix = "peak" then
              body_items (loop @ [ bnez a6 body_label ])
            else
              let l = Label.named (Printf.sprintf "%s_pass%d" prefix index) in
              (Item.label l :: loop) @ [ bnez a6 l ]
          in
          Ok
            (starts @ List.map Item.insn b_at
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
    let* next_a = Insn.load_imm t2 (a_block p :> int) in
    let* next_c = Insn.load_imm t2 (2 * (tile :> int) * (c_row p :> int)) in
    Ok
      (start_items
      @ [
          lw a3 a0 0;
          lw a7 a0 4;
          lw a4 a0 8;
          li a5 (p.blocks :> int);
          Item.label block;
        ]
      @ blocks @ List.map Item.insn next_a
      @ [ add a3 a3 t2 ]
      @ List.map Item.insn next_c
      @ [ add a4 a4 t2 ]
      @ [ addi a5 a5 (-1); bnez a5 block; li a0 0; Item.insn Ret ])

  (* A and B packed per tile, k-major, zero beyond M and W, each with a step
     past its end for the loop's load ahead. *)
  let inputs p =
    let s = p.s and e = (Elem.size p.s.src :> int) in
    let tl = (tile :> int) and depth = (s.depth :> int) in
    let rows = (s.rows :> int) and width = (s.width :> int) in
    let pack ~outer ~inner value =
      let b = Bytes.make ((outer * depth * inner * tl * e) + 128) '\000' in
      let n = ref 0 in
      for o = 0 to outer - 1 do
        for k = 0 to depth - 1 do
          for i = 0 to (inner * tl) - 1 do
            (match value o k i with
            | Some x -> Elem.set s.src b !n x
            | None -> ());
            incr n
          done
        done
      done;
      Bytes.to_string b
    in
    let a =
      pack
        ~outer:(p.blocks :> int)
        ~inner:(p.tile_rows :> int)
        (fun blk k i ->
          let row = (blk * 2 * tl) + i in
          if row < rows then
            Some (Gemm.a s (Core.Dim.index row) (Core.Dim.index k))
          else None)
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
                let col = (2 * index * tl) + i in
                if col < width then
                  Some (Gemm.b s (Core.Dim.index k) (Core.Dim.index col))
                else None)
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
          slot L.Placement.In
            (Coral_transfer.Transfer_units.Bytes.of_int
               (String.length (List.nth inputs 0)));
          slot L.Placement.In
            (Coral_transfer.Transfer_units.Bytes.of_int
               (String.length (List.nth inputs 1)));
          slot L.Placement.Out
            (Coral_transfer.Transfer_units.Bytes.of_int (String.length expected));
        ];
      inputs;
      expected;
      shape =
        Printf.sprintf
          "C[%dx%d] %s = A[%dx%d] %s x B[%dx%d] %s, tiles %d deep, %d columns \
           (the last %d wide) in %d pass%s; inner loop:"
          (s.rows :> int)
          (s.width :> int)
          (Elem.name s.dst)
          (s.rows :> int)
          (s.depth :> int)
          (Elem.name s.src)
          (s.depth :> int)
          (s.width :> int)
          (Elem.name s.src)
          (p.tm :> int)
          (p.columns :> int)
          (p.last :> int)
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
          Gemm.rows = Core.Dim.extent 64;
          width = Core.Dim.extent 32;
          depth = Core.Dim.extent depth;
          blocks = Core.Dim.extent 1;
          src;
          dst;
          sa = 1;
          sb = 1;
        }
    | Rvv { width } ->
        let r = gemm_shape ?width (Input.to_gemm input) in
        {
          r with
          rows = Core.Dim.extent ((r.rows :> int) * (r.blocks :> int));
          blocks = Core.Dim.extent 1;
          dst;
        }
  in
  make (plan op t s)

(* Coral NPU's Zvt matrix instructions, from coralview's Rivet table of them:
   one encoder, made once. *)
let zvt =
  Result.map
    (Lower.Extension.make ~feature:Insn.zvt_feature)
    (Coral_isa.Coral_zvt.definition ())

(* A peak kernel as a module under [profile], with the Zvt extension. *)
let build t profile =
  Result.bind zvt (fun zvt -> build_with ~extensions:[ zvt ] t profile)
