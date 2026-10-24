(* The peak-throughput kernels as typed Rivet modules: one RVV instruction
   repeated on independent accumulators, and matrix multiplies that keep two or
   three rows of C in LMUL=8 accumulators. Each is [peak_entry(table)] under the
   Coral runtime's [Table] ABI, with its timed loop between the exported labels
   [peak_body] and [peak_body_end]. *)

module L = Coral_layout
module Rv32_profile = Machine_rivet_riscv32.Rv32_profile
module Item = Machine_rivet_riscv32.Rv32_kernel_item
module Kernel = Machine_rivet_riscv32.Rv32_kernel
module Lower = Machine_rivet_riscv32.Rv32_kernel_lower
module Insn = Machine_rivet_riscv32.Rv32_kernel_insn
module Imm = Machine_rivet_riscv32.Rv32_kernel_imm
module Label = Machine_rivet_riscv32.Rv32_kernel_label
module Reg = Machine_rivet_riscv32.Rv32_kernel_reg
module Vtype = Machine_rivet_riscv32.Rv32_kernel_vtype
module Gpr = Reg.Gpr
module Vreg = Reg.Vreg

let entry = "peak_entry"

(* The profile every peak image is built for: Coral NPU's RV32 core with the
   vector unit and BF16 widening multiply-add. *)
let rv32_profile =
  Machine_rivet_riscv32.Rv32_profile.Rv32imf_zve32f_zvfbfwma_ilp32f

(* An element type of the inputs or the accumulators. *)
module Elem = struct
  type t = Bf16 | F32 | I8 | I16 | I32

  let bytes = function Bf16 | I16 -> 2 | F32 | I32 -> 4 | I8 -> 1

  let name = function
    | Bf16 -> "bf16"
    | F32 -> "fp32"
    | I8 -> "int8"
    | I16 -> "int16"
    | I32 -> "int32"

  (* [x] (a small integer, exact in every type) at [at] in [b]. *)
  let set t b at x =
    let f32 = Int32.bits_of_float (float_of_int x) in
    match t with
    | Bf16 ->
        Bytes.set_uint16_le b at
          (Int32.to_int (Int32.shift_right_logical f32 16))
    | F32 -> Bytes.set_int32_le b at f32
    | I8 -> Bytes.set_int8 b at x
    | I16 -> Bytes.set_int16_le b at x
    | I32 -> Bytes.set_int32_le b at (Int32.of_int x)
end

(* Everything that can fail while a typed kernel is made. *)
type error =
  [ Kernel.error
  | Lower.error
  | `Constant_range of Insn.Constant_range.t
  | `Register_index of Reg.Register_index.t ]

let pp_error ppf : error -> unit = function
  | #Kernel.error as e -> Kernel.pp_error ppf e
  | #Lower.error as e -> Lower.pp_error ppf e
  | `Constant_range c -> Insn.Constant_range.pp ppf c
  | `Register_index i -> Reg.Register_index.pp ppf i

(* The one place a typed error becomes a string, for the command line. *)
let to_cli r = Result.map_error (Fmt.str "%a" pp_error) (Err.export r)

(* Coral NPU's Zvt matrix instructions, from coralview's Rivet table of them:
   one encoder, made once. *)
let zvt =
  Result.map
    (Lower.Extension.make ~feature:Insn.zvt_feature)
    (Coral_isa.Coral_zvt.definition ())

(* A kernel's items made into a module under [profile]: finalized, then lowered
   with the Zvt extension. *)
let typed_module ~name ~extensions items profile =
  let widen_kernel (e : Kernel.error) : error = (e :> error) in
  let widen_lower (e : Lower.error) : error = (e :> error) in
  Err.bind
    (Err.map_error widen_kernel (Kernel.finalize ~name items))
    (fun k -> Err.map_error widen_lower (Lower.module_ ~extensions ~profile k))

type 'item module_ = 'item Asm_core.Normalized_ast.module_

(* A table-driven kernel: its items, the slots its table points at, in order,
   the input bytes of each [In] slot, and the bytes its [Out] slot must hold. *)
type t = {
  items : unit -> (Item.t list, error) Err.t;
  slots : L.Slot.t list;
  inputs : string list;
  expected : string;
  shape : string;  (** One line naming the computation. *)
  matrix : bool;
      (** Zvt instructions: the matrix core runs it, with the matrix unit on. *)
}

(* The kernel as a module under [profile]. *)
let build t profile : (Riscv32_encode.Instruction.t module_, string) result =
  Result.bind zvt (fun zvt ->
      to_cli
        (Err.bind (t.items ()) (fun items ->
             typed_module ~name:"peak" ~extensions:[ zvt ] items profile)))

let slot placement size =
  { L.Slot.align = 64L; home = Any; placement; size = Int64.of_int size }

(* {1 One instruction}

   [instr] on the accumulators v16-v31 in register groups, its sources v8 and
   v12 at [sew] and [lmul] with every element one (1.0 for [vf*], BF16 1.0 for
   [*bf16]). Widening instructions ([vw*], [vfw*]) accumulate at twice the SEW
   and LMUL. Afterwards every accumulator is stored to the [Out] slot, which
   must hold the sum in every element. *)

let iterations = 64

(* The widths of a kernel written with literals; a computed one is checked where
   it is chosen. *)
let sew_of_bits n =
  match Vtype.Sew.of_bits n with
  | Some sew -> sew
  | None -> invalid_arg (Printf.sprintf "Peak_kernels: no element width e%d" n)

let lmul_of_factor n =
  match Vtype.Lmul.of_factor n with
  | Some lmul -> lmul
  | None -> invalid_arg (Printf.sprintf "Peak_kernels: no register group m%d" n)

(* [vl] = [avl], or VLMAX into [t0] without one. *)
let vset_item ?avl sew lmul =
  Item.insn
    (Insn.Vsetvli
       {
         avl =
           (match avl with
           | Some r -> Vtype.Avl.Reg r
           | None -> Vtype.Avl.Vlmax { vl = Gpr.t0 });
         vtype =
           {
             Vtype.sew = sew_of_bits sew;
             lmul = lmul_of_factor lmul;
             tail = Vtype.Policy.Agnostic;
             mask = Vtype.Policy.Agnostic;
           };
       })

let entry_label = Label.named entry
let body_label = Label.named "peak_body"
let body_end_label = Label.named "peak_body_end"

let start_items =
  [
    Item.text;
    Item.align 4;
    Item.export ~name:entry entry_label;
    Item.label entry_label;
  ]

let body_items items =
  [ Item.export ~name:"peak_body" body_label; Item.label body_label ]
  @ items
  @ [
      Item.export ~name:"peak_body_end" body_end_label;
      Item.label body_end_label;
    ]

(* Typed instructions as items, in assembly operand order. *)
let imm12 = Imm.Imm12.of_int
let add dst lhs rhs = Item.insn (Insn.Add { dst; lhs; rhs })
let addi dst src n = Item.insn (Insn.Addi { dst; src; imm = imm12 n })
let bnez src target = Item.insn (Insn.Bnez { src; target })
let li dst n = Item.insn (Insn.Li { dst; imm = Imm.Li.of_int n })
let lw dst base n = Item.insn (Insn.Lw { dst; base; offset = imm12 n })
let mv dst src = Item.insn (Insn.Mv { dst; src })

let srli dst src n =
  Item.insn (Insn.Srli { dst; src; shamt = Imm.Shamt.of_int n })

let sw src base n = Item.insn (Insn.Sw { src; base; offset = imm12 n })
let flw dst base n = Item.insn (Insn.Flw { dst; base; offset = imm12 n })
let flh dst base n = Item.insn (Insn.Flh { dst; base; offset = imm12 n })
let vle eew dst base = Item.insn (Insn.Vle { eew; dst; base })
let vmv_v_i dst = Item.insn (Insn.Vmv_v_i { dst; imm = Imm.Simm5.zero })
let vse eew src base = Item.insn (Insn.Vse { eew; src; base })

(* What each vector-vector operation computes in this benchmark: [one] is the
   bit pattern of 1 in its inputs, and every accumulator element ends up
   holding [sum] after the loop. Exhaustive, so a new operation cannot take a
   neighbour's answer by name. *)
module Vv_op = struct
  include Insn.Vv_op

  let widening = function
    | Vfwmaccbf16 | Vwmacc -> true
    | Vadd | Vfadd | Vfmacc | Vfmul | Vmacc | Vmul -> false

  let floating = function
    | Vfadd | Vfmacc | Vfmul | Vfwmaccbf16 -> true
    | Vadd | Vmacc | Vmul | Vwmacc -> false

  let one = function
    | Vfwmaccbf16 -> 0x3f80
    | Vfadd | Vfmacc | Vfmul -> 0x3f800000
    | Vadd | Vmacc | Vmul | Vwmacc -> 1

  let sum = function
    | Vfmacc | Vfwmaccbf16 | Vmacc | Vwmacc -> iterations
    | Vadd | Vfadd -> 2
    | Vfmul | Vmul -> 1

  (* The accumulator's element type for sources of [sew] bits, if there is one:
     no half or double-precision float here. *)
  let accumulator op ~sew : Elem.t option =
    match (widening op, floating op, sew) with
    | false, false, 8 -> Some I8
    | false, false, 16 -> Some I16
    | false, false, 32 -> Some I32
    | false, true, 32 -> Some F32
    | true, false, 8 -> Some I16
    | true, false, 16 -> Some I32
    | true, true, 16 -> Some F32
    | _ -> None
end

let elementwise ~sew ~lmul op =
  let name = Vv_op.mnemonic op in
  let dsew, dlmul =
    if Vv_op.widening op then (2 * sew, 2 * lmul) else (sew, lmul)
  in
  let one = Vv_op.one op and sum = Vv_op.sum op in
  (* A benchmark error, not an assembler one: the combination has no kernel. *)
  match
    ( Vtype.Sew.of_bits sew,
      Vtype.Lmul.of_factor lmul,
      Vtype.Sew.of_bits dsew,
      Vtype.Lmul.of_factor dlmul,
      Vv_op.accumulator op ~sew )
  with
  | Some _, Some _, Some _, Some _, Some acc ->
      let items () =
        let open Err.Syntax in
        let* accumulators =
          Err.List.map
            (fun k -> Reg.Vreg.of_index (16 + (k * dlmul)))
            (List.init (16 / dlmul) Fun.id)
        in
        let* set_one = Insn.load_imm Gpr.t1 one in
        Ok
          (start_items
          @ [ lw Gpr.a6 Gpr.a0 0; vset_item dsew dlmul ]
          @ List.map vmv_v_i accumulators
          @ [ vset_item sew lmul ]
          @ List.map Item.insn set_one
          @ [
              Item.insn (Vmv_v_x { dst = Vreg.v8; src = Gpr.t1 });
              Item.insn (Vmv_v_x { dst = Vreg.v12; src = Gpr.t1 });
              li Gpr.a1 iterations;
            ]
          @ body_items
              (List.map
                 (fun dst ->
                   Item.insn (Vv { op; dst; a = Vreg.v8; b = Vreg.v12 }))
                 accumulators
              @ [ addi Gpr.a1 Gpr.a1 (-1); bnez Gpr.a1 body_label ])
          @ [ vset_item dsew dlmul ]
          @ List.concat_map
              (fun v ->
                [
                  vse (sew_of_bits dsew) v Gpr.a6;
                  addi Gpr.a6 Gpr.a6 (16 * dlmul);
                ])
              accumulators
          @ [ li Gpr.a0 0; Item.insn Ret ])
      in
      let expected = Bytes.create 256 in
      for i = 0 to (256 / Elem.bytes acc) - 1 do
        Elem.set acc expected (i * Elem.bytes acc) sum
      done;
      Ok
        {
          items;
          slots = [ slot L.Placement.Out 256 ];
          inputs = [];
          expected = Bytes.to_string expected;
          shape = Printf.sprintf "%d iterations of:" iterations;
          matrix = false;
        }
  | _ -> Error (Printf.sprintf "%s has no kernel at e%d,m%d" name sew lmul)

(* {1 Matrix multiplies}

   C[M x W] = A[M x K] B[K x W], R rows of C at a time (M = R x blocks). The
   table holds A, B and C. A is packed per block of rows, k-major (the R values
   of each k together); B and C are row-major. The inputs are small integers,
   so every product and sum is exact in each accumulator type. The loops load
   one step ahead, up to two rows past the end of B and a word past each block
   of A, so both slots carry that much padding. *)

module Gemm = struct
  (* The input types of the RVV kernels. Alphabetical. *)
  module Input = struct
    type t = Bf16 | F32 | I16 | I32 | I8

    let all = [ Bf16; F32; I16; I32; I8 ]

    let name = function
      | Bf16 -> "bf16"
      | F32 -> "f32"
      | I16 -> "i16"
      | I32 -> "i32"
      | I8 -> "i8"

    let of_name n = List.find_opt (fun t -> String.equal (name t) n) all
  end

  type shape = {
    rows : int;  (** R *)
    width : int;  (** W *)
    depth : int;  (** K *)
    blocks : int;
    src : Elem.t;
    dst : Elem.t;
    sa : int;  (** A's scale *)
    sb : int;  (** B's scale *)
  }

  let a s i k =
    ((((7 * k * k) + (13 * i) + (5 * i * k) + (3 * k)) mod 23 mod 9) - 4) * s.sa

  let b s k j =
    ((((3 * k * k) + (11 * j) + (7 * k * j) + k) mod 29 mod 7) - 3) * s.sb

  let inputs s =
    let m = s.rows * s.blocks and e = Elem.bytes s.src in
    let pa = Bytes.make ((m * s.depth * e) + 64) '\000' in
    let n = ref 0 in
    for blk = 0 to s.blocks - 1 do
      for k = 0 to s.depth - 1 do
        for r = 0 to s.rows - 1 do
          Elem.set s.src pa (!n * e) (a s ((blk * s.rows) + r) k);
          incr n
        done
      done
    done;
    let e = Elem.bytes s.src in
    let pb = Bytes.make ((s.depth + 2) * s.width * e) '\000' in
    for k = 0 to s.depth - 1 do
      for j = 0 to s.width - 1 do
        Elem.set s.src pb (((k * s.width) + j) * e) (b s k j)
      done
    done;
    [ Bytes.to_string pa; Bytes.to_string pb ]

  let product s =
    let m = s.rows * s.blocks and e = Elem.bytes s.dst in
    let c = Bytes.create (m * s.width * e) in
    for i = 0 to m - 1 do
      for j = 0 to s.width - 1 do
        let sum = ref 0 in
        for k = 0 to s.depth - 1 do
          sum := !sum + (a s i k * b s k j)
        done;
        Elem.set s.dst c (((i * s.width) + j) * e) !sum
      done
    done;
    Bytes.to_string c

  (* C is computed in column tiles of [vl] lanes, the most one vector holds:
     [lanes], or W when that is narrower (then the vector is partly idle). A W
     wider than [lanes] must be a multiple of it; each block of rows then runs
     once per tile, over the same A and the tile's columns of B and C. *)
  type tiling = {
    vl : int;  (** lanes of a tile *)
    tiles : int;
    b_row : int;  (** bytes from a row of B to the next *)
    c_row : int;  (** bytes from a row of C to the next *)
  }

  let tiling s ~lanes =
    let vl = min s.width lanes in
    if s.width mod vl <> 0 then
      invalid_arg
        (Printf.sprintf "Peak_kernels.Gemm: W = %d is not a multiple of %d"
           s.width lanes);
    {
      vl;
      tiles = s.width / vl;
      b_row = s.width * Elem.bytes s.src;
      c_row = s.width * Elem.bytes s.dst;
    }

  (* {2 Typed fragments} *)

  (* [s0] and [s1] are the caller's: [s0] counts blocks, [s1] holds B. Tiles
     also use [s2], the tiles left in this block, and [s3], B at this tile. *)
  let saved t =
    if t.tiles = 1 then [ Gpr.s0; Gpr.s1 ]
    else [ Gpr.s0; Gpr.s1; Gpr.s2; Gpr.s3 ]

  let prologue t ~a ~c =
    [ addi Gpr.sp Gpr.sp (-16) ]
    @ List.mapi (fun k r -> sw r Gpr.sp (4 * k)) (saved t)
    @ [ lw a Gpr.a0 0; lw Gpr.s1 Gpr.a0 4; lw c Gpr.a0 8 ]

  let epilogue t =
    List.mapi (fun k r -> lw r Gpr.sp (4 * k)) (saved t)
    @ [ addi Gpr.sp Gpr.sp 16; li Gpr.a0 0; Item.insn Ret ]

  (* The start of a tile, and of its first block of rows. Tiles are the outer
     loop, so moving on between blocks costs what it does with one tile. *)
  let block_head s t ~tile =
    (if t.tiles = 1 then []
     else [ li Gpr.s2 t.tiles; mv Gpr.s3 Gpr.s1; Item.label tile ])
    @ [ li Gpr.s0 s.blocks ]

  let block_start s t ~tile ~block = block_head s t ~tile @ [ Item.label block ]

  (* B at the current tile. *)
  let b_base t = if t.tiles = 1 then Gpr.s1 else Gpr.s3

  (* B's first row for the next block, loaded while the last multiply-adds and
     the stores of this one are still in flight: [per] k steps a trip. *)
  let next_b s t ~per ~eew =
    [
      mv Gpr.a2 (b_base t);
      addi Gpr.a3 Gpr.a2 t.b_row;
      li Gpr.a0 (s.depth / per);
      vle eew Vreg.v0 Gpr.a2;
    ]

  (* [r] += [n] in an [addi], or through [t3] (free here) when [n] is too far. *)
  let add_imm r n =
    if Imm.Imm12.(in_range (of_int n)) then Ok [ addi r r n ]
    else
      Err.map
        (fun set -> List.map Item.insn set @ [ add r r Gpr.t3 ])
        (Insn.load_imm Gpr.t3 n)

  (* The R accumulators stored to C at [c], then the next block, and after the
     last the next tile: A back to the first block, C to the next tile's
     columns of the first row, B on by a tile. *)
  let store_rows s t ~c ~eew ~block ~tile accumulators =
    let open Err.Syntax in
    let rows = List.length accumulators in
    let stores =
      List.concat_map (fun v -> [ vse eew v c; addi c c t.c_row ]) accumulators
      @ [ addi Gpr.s0 Gpr.s0 (-1); bnez Gpr.s0 block ]
    in
    if t.tiles = 1 then Ok stores
    else
      let* a =
        add_imm Gpr.a1 (-(s.blocks * rows * s.depth * Elem.bytes s.src))
      in
      let* c' =
        add_imm c ((t.vl * Elem.bytes s.dst) - (s.blocks * rows * t.c_row))
      in
      let* b = add_imm Gpr.s3 (t.vl * Elem.bytes s.src) in
      Ok (stores @ a @ c' @ b @ [ addi Gpr.s2 Gpr.s2 (-1); bnez Gpr.s2 tile ])

  let zeroed accumulators = List.map vmv_v_i accumulators

  (* INT8 x INT8 -> INT16. Two rows of C in e16 m8 accumulators, v16 and v24:
     each vwmacc.vx is 64 multiply-adds, four uops on the multiply-add units.
     B rows (64 bytes, e8 m4) alternate between v0 and v4. One lw carries A's
     values for two k steps (two rows each); vwmacc.vx reads the low byte of its
     scalar, so srli is all the unpacking. *)
  let i8 s () =
    let open Err.Syntax in
    let open Gpr in
    let t = tiling s ~lanes:64 in
    let vl = t0 and tile = Label.named "peak_tile" in
    let block = Label.named "peak_block" in
    let rows2 = 2 * t.b_row in
    let mac acc scalar src = Item.insn (Vwmacc_vx { acc; scalar; src }) in
    let next_b = next_b s t ~per:4 ~eew:Vtype.Sew.E8 in
    let v0 = Vreg.v0 and v4 = Vreg.v4 and v16 = Vreg.v16 and v24 = Vreg.v24 in
    let* stores =
      store_rows s t ~c:a6 ~eew:Vtype.Sew.E16 ~block ~tile [ v16; v24 ]
    in
    Ok
      (start_items @ prologue t ~a:a1 ~c:a6
      @ [ li vl t.vl ]
      @ block_head s t ~tile
      @ [ vset_item ~avl:vl 8 4 ]
      @ next_b
      @ [ lw a4 a1 0; vset_item ~avl:vl 16 8; Item.label block ]
      @ zeroed [ v16; v24 ]
      @ [ vset_item ~avl:vl 8 4 ]
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
      @ next_b
      @ [ vset_item ~avl:vl 16 8 ]
      @ stores @ epilogue t)

  (* INT16 x INT16 -> INT32. Three rows of C in e32 m8 accumulators, v8, v16
     and v24: each vwmacc.vx is 32 multiply-adds, four uops. B rows (64 bytes,
     e16 m4) alternate between v0 and v4. A comes as three words per two k
     steps (two values each); srli 16 unpacks the upper halves. *)
  let i16 s () =
    let open Err.Syntax in
    let open Gpr in
    let t = tiling s ~lanes:32 in
    let vl = t6 and tile = Label.named "peak_tile" in
    let block = Label.named "peak_block" in
    let rows2 = 2 * t.b_row in
    let next_b = next_b s t ~per:4 ~eew:Vtype.Sew.E16 in
    let mac acc scalar src = Item.insn (Vwmacc_vx { acc; scalar; src }) in
    let v0 = Vreg.v0 and v4 = Vreg.v4 and v8 = Vreg.v8 in
    let v16 = Vreg.v16 and v24 = Vreg.v24 in
    let half ~next ~cur =
      let n0, n1, n2 = next and c0, c1, c2 = cur in
      [
        lw n0 a1 12;
        lw n1 a1 16;
        lw n2 a1 20;
        vle E16 v4 a3;
        srli t3 c0 16;
        srli t4 c1 16;
        srli t5 c2 16;
        mac v8 c0 v0;
        mac v16 t3 v0;
        mac v24 c1 v0;
        addi a2 a2 rows2;
        vle E16 v0 a2;
        mac v8 t4 v4;
        mac v16 c2 v4;
        mac v24 t5 v4;
        addi a3 a3 rows2;
      ]
    in
    let cur = (a4, a5, a6) and next = (t0, t1, t2) in
    let second =
      (* the next three words, then the current ones a step ahead *)
      let c0, c1, c2 = next in
      [
        addi a1 a1 24;
        lw a4 a1 0;
        lw a5 a1 4;
        lw a6 a1 8;
        vle E16 v4 a3;
        srli t3 c0 16;
        srli t4 c1 16;
        srli t5 c2 16;
        mac v8 c0 v0;
        mac v16 t3 v0;
        mac v24 c1 v0;
        addi a2 a2 rows2;
        vle E16 v0 a2;
        mac v8 t4 v4;
        mac v16 c2 v4;
        mac v24 t5 v4;
        addi a3 a3 rows2;
      ]
    in
    let* stores =
      store_rows s t ~c:a7 ~eew:Vtype.Sew.E32 ~block ~tile [ v8; v16; v24 ]
    in
    Ok
      (start_items @ prologue t ~a:a1 ~c:a7
      @ [ li vl t.vl ]
      @ block_head s t ~tile
      @ [ vset_item ~avl:vl 16 4 ]
      @ next_b
      @ [
          lw a4 a1 0;
          lw a5 a1 4;
          lw a6 a1 8;
          vset_item ~avl:vl 32 8;
          Item.label block;
        ]
      @ zeroed [ v8; v16; v24 ]
      @ [ vset_item ~avl:vl 16 4 ]
      @ body_items
          (half ~next ~cur @ second @ [ addi a0 a0 (-1); bnez a0 body_label ])
      @ next_b
      @ [ vset_item ~avl:vl 32 8 ]
      @ stores @ epilogue t)

  (* BF16 x BF16 -> FP32. Three rows of C in e32 m8 accumulators: each
     vfwmaccbf16.vf is 32 multiply-adds, four uops. B rows (64 bytes, e16 m4)
     alternate between v0 and v4, A's values load by flh a k step ahead. *)
  let bf16 s () =
    let open Err.Syntax in
    let open Gpr in
    let open Reg.Fpr in
    let t = tiling s ~lanes:32 in
    let vl = t0 and tile = Label.named "peak_tile" in
    let block = Label.named "peak_block" in
    let rows2 = 2 * t.b_row in
    let mac acc scalar src = Item.insn (Vfwmaccbf16_vf { acc; scalar; src }) in
    let next_b = next_b s t ~per:2 ~eew:Vtype.Sew.E16 in
    let v0 = Vreg.v0 and v4 = Vreg.v4 and v8 = Vreg.v8 in
    let v16 = Vreg.v16 and v24 = Vreg.v24 in
    let* stores =
      store_rows s t ~c:a7 ~eew:Vtype.Sew.E32 ~block ~tile [ v8; v16; v24 ]
    in
    Ok
      (start_items @ prologue t ~a:a1 ~c:a7
      @ [ li vl t.vl ]
      @ block_head s t ~tile
      @ [ vset_item ~avl:vl 16 4 ]
      @ next_b
      @ [
          flh fa0 a1 0;
          flh fa1 a1 2;
          flh fa2 a1 4;
          vset_item ~avl:vl 32 8;
          Item.label block;
        ]
      @ zeroed [ v8; v16; v24 ]
      @ [ vset_item ~avl:vl 16 4 ]
      @ body_items
          [
            flh fa3 a1 6;
            flh fa4 a1 8;
            flh fa5 a1 10;
            vle E16 v4 a3;
            mac v8 fa0 v0;
            mac v16 fa1 v0;
            mac v24 fa2 v0;
            addi a1 a1 12;
            flh fa0 a1 0;
            flh fa1 a1 2;
            flh fa2 a1 4;
            addi a2 a2 rows2;
            vle E16 v0 a2;
            mac v8 fa3 v4;
            mac v16 fa4 v4;
            mac v24 fa5 v4;
            addi a3 a3 rows2;
            addi a0 a0 (-1);
            bnez a0 body_label;
          ]
      @ next_b
      @ [ vset_item ~avl:vl 32 8 ]
      @ stores @ epilogue t)

  (* The two instances of {!word}: INT32 through the integer registers, FP32
     through the floating-point ones. The register class is the type
     parameter, so an integer register cannot reach [flw] or [vfmacc.vf]. *)
  module Scalar_ops = struct
    type 'r t = {
      load : 'r -> Gpr.t -> int -> Item.t;  (** [load dst base offset] *)
      mac : acc:Vreg.t -> scalar:'r -> src:Vreg.t -> Item.t;
      cur : 'r * 'r * 'r;
      next : 'r * 'r * 'r;
    }

    let float =
      let open Reg.Fpr in
      {
        load = flw;
        mac =
          (fun ~acc ~scalar ~src -> Item.insn (Vfmacc_vf { acc; scalar; src }));
        cur = (fa4, fa5, fa6);
        next = (ft1, ft2, ft3);
      }

    let int =
      let open Gpr in
      {
        load = lw;
        mac =
          (fun ~acc ~scalar ~src -> Item.insn (Vmacc_vx { acc; scalar; src }));
        cur = (a4, a5, a6);
        next = (t1, t2, t3);
      }
  end

  (* 32-bit inputs, INT32 by lw and vmacc.vx or FP32 by flw and vfmacc.vf.
     Three rows of C in e32 m8 accumulators, v8, v16 and v24: each
     multiply-add is 32 lanes, four uops. B rows (128 bytes, e32 m8) load into
     v0, A's values a k step ahead. The scalar loads come before the vector
     load: a scalar load after a vector load waits for it. *)
  let word (ops : _ Scalar_ops.t) s () =
    let open Err.Syntax in
    let open Gpr in
    let t = tiling s ~lanes:32 in
    let vl = t0 and tile = Label.named "peak_tile" in
    let block = Label.named "peak_block" in
    let rows1 = t.b_row in
    let v0 = Vreg.v0 and v8 = Vreg.v8 in
    let v16 = Vreg.v16 and v24 = Vreg.v24 in
    let loads (r0, r1, r2) off =
      [ ops.load r0 a1 off; ops.load r1 a1 (off + 4); ops.load r2 a1 (off + 8) ]
    in
    let macs (r0, r1, r2) =
      [
        ops.mac ~acc:v8 ~scalar:r0 ~src:v0;
        ops.mac ~acc:v16 ~scalar:r1 ~src:v0;
        ops.mac ~acc:v24 ~scalar:r2 ~src:v0;
      ]
    in
    let* stores =
      store_rows s t ~c:a7 ~eew:Vtype.Sew.E32 ~block ~tile [ v8; v16; v24 ]
    in
    Ok
      (start_items @ prologue t ~a:a1 ~c:a7
      @ [ li vl t.vl; vset_item ~avl:vl 32 8 ]
      @ block_start s t ~tile ~block
      @ zeroed [ v8; v16; v24 ]
      @ [ mv a2 (b_base t); li a0 (s.depth / 2) ]
      @ loads ops.cur 0
      @ body_items
          (loads ops.next 12
          @ [ vle E32 v0 a2 ]
          @ macs ops.cur
          @ [ addi a2 a2 rows1; addi a1 a1 24 ]
          @ loads ops.cur 0
          @ [ vle E32 v0 a2 ]
          @ macs ops.next
          @ [ addi a2 a2 rows1; addi a0 a0 (-1); bnez a0 body_label ])
      @ stores @ epilogue t)

  let make s items_of =
    let m = s.rows * s.blocks in
    let inputs = inputs s in
    let expected = product s in
    {
      items = items_of s;
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
          "C[%dx%d] %s = A[%dx%d] %s x B[%dx%d] %s, %d rows at a time; inner \
           loop:"
          m s.width (Elem.name s.dst) m s.depth (Elem.name s.src) s.depth
          s.width (Elem.name s.src) s.rows;
      matrix = false;
    }
end

(* Each RVV kernel's shape, with [width] if given, and its items. *)
let gemm_kernel ?width (input : Gemm.Input.t) =
  let open Gemm in
  let shape ~rows ~width:lanes ~depth ~blocks ~src ~dst ~sa ~sb () =
    let width = Option.value width ~default:lanes in
    { rows; width; depth; blocks; src; dst; sa; sb }
  in
  match input with
  | Bf16 ->
      ( shape ~rows:3 ~width:32 ~depth:128 ~blocks:3 ~src:Bf16 ~dst:F32 ~sa:1
          ~sb:1 (),
        bf16 )
  | F32 ->
      ( shape ~rows:3 ~width:32 ~depth:64 ~blocks:3 ~src:F32 ~dst:F32 ~sa:1
          ~sb:1 (),
        word Scalar_ops.float )
  | I8 ->
      ( shape ~rows:2 ~width:64 ~depth:128 ~blocks:4 ~src:I8 ~dst:I16 ~sa:1
          ~sb:1 (),
        i8 )
  | I16 ->
      ( shape ~rows:3 ~width:32 ~depth:128 ~blocks:3 ~src:I16 ~dst:I32 ~sa:37
          ~sb:53 (),
        i16 )
  | I32 ->
      ( shape ~rows:3 ~width:32 ~depth:64 ~blocks:3 ~src:I32 ~dst:I32 ~sa:1009
          ~sb:997 (),
        word Scalar_ops.int )

(* The shape of [gemm ?width input], which the Zvt kernels can take too. *)
let gemm_shape ?width input = fst (gemm_kernel ?width input)

(* [gemm I8], or with a width [gemm ~width:48 I8]: narrower than a vector holds
   leaves it partly idle, a multiple of it runs the rows once per column tile. *)
let gemm ?width input =
  let s, items = gemm_kernel ?width input in
  Gemm.make s items
