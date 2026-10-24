module A = Machine_rivet_riscv32.Rv32_asm
module C = Contraction

module Mutation = struct
  type t = A_row_stride | Tail_vl
end

let symbol = C.symbol
let i = A.insn
let r = A.reg

exception Refused of string

let refuse why = raise (Refused why)

(* A byte offset as an [int], refused past 31 bits. *)
let bytes (n : int64) =
  let b = Int64.mul 4L n in
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

let items ?mutation (d : C.t) () =
  (match d.C.seed with
  | C.Seed.Zero -> ()
  | C.Seed.Tensor -> refuse "a contraction seeded from its output");
  let n = Int64.to_int d.C.n and m = Int64.to_int d.C.m in
  let k = Int64.to_int d.C.k and batch = Int64.to_int d.C.batch in
  let fused = d.C.numerics = C.Numerics.Fused_sequential in
  let rows = if fused then 3 else 2 in
  let accs = [ "v8"; "v16"; "v24" ] in
  let fregs = [ "fa0"; "fa1"; "fa2" ] in
  let a_outer =
    bytes d.C.a.C.Operand.outer
    + if mutation = Some Mutation.A_row_stride then 4 else 0
  and a_k = bytes d.C.a.C.Operand.k in
  let b_outer = bytes d.C.b.C.Operand.outer and b_k = bytes d.C.b.C.Operand.k in
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
    let step =
      List.map2 (fun f p -> i "flw" [ r f; A.mem p 0 ]) fs ptrs
      @ (if b_outer = 4 then [ i "vle32.v" [ r "v0"; A.mem "a7" 0 ] ]
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
    @ add_const "a0" (bytes d.C.a.C.Operand.batch)
    @ add_const "a1" (bytes d.C.b.C.Operand.batch)
    @ add_const "a2" (bytes d.C.c.C.Output.batch)
  in
  let name = symbol d in
  [ A.text; A.align 4; A.global name; A.label name ]
  @ [ i "addi" [ r "sp"; r "sp"; A.imm (-16) ] ]
  @ List.mapi
      (fun j s -> i "sw" [ r s; A.mem "sp" (4 * j) ])
      [ "s0"; "s1"; "s2"; "s3" ]
  @ A.load_imm "s1" b_outer @ A.load_imm "s2" c_col
  @ loop batch ~counter:"t6" per_batch
  @ List.mapi
      (fun j s -> i "lw" [ r s; A.mem "sp" (4 * j) ])
      [ "s0"; "s1"; "s2"; "s3" ]
  @ [
      i "addi" [ r "sp"; r "sp"; A.imm 16 ];
      i "li" [ r "a0"; A.imm 0 ];
      i "ret" [];
    ]

let module_ ?mutation d =
  match items ?mutation d () with
  | items -> A.module_ (symbol d) (fun () -> items)
  | exception Refused why -> Error why
