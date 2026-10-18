(* Per-form conformance under qemu: every admitted RV32 form instance, emitted
   by the route's own mapping, runs over boundary and seeded operands in one
   process each, and its results (and, for a store, the memory it wrote) must
   equal [Rv32_sem]'s bit for bit — a NaN included, since the semantics model
   the canonical NaN. Mutations are in [Conformance_mutation_test]. *)

open Machine_ir
open Machine_target_riscv32
module O = Rv32_op
module F = Machine_rivet_riscv32_mir.Rv32_mir_form
module A = Machine_rivet_riscv32.Rv32_asm
module H = Machine_rivet_riscv32.Rv32_harness
module E = Machine_interp.Mir_sel_env
module Mem = Machine_interp.Mir_memory
module Loc = Mir_phys.Loc

(* {1 Operands} *)

type reg = G of int | Fr of int

let view = function G k -> Rv32_reg.x k | Fr k -> Rv32_reg.f k
let rname r = (view r).Mir_target.View.name
let a k = G (10 + k)
let fa k = Fr (10 + k)
let base = G 18 (* s2: the vector's memory area *)

let ints =
  List.map
    (fun x -> Int64.logand x 0xFFFF_FFFFL)
    [
      0L;
      1L;
      2L;
      -1L;
      -2L;
      0x7FFF_FFFFL;
      0x8000_0000L;
      0x8000_0001L;
      2047L;
      2048L;
      -2048L;
      -2049L;
      0xFFFFL;
      0x10000L;
      0xFFL;
      0x100L;
      31L;
      32L;
      0x1234_5678L;
      0xDEAD_BEEFL;
    ]

let floats =
  [
    0x0000_0000L;
    0x8000_0000L;
    0x3F80_0000L;
    0xBF80_0000L;
    0x3F00_0000L;
    0xBF00_0000L;
    0x3FC0_0000L;
    0x4B00_0000L;
    0x4AFF_FFFFL;
    0xCB00_0001L;
    0x4F00_0000L;
    0xCF00_0000L;
    0x4EFF_FFFFL;
    0x7F7F_FFFFL;
    0x0080_0000L;
    0x0000_0001L;
    0x807F_FFFFL;
    0x7F80_0000L;
    0xFF80_0000L;
    0x7FC0_0000L;
    0xFFC0_0001L;
    0x7F80_0001L;
    0x4040_0000L;
    0x3EAA_AAABL;
  ]

let pairs xs = List.concat_map (fun x -> List.map (fun y -> [ x; y ]) xs) xs
let singles xs = List.map (fun x -> [ x ]) xs

let triples xs =
  let xs = List.filteri (fun i _ -> i mod 2 = 0) xs in
  List.concat_map (fun x -> List.map (fun p -> x :: p) (pairs xs)) xs

(* {1 Instances} *)

type instance = {
  name : string;
  op : O.t;
  uses : reg list;  (** the form's operand registers, in order *)
  defs : reg list;
  vectors : int64 list list;
      (** each vector's operand words, the base register's excluded *)
  memory : bool;  (** the form reads or writes the vector's memory area *)
}

type test_instance = {
  tname : string;
  test : O.test;
  tuses : reg list;
  tvectors : int64 list list;
}

let v k ty = { Mir_value.id = Mir_id.Value.of_int k; ty }
let gv k = v k Mir_type.i32
let fv k = v k Mir_type.F32
let pv k = v k Mir_type.Pred

let inst ?(memory = false) name op uses defs vectors =
  { name; op; uses; defs; vectors; memory }

let alu =
  List.map
    (fun o ->
      inst
        ("alu " ^ O.Alu.name o)
        (O.Alu (o, gv 0, gv 1))
        [ a 1; a 2 ]
        [ a 4 ]
        (pairs ints))
    O.Alu.[ Add; And; Div; Mul; Mulhu; Or; Rem; Sll; Slt; Sltu; Srl; Sub; Xor ]

let alui =
  List.concat_map
    (fun (o, ks) ->
      List.map
        (fun k ->
          inst
            (Fmt.str "alui %s %Ld" (O.Alui.name o) k)
            (O.Alui (o, gv 0, k))
            [ a 1 ]
            [ a 4 ]
            (singles ints))
        ks)
    O.Alui.
      [
        (Andi, [ 0L; 255L; -1L; -2048L; 2047L ]);
        (Ori, [ 0L; 1L; -2048L; 2047L ]);
        (Xori, [ 1L; -1L; 0x555L ]);
        (Slli, [ 0L; 1L; 16L; 31L ]);
        (Srli, [ 0L; 1L; 16L; 31L ]);
        (Srai, [ 0L; 1L; 16L; 31L ]);
        (Slti, [ 0L; 1L; -1L; 2047L; -2048L ]);
        (Sltiu, [ 0L; 1L; -1L; 2047L; -2048L ]);
      ]

let constants =
  List.map
    (fun k ->
      inst (Fmt.str "addi %Ld" k)
        (O.Addi (gv 0, k))
        [ a 1 ]
        [ a 4 ]
        (singles ints))
    [ 0L; 1L; -1L; 2047L; -2048L ]
  @ List.map
      (fun k ->
        inst (Fmt.str "add.wide %Lx" k)
          (O.Add_wide (gv 0, k))
          [ a 1 ]
          [ a 4 ]
          (singles ints))
      [ 0x800L; 4096L; 0x1234_5678L; -0x7FFF_F800L; 0x7FFF_FFFFL ]
  @ List.map
      (fun k ->
        inst (Fmt.str "li %Ld" k) (O.Li (Mir_type.i32, k)) [] [ a 4 ] [ [] ])
      [ 0L; 1L; -1L; 2047L; -2048L ]
  @ List.map
      (fun k -> inst (Fmt.str "lui %Lx" k) (O.Lui k) [] [ a 4 ] [ [] ])
      [ 0L; 1L; 0x80000L; 0xFFFFFL; 0x12345L ]

let moves =
  [
    inst "mv" (O.Mv (gv 0)) [ a 1 ] [ a 4 ] (singles ints);
    inst "zext"
      (O.Zext (gv 0))
      [ a 1 ]
      [ a 4 ]
      (singles [ 0L; 1L; 0xFFL; 0xFFFFL ]);
    inst "sext.i8"
      (O.Sext (Mir_width.W8, gv 0))
      [ a 1 ]
      [ a 4 ]
      (singles [ 0L; 1L; 0x7FL; 0x80L; 0xFFL ]);
    inst "sext.i16"
      (O.Sext (Mir_width.W16, gv 0))
      [ a 1 ]
      [ a 4 ]
      (singles [ 0L; 1L; 0x7FFFL; 0x8000L; 0xFFFFL ]);
    inst "trunc.i8"
      (O.Trunc (Mir_width.W8, gv 0))
      [ a 1 ]
      [ a 4 ]
      (singles ints);
    inst "trunc.i16"
      (O.Trunc (Mir_width.W16, gv 0))
      [ a 1 ]
      [ a 4 ]
      (singles ints);
  ]

(* a select's destination in every relation to its operands *)
let selects =
  let preds = [ [ 0L ]; [ 1L ] ] in
  let vs xs =
    List.concat_map (fun p -> List.map (fun x -> p @ x) (pairs xs)) preds
  in
  List.map
    (fun (n, rd) ->
      inst ("sel, rd " ^ n)
        (O.Sel (pv 0, gv 1, gv 2))
        [ a 1; a 2; a 3 ]
        [ rd ]
        (vs [ 5L; 0xFFFF_FFFFL; 0L ]))
    [ ("apart", a 4); ("= then", a 2); ("= else", a 3); ("= predicate", a 1) ]
  @ List.map
      (fun (n, rd) ->
        inst ("fsel, rd " ^ n)
          (O.Fsel (pv 0, fv 1, fv 2))
          [ a 1; fa 2; fa 3 ]
          [ rd ]
          (vs [ 0x3F80_0000L; 0x7FC0_0000L; 0x8000_0000L ]))
      [ ("apart", fa 4); ("= then", fa 2); ("= else", fa 3) ]

let fp =
  List.map
    (fun o ->
      inst (O.Fop.name o)
        (O.Fbin (o, fv 0, fv 1))
        [ fa 1; fa 2 ]
        [ fa 4 ]
        (pairs floats))
    O.Fop.[ Add; Div; Max; Min; Mul; Sub ]
  @ List.map
      (fun o ->
        inst (O.Fcmp.name o)
          (O.Fcmp (o, fv 0, fv 1))
          [ fa 1; fa 2 ]
          [ a 4 ]
          (pairs floats))
      O.Fcmp.[ Eq; Le; Lt ]
  @ List.map
      (fun o ->
        inst (O.Sgnj.name o)
          (O.Fsgnj (o, fv 0, fv 1))
          [ fa 1; fa 2 ]
          [ fa 4 ]
          (pairs floats))
      O.Sgnj.[ J; Jn; Jx ]
  @ [
      inst "fsqrt.s" (O.Fsqrt (fv 0)) [ fa 1 ] [ fa 4 ] (singles floats);
      inst "fmadd.s"
        (O.Fmadd (fv 0, fv 1, fv 2))
        [ fa 1; fa 2; fa 3 ]
        [ fa 4 ]
        (triples floats);
      inst "fcvt.s.w"
        (O.Fcvt_s_w (gv 0))
        [ a 1 ]
        [ fa 4 ]
        (singles (ints @ [ 0x0100_0001L; 0xFEFF_FFFFL; 0x7FFF_FFC0L ]));
      inst "fcvt.s.wu"
        (O.Fcvt_s_wu (gv 0))
        [ a 1 ]
        [ fa 4 ]
        (singles
           (ints @ [ 0x0100_0001L; 0xFEFF_FFFFL; 0xFFFF_FF80L; 0xFFFF_FF7FL ]));
      inst "fcvt.w.s" (O.Fcvt_w_s (fv 0)) [ fa 1 ] [ a 4 ] (singles floats);
      inst "fmv.w.x"
        (O.Fmv_w_x (gv 0))
        [ a 1 ]
        [ fa 4 ]
        (singles (ints @ floats));
      inst "fmv.x.w" (O.Fmv_x_w (fv 0)) [ fa 1 ] [ a 4 ] (singles floats);
    ]

(* each vector's 16-byte memory area starts as the bytes 0x10..0x1f *)
let accesses =
  let offs = [ 0L; 1L; 3L; 4L; 6L; 12L ] in
  List.concat_map
    (fun (m, offsets) ->
      List.map
        (fun k ->
          inst ~memory:true
            (Fmt.str "%s %Ld" (O.Msz.load m) k)
            (O.Load (m, v 0 Mir_type.Ptr, k))
            [ base ]
            [ a 4 ]
            [ [] ])
        offsets
      @ List.map
          (fun k ->
            inst ~memory:true
              (Fmt.str "%s %Ld" (O.Msz.store m) k)
              (O.Store (m, v 0 Mir_type.Ptr, k, gv 1))
              [ base; a 1 ]
              []
              (singles [ 0xFFFF_FFFFL; 0x1234_5678L; 0L ]))
          offsets)
    O.Msz.[ (B, offs); (H, [ 0L; 2L; 6L; 12L ]); (W, [ 0L; 4L; 12L ]) ]
  @ [
      inst ~memory:true "flw 4"
        (O.Flw (v 0 Mir_type.Ptr, 4L))
        [ base ]
        [ fa 4 ]
        [ [] ];
      inst ~memory:true "fsw 8"
        (O.Fsw (v 0 Mir_type.Ptr, 8L, fv 1))
        [ base; fa 1 ]
        [] (singles floats);
      inst ~memory:true "sw.pair 4"
        (O.Sw_pair (v 0 Mir_type.Ptr, 4L, gv 1, gv 2))
        [ base; a 1; a 2 ]
        []
        (pairs [ 0L; 0xFFFF_FFFFL; 0x1234_5678L ]);
    ]

let instances = alu @ alui @ constants @ moves @ selects @ fp @ accesses

let tests =
  List.map
    (fun c ->
      {
        tname = O.Bcond.name c;
        test = O.Bcmp (c, gv 0, gv 1);
        tuses = [ a 1; a 2 ];
        tvectors = pairs ints;
      })
    O.Bcond.[ Eq; Ge; Geu; Lt; Ltu; Ne ]
  @ [
      {
        tname = "beqz";
        test = O.Beqz (gv 0);
        tuses = [ a 1 ];
        tvectors = singles ints;
      };
      {
        tname = "bnez";
        test = O.Bnez (gv 0);
        tuses = [ a 1 ];
        tvectors = singles ints;
      };
    ]

(* {1 The model's expectation} *)

exception Defect

let area_bytes = 16

let area j =
  String.init area_bytes (fun i -> Char.chr (0x10 + i + (j land 0x3 * 0x10)))

(* What [Rv32_sem] says one vector produces: its result words and, for a
   memory form, the area's bytes after. *)
let expected ~(sem : E.t -> O.t -> Machine_interp.Mir_datum.t list) i j words =
  let memory = Mem.create () in
  let key =
    Option.get (Mem.alloc memory ~size:(Int64.of_int area_bytes) ~align:16L ())
  in
  Mem.write_string memory key ~offset:0L (area j);
  let p = Mem.pointer memory key ~lo:0L ~hi:(Int64.of_int area_bytes) in
  let words = Array.of_list words in
  let get (x : Mir_value.t) =
    let k = Mir_id.Value.to_int x.Mir_value.id in
    if i.memory && k = 0 then Machine_interp.Mir_datum.Ptr p
    else Machine_interp.Mir_datum.Bits words.(if i.memory then k - 1 else k)
  in
  let env =
    {
      E.get;
      memory;
      view = (fun _ -> None);
      defect = (fun _ -> raise Defect);
      call = (fun _ _ -> raise Defect);
    }
  in
  let results =
    List.map
      (function
        | Machine_interp.Mir_datum.Bits b -> Int64.logand b 0xFFFF_FFFFL
        | _ -> raise Defect)
      (sem env i.op)
  in
  let bytes =
    if i.memory then
      Some
        (String.init area_bytes (fun n ->
             match
               (Mem.read_bytes memory key ~offset:(Int64.of_int n) ~n:1).(0)
             with
             | Some b -> Char.chr b
             | None -> '\000'))
    else None
  in
  (results, bytes)

(* {1 The process} *)

let label_counter = ref 0

let fresh_label () =
  incr label_counter;
  Printf.sprintf ".Lc%d" !label_counter

let env mutation =
  Err.Escape.with_escape (fun esc ->
      {
        F.mutation;
        esc;
        reference = (fun _ -> None);
        table_slot = (fun _ -> None);
        fresh_label;
      })
  |> Err.payload |> Result.get_ok

let load_word r off =
  match r with
  | G _ -> F.i "lw" [ A.reg (rname r); A.mem "s0" off ]
  | Fr _ -> F.i "flw" [ A.reg (rname r); A.mem "s0" off ]

let store_word r off =
  match r with
  | G _ -> F.i "sw" [ A.reg (rname r); A.mem "s1" off ]
  | Fr _ -> F.i "fsw" [ A.reg (rname r); A.mem "s1" off ]

(* One process over every vector: inputs from slot 0, results to slot 1, the
   memory areas in slot 2; s0-s2 walk them and are restored. *)
let process ~body ~inputs ~outputs n =
  let per_vector j =
    List.mapi (fun k r -> load_word r (4 * k)) inputs
    @ body j
    @ List.mapi (fun k r -> store_word r (4 * k)) outputs
    @ [
        F.i "addi" [ A.reg "s0"; A.reg "s0"; A.imm (4 * List.length inputs) ];
        F.i "addi" [ A.reg "s1"; A.reg "s1"; A.imm (4 * List.length outputs) ];
        F.i "addi" [ A.reg "s2"; A.reg "s2"; A.imm area_bytes ];
      ]
  in
  let items =
    [
      F.i "addi" [ A.reg "sp"; A.reg "sp"; A.imm (-16) ];
      F.i "sw" [ A.reg "s0"; A.mem "sp" 0 ];
      F.i "sw" [ A.reg "s1"; A.mem "sp" 4 ];
      F.i "sw" [ A.reg "s2"; A.mem "sp" 8 ];
      F.i "lw" [ A.reg "s0"; A.mem "a0" 0 ];
      F.i "lw" [ A.reg "s1"; A.mem "a0" 4 ];
      F.i "lw" [ A.reg "s2"; A.mem "a0" 8 ];
    ]
    @ List.concat (List.init n per_vector)
    @ [
        F.i "lw" [ A.reg "s0"; A.mem "sp" 0 ];
        F.i "lw" [ A.reg "s1"; A.mem "sp" 4 ];
        F.i "lw" [ A.reg "s2"; A.mem "sp" 8 ];
        F.i "addi" [ A.reg "sp"; A.reg "sp"; A.imm 16 ];
        F.i "li" [ A.reg "a0"; A.imm 0 ];
        F.i "ret" [];
      ]
  in
  Machine_rivet_riscv32_mir.Rv32_mir_module.(render (lay_out items))

let run ~items ~inputs ~outputs vectors =
  let n = List.length vectors in
  let words xs =
    String.concat ""
      (List.map
         (fun x ->
           let b = Bytes.create 4 in
           Bytes.set_int32_le b 0 (Int64.to_int32 x);
           Bytes.to_string b)
         xs)
  in
  match
    A.module_ "conformance" (fun () ->
        [ A.text; A.align 4; A.global "conformance"; A.label "conformance" ]
        @ items)
  with
  | Error what -> Error ("no form for " ^ what)
  | Ok m -> (
      let slot size placement =
        { H.size = max 4 size; align = 16; placement }
      in
      let slots =
        [
          slot (4 * n * max 1 inputs) H.Placement.In;
          slot (4 * n * max 1 outputs) H.Placement.Out;
          slot (area_bytes * n) H.Placement.Out;
        ]
      in
      match H.prepare ~entry:"conformance" ~kernel:[ m ] slots with
      | Error e -> Error e
      | Ok p -> (
          let bound = function
            | 0 -> Some (words (List.concat vectors))
            | 2 -> Some (String.concat "" (List.init n area))
            | _ -> None
          in
          match H.run p ~bound with
          | Error e -> Error e
          | Ok o -> Ok (List.assoc 1 o.H.outputs, List.assoc 2 o.H.outputs)))

let word s k =
  Int64.logand (Int64.of_int32 (String.get_int32_le s (4 * k))) 0xFFFF_FFFFL

(* Disagreements of one instance: how many vectors, and the first. *)
let check_instance ?mutation ~sem i =
  let e = env mutation in
  let uses = List.map (fun r -> Loc.Reg (view r)) i.uses
  and defs = List.map (fun r -> Loc.Reg (view r)) i.defs in
  let inputs = List.filter (fun r -> r <> base) i.uses in
  let body _ = F.items e i.op ~uses ~defs in
  let n = List.length i.vectors in
  match
    run
      ~items:(process ~body ~inputs ~outputs:i.defs n)
      ~inputs:(List.length inputs) ~outputs:(List.length i.defs) i.vectors
  with
  | Error e -> Some (Fmt.str "%s: %s" i.name e)
  | Ok (out, areas) -> (
      let bad = ref [] in
      List.iteri
        (fun j words ->
          let results, bytes = expected ~sem i j words in
          let got =
            List.mapi
              (fun k _ -> word out ((j * List.length i.defs) + k))
              results
          in
          let area_got = String.sub areas (j * area_bytes) area_bytes in
          if
            got <> results
            || match bytes with Some b -> b <> area_got | None -> false
          then
            bad :=
              Fmt.str "%a -> %a, expected %a"
                Fmt.(list ~sep:(any " ") (fmt "%Lx"))
                words
                Fmt.(list ~sep:(any " ") (fmt "%Lx"))
                got
                Fmt.(list ~sep:(any " ") (fmt "%Lx"))
                results
              :: !bad)
        i.vectors;
      match List.rev !bad with
      | [] -> None
      | first :: _ ->
          Some
            (Fmt.str "%s: %d of %d differ, first %s" i.name (List.length !bad) n
               first))

let check_test ?mutation t =
  let e = env mutation in
  let uses = List.map (fun r -> Loc.Reg (view r)) t.tuses in
  let body _ =
    let taken = fresh_label () and over = fresh_label () in
    [
      F.i "li" [ A.reg "a4"; A.imm 0 ];
      F.branch e t.test ~uses ~label:taken ~inverted:false;
      F.i "j" [ A.sym over ];
      F.Item.Label taken;
      F.i "li" [ A.reg "a4"; A.imm 1 ];
      F.Item.Label over;
    ]
  in
  let n = List.length t.tvectors in
  match
    run
      ~items:(process ~body ~inputs:t.tuses ~outputs:[ a 4 ] n)
      ~inputs:(List.length t.tuses) ~outputs:1 t.tvectors
  with
  | Error e -> Some (Fmt.str "%s: %s" t.tname e)
  | Ok (out, _) ->
      let bad =
        List.filteri
          (fun j words ->
            let ws = Array.of_list words in
            let env =
              {
                E.get =
                  (fun x ->
                    Machine_interp.Mir_datum.Bits
                      ws.(Mir_id.Value.to_int x.Mir_value.id));
                memory = Mem.create ();
                view = (fun _ -> None);
                defect = (fun _ -> raise Defect);
                call = (fun _ _ -> raise Defect);
              }
            in
            let want = if Rv32_sem.test env t.test then 1L else 0L in
            not (Int64.equal want (word out j)))
          t.tvectors
      in
      if bad = [] then None
      else Some (Fmt.str "%s: %d of %d differ" t.tname (List.length bad) n)

let summary ?mutation ~sem () =
  let failures =
    List.filter_map (check_instance ?mutation ~sem) instances
    @ List.filter_map (check_test ?mutation) tests
  in
  let vectors =
    List.fold_left (fun n i -> n + List.length i.vectors) 0 instances
    + List.fold_left (fun n t -> n + List.length t.tvectors) 0 tests
  in
  (List.length instances + List.length tests, vectors, failures)

let%expect_test "every form agrees with its semantics under qemu" =
  let forms, vectors, failures = summary ~sem:Rv32_sem.exec () in
  Fmt.pr "%d form instances, %d vectors@." forms vectors;
  List.iter (Fmt.pr "%s@.") failures;
  [%expect {| 136 form instances, 17619 vectors |}]
