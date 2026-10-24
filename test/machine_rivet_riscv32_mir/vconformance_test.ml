(* Per-form conformance of the Zve32f vector forms under qemu: every instance,
   emitted by the route's own mapping, runs in one process per VLEN (128, 512,
   1024, tail and masked-off elements filled with ones) over lanes made of the
   scalar suite's boundary values, and its results — scalar, vector or the
   memory it wrote — must equal [Rv32_sem]'s bit for bit. Select and insert
   destinations alias every operand; strided accesses take strides 0 to 12. *)

open Machine_ir
open Machine_target_riscv32
module O = Rv32_op
module Vo = Rv32_vop
module F = Machine_rivet_riscv32_mir.Rv32_mir_form
module A = Machine_rivet_riscv32.Rv32_asm
module H = Machine_rivet_riscv32.Rv32_harness
module E = Machine_interp.Mir_sel_env
module Mem = Machine_interp.Mir_memory
module D = Machine_interp.Mir_datum
module Loc = Mir_phys.Loc
module C = Conformance_test

type reg = G of int | Fr of int | Vr of int | Vn of int * Mir_width.t
(* [Vn]: a vector register of i8 or i16 lanes, loaded as its four words *)

let view = function
  | G k -> Rv32_reg.x k
  | Fr k -> Rv32_reg.f k
  | Vr k | Vn (k, _) -> Rv32_reg.v k

let rname r = (view r).Mir_target.View.name
let width = function G _ | Fr _ -> 1 | Vr _ | Vn _ -> 4
let base = G 18

type instance = {
  name : string;
  op : Vo.t;
  uses : reg list;
  defs : reg list;
  vectors : int64 list list;  (** each vector's operand words, base excluded *)
  memory : bool;
}

let inst ?(memory = false) name op uses defs vectors =
  { name; op; uses; defs; vectors; memory }

let v k ty = { Mir_value.id = Mir_id.Value.of_int k; ty }
let vf k = v k Vo.f32x4
let vi k = v k Vo.i32x4
let vm k = v k Vo.mask4
let gv k = v k Mir_type.i32
let fv k = v k Mir_type.F32

(* a list cut into vectors of [n] words, the last padded from the front *)
let rec chunks n xs =
  match xs with
  | [] -> []
  | _ ->
      let take = List.filteri (fun i _ -> i < n) xs
      and rest = List.filteri (fun i _ -> i >= n) xs in
      let take =
        if List.length take = n then take
        else take @ List.filteri (fun i _ -> i < n - List.length take) C.floats
      in
      take :: chunks n rest

(* operand vectors of four lanes each, [k] operands a vector *)
(* [n] at a time, a short last group dropped *)
let rec group n xs =
  if List.length xs < n then []
  else
    List.filteri (fun i _ -> i < n) xs
    :: group n (List.filteri (fun i _ -> i >= n) xs)

let lanes k xs = List.map List.concat (group k (chunks 4 xs))
let float_pairs = lanes 2 (List.concat (C.pairs C.floats))
let float_singles = chunks 4 C.floats
let float_triples = lanes 3 (List.concat (C.triples C.floats))
let int_pairs = lanes 2 (List.concat (C.pairs C.ints))
let masks = [ 0L; 0xFFFF_FFFFL ]

let mask_lanes =
  List.concat_map (fun a -> List.map (fun b -> [ a; b; b; a ]) masks) masks

let fop =
  List.map
    (fun o ->
      inst (Rv32_kind.Fop.vname o)
        (Vo.Vfbin (o, vf 0, vf 1))
        [ Vr 1; Vr 2 ] [ Vr 4 ] float_pairs)
    Rv32_kind.Fop.[ Add; Div; Max; Min; Mul; Sub ]
  @ List.concat_map
      (fun o ->
        [
          inst (Rv32_kind.Fcmp.vname o)
            (Vo.Vfcmp (o, vf 0, vf 1))
            [ Vr 1; Vr 2 ] [ Vr 4 ] float_pairs;
          inst
            (Rv32_kind.Fcmp.vname o ^ ", rd = a")
            (Vo.Vfcmp (o, vf 0, vf 1))
            [ Vr 1; Vr 2 ] [ Vr 1 ] float_pairs;
        ])
      Rv32_kind.Fcmp.[ Eq; Le; Lt ]
  @ List.map
      (fun o ->
        inst (Rv32_kind.Sgnj.vname o)
          (Vo.Vsgnj (o, vf 0, vf 1))
          [ Vr 1; Vr 2 ] [ Vr 4 ] float_pairs)
      Rv32_kind.Sgnj.[ J; Jn; Jx ]
  @ [
      inst "vfsqrt.v" (Vo.Vfsqrt (vf 0)) [ Vr 1 ] [ Vr 4 ] float_singles;
      inst "vfmacc.vv"
        (Vo.Vfmacc (vf 0, vf 1, vf 2))
        [ Vr 4; Vr 1; Vr 2 ] [ Vr 4 ] float_triples;
      inst "vfcvt.f.x.v"
        (Vo.Vfcvt_f_x (vi 0))
        [ Vr 1 ] [ Vr 4 ]
        (chunks 4 (C.ints @ [ 0x0100_0001L; 0xFEFF_FFFFL; 0x7FFF_FFC0L ]));
      inst "vfcvt.rtz.x.f.v"
        (Vo.Vfcvt_x_f (vf 0))
        [ Vr 1 ] [ Vr 4 ] float_singles;
      inst "vfmv.v.f"
        (Vo.Vfmv_v_f (Vo.f32x4, fv 0))
        [ Fr 11 ] [ Vr 4 ] (C.singles C.floats);
      inst "vmv.v.x"
        (Vo.Vmv_v_x (Vo.i32x4, gv 0))
        [ G 11 ] [ Vr 4 ] (C.singles C.ints);
      inst "vmv.v.v" (Vo.Vmv (vf 0)) [ Vr 1 ] [ Vr 4 ] float_singles;
      inst "vcast" (Vo.Vcast (Vo.i32x4, vf 0)) [ Vr 1 ] [ Vr 4 ] float_singles;
      inst "vnot" (Vo.Vnot (vm 0)) [ Vr 1 ] [ Vr 4 ] mask_lanes;
    ]
  @ List.map
      (fun o ->
        inst (Rv32_kind.Logic.name o)
          (Vo.Vlogic (o, vi 0, vi 1))
          [ Vr 1; Vr 2 ] [ Vr 4 ] int_pairs)
      Rv32_kind.Logic.[ And; Or; Xor ]

let lanes_moves =
  let with_scalar k =
    List.map (fun l -> l @ [ List.nth C.floats (k * 5) ]) float_singles
  in
  List.concat_map
    (fun k ->
      [
        inst
          (Fmt.str "vfmv.f.s [%d]" k)
          (Vo.Vfmv_f_s (k, vf 0))
          [ Vr 1 ] [ Fr 14 ] float_singles;
        inst (Fmt.str "vinsert [%d]" k)
          (Vo.Vinsert (k, vf 0, fv 1))
          [ Vr 1; Fr 11 ] [ Vr 4 ] (with_scalar k);
        inst
          (Fmt.str "vinsert [%d], rd = vector" k)
          (Vo.Vinsert (k, vf 0, fv 1))
          [ Vr 1; Fr 11 ] [ Vr 1 ] (with_scalar k);
      ])
    [ 0; 1; 2; 3 ]

let merges =
  let vs =
    List.concat_map
      (fun m ->
        List.map (fun p -> p @ m) (List.filteri (fun i _ -> i < 12) float_pairs))
      mask_lanes
  in
  List.map
    (fun (n, rd) ->
      inst ("vmerge, rd " ^ n)
        (Vo.Vmerge (vf 0, vf 1, vm 2))
        [ Vr 1; Vr 2; Vr 3 ] [ rd ] vs)
    [ ("apart", Vr 4); ("= else", Vr 1); ("= then", Vr 2); ("= mask", Vr 3) ]

let accesses =
  let ptr = v 0 Mir_type.Ptr in
  [
    inst ~memory:true "vle32.v"
      (Vo.Vle32 (Vo.f32x4, ptr))
      [ base ] [ Vr 4 ] [ [] ];
    inst ~memory:true "vse32.v"
      (Vo.Vse32 (ptr, vf 1))
      [ base; Vr 1 ] [] float_singles;
  ]
  @ List.concat_map
      (fun s ->
        [
          inst ~memory:true
            (Fmt.str "vlse32.v, stride %Ld" s)
            (Vo.Vlse32 (Vo.f32x4, ptr, gv 1))
            [ base; G 11 ] [ Vr 4 ] [ [ s ] ];
          inst ~memory:true
            (Fmt.str "vsse32.v, stride %Ld" s)
            (Vo.Vsse32 (ptr, gv 1, vf 2))
            [ base; G 11; Vr 1 ] []
            (List.map
               (fun l -> s :: l)
               (List.filteri (fun i _ -> i < 3) float_singles));
        ])
      [ 0L; 4L; 8L; 12L ]

let integer_lanes =
  let ptr = v 0 Mir_type.Ptr in
  let narrow =
    chunks 4 (C.ints @ [ 0x8081_7F80L; 0x7FFF_8000L; 0xFFFE_0001L ])
  in
  List.concat_map
    (fun (from, firsts) ->
      List.concat_map
        (fun signed ->
          List.map
            (fun first ->
              inst
                (Fmt.str "v%cext.%s[%d]"
                   (if signed then 's' else 'z')
                   (Mir_width.name from) first)
                (Vo.Vext
                   { signed; from; first; src = v 0 (Vo.int_register from) })
                [ Vn (1, from) ]
                [ Vr 4 ] narrow)
            firsts)
        [ true; false ])
    [ (Mir_width.W8, [ 0; 4; 8; 12 ]); (Mir_width.W16, [ 0; 4 ]) ]
  @ List.map
      (fun w ->
        inst ~memory:true
          (Fmt.str "vle%d.v" (Mir_width.bits w))
          (Vo.Vle_int (w, ptr))
          [ base ]
          [ Vn (4, w) ]
          [ [] ])
      [ Mir_width.W8; Mir_width.W16 ]
  @ List.concat_map
      (fun (o, ks) ->
        List.map
          (fun k ->
            inst
              (Fmt.str "%s %d" (Rv32_kind.Shift.name o) k)
              (Vo.Vshift (o, k, vi 0))
              [ Vr 1 ] [ Vr 4 ] (chunks 4 C.ints))
          ks)
      Rv32_kind.Shift.
        [ (Sll, [ 0; 1; 17; 31 ]); (Srl, [ 1; 4; 31 ]); (Sra, [ 1; 4; 31 ]) ]

let instances = fop @ lanes_moves @ merges @ accesses @ integer_lanes

(* {1 The model's expectation} *)

let area_bytes = 64

let area j =
  String.init area_bytes (fun i -> Char.chr ((0x10 + i + (j * 3)) land 0xFF))

let datum r words =
  match r with
  | Vr _ -> D.Lanes (Array.of_list words)
  | Vn (_, w) ->
      let b = Mir_width.bits w in
      let per = 32 / b in
      let ws = Array.of_list words in
      D.Lanes
        (Array.init (4 * per) (fun j ->
             Mir_width.normalize w
               (Int64.shift_right_logical ws.(j / per) (b * (j mod per)))))
  | G _ | Fr _ -> D.Bits (List.hd words)

(* the operand words of each use, in order, the base excluded *)
let split i words =
  let rec go uses words =
    match uses with
    | [] -> []
    | r :: rest when r = base && i.memory -> `Base :: go rest words
    | r :: rest ->
        let n = width r in
        let take = List.filteri (fun k _ -> k < n) words
        and drop = List.filteri (fun k _ -> k >= n) words in
        `Words (r, take) :: go rest drop
  in
  go i.uses words

let expected ~sem i j words =
  let memory = Mem.create () in
  let key =
    Option.get (Mem.alloc memory ~size:(Int64.of_int area_bytes) ~align:16L ())
  in
  Mem.write_string memory key ~offset:0L (area j);
  let p = Mem.pointer memory key ~lo:0L ~hi:(Int64.of_int area_bytes) in
  let ops = Array.of_list (split i words) in
  let get (x : Mir_value.t) =
    match ops.(Mir_id.Value.to_int x.Mir_value.id) with
    | `Base -> D.Ptr p
    | `Words (r, ws) -> datum r ws
  in
  let env =
    {
      E.get;
      memory;
      view = (fun _ -> None);
      defect = (fun _ -> raise C.Defect);
      call = (fun _ _ -> raise C.Defect);
    }
  in
  let words_of = function
    | D.Bits b -> [ Int64.logand b 0xFFFF_FFFFL ]
    | D.Lanes l -> Array.to_list l
    | _ -> raise C.Defect
  in
  (* a register of narrow lanes is its four words *)
  let pack w l =
    let b = Mir_width.bits w in
    let per = 32 / b in
    List.init 4 (fun k ->
        let acc = ref 0L in
        for j = per - 1 downto 0 do
          acc :=
            Int64.logor (Int64.shift_left !acc b)
              (Mir_width.normalize w l.((k * per) + j))
        done;
        !acc)
  in
  let results =
    List.concat
      (List.map2
         (fun r d ->
           match (r, d) with
           | Vn (_, w), D.Lanes l -> pack w l
           | _, d -> words_of d)
         i.defs (sem env (O.Vec i.op)))
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

let load r off =
  match r with
  | G _ -> [ F.i "lw" [ A.reg (rname r); A.mem "s0" off ] ]
  | Fr _ -> [ F.i "flw" [ A.reg (rname r); A.mem "s0" off ] ]
  | Vr _ | Vn _ ->
      [
        F.i "addi" [ A.reg "t0"; A.reg "s0"; A.imm off ];
        F.i "vle32.v" [ A.reg (rname r); A.mem "t0" 0 ];
      ]

let store r off =
  match r with
  | G _ -> [ F.i "sw" [ A.reg (rname r); A.mem "s1" off ] ]
  | Fr _ -> [ F.i "fsw" [ A.reg (rname r); A.mem "s1" off ] ]
  | Vr _ | Vn _ ->
      [
        F.i "addi" [ A.reg "t0"; A.reg "s1"; A.imm off ];
        F.i "vse32.v" [ A.reg (rname r); A.mem "t0" 0 ];
      ]

let offsets regs =
  let at = ref 0 in
  List.map
    (fun r ->
      let o = !at in
      at := !at + (4 * width r);
      (r, o))
    regs

let total regs = List.fold_left (fun n r -> n + width r) 0 regs

let process ~body ~inputs ~outputs n =
  let per_vector j =
    List.concat_map (fun (r, o) -> load r o) (offsets inputs)
    @ body j
    @ List.concat_map (fun (r, o) -> store r o) (offsets outputs)
    @ [
        F.i "addi" [ A.reg "s0"; A.reg "s0"; A.imm (4 * total inputs) ];
        F.i "addi" [ A.reg "s1"; A.reg "s1"; A.imm (4 * total outputs) ];
        F.i "addi" [ A.reg "s2"; A.reg "s2"; A.imm area_bytes ];
      ]
  in
  [
    F.i "addi" [ A.reg "sp"; A.reg "sp"; A.imm (-16) ];
    F.i "sw" [ A.reg "s0"; A.mem "sp" 0 ];
    F.i "sw" [ A.reg "s1"; A.mem "sp" 4 ];
    F.i "sw" [ A.reg "s2"; A.mem "sp" 8 ];
    F.i "lw" [ A.reg "s0"; A.mem "a0" 0 ];
    F.i "lw" [ A.reg "s1"; A.mem "a0" 4 ];
    F.i "lw" [ A.reg "s2"; A.mem "a0" 8 ];
    F.vsetivli ();
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
  |> Machine_rivet_riscv32_mir.Rv32_mir_module.lay_out
  |> Machine_rivet_riscv32_mir.Rv32_mir_module.render

let words xs =
  String.concat ""
    (List.map
       (fun x ->
         let b = Bytes.create 4 in
         Bytes.set_int32_le b 0 (Int64.to_int32 x);
         Bytes.to_string b)
       xs)

let run ~cpu ~items ~in_words ~out_words vectors =
  let n = List.length vectors in
  match
    A.module_ "vconformance" (fun () ->
        [ A.text; A.align 4; A.global "vconformance"; A.label "vconformance" ]
        @ items)
  with
  | Error what -> Error ("no form for " ^ what)
  | Ok m -> (
      let slot size placement =
        { H.size = max 4 size; align = 16; placement }
      in
      match
        H.prepare
          ~profile:Machine_rivet_riscv32.Rv32_profile.Rv32imf_zve32f_ilp32f
          ~entry:"vconformance" ~kernel:[ m ]
          [
            slot (4 * n * max 1 in_words) H.Placement.In;
            slot (4 * n * max 1 out_words) H.Placement.Out;
            slot (area_bytes * n) H.Placement.Out;
          ]
      with
      | Error e -> Error e
      | Ok p -> (
          let bound = function
            | 0 -> Some (words (List.concat vectors))
            | 2 -> Some (String.concat "" (List.init n area))
            | _ -> None
          in
          match H.run ~cpu p ~bound with
          | Error e -> Error e
          | Ok o -> Ok (List.assoc 1 o.H.outputs, List.assoc 2 o.H.outputs)))

let check_instance ?mutation ~sem ~cpu i =
  let e = C.env mutation in
  let uses = List.map (fun r -> Loc.Reg (view r)) i.uses
  and defs = List.map (fun r -> Loc.Reg (view r)) i.defs in
  let inputs = List.filter (fun r -> not (r = base && i.memory)) i.uses in
  let body _ = F.items e (O.Vec i.op) ~uses ~defs in
  let n = List.length i.vectors in
  match
    run ~cpu
      ~items:(process ~body ~inputs ~outputs:i.defs n)
      ~in_words:(total inputs) ~out_words:(total i.defs) i.vectors
  with
  | Error e -> Some (Fmt.str "%s: %s" i.name e)
  | Ok (out, areas) ->
      let bad = ref 0 and first = ref None in
      List.iteri
        (fun j ws ->
          let results, bytes = expected ~sem i j ws in
          let got =
            List.mapi (fun k _ -> C.word out ((j * total i.defs) + k)) results
          in
          let area_got = String.sub areas (j * area_bytes) area_bytes in
          if
            got <> results
            || match bytes with Some b -> b <> area_got | None -> false
          then (
            incr bad;
            if !first = None then
              first :=
                Some
                  (Fmt.str "%a -> %a, expected %a"
                     Fmt.(list ~sep:(any " ") (fmt "%Lx"))
                     ws
                     Fmt.(list ~sep:(any " ") (fmt "%Lx"))
                     got
                     Fmt.(list ~sep:(any " ") (fmt "%Lx"))
                     results)))
        i.vectors;
      if !bad = 0 then None
      else
        Some
          (Fmt.str "%s: %d of %d differ, first %s" i.name !bad n
             (Option.value ~default:"" !first))

let cpu vlen =
  Fmt.str
    "rv32,d=false,zve32f=true,vlen=%d,elen=32,rvv_ta_all_1s=true,rvv_ma_all_1s=true"
    vlen

let summary ?mutation ~sem vlen =
  List.filter_map (check_instance ?mutation ~sem ~cpu:(cpu vlen)) instances

let%expect_test "every vector form agrees with its semantics at every VLEN" =
  Fmt.pr "%d vector form instances, %d vectors@." (List.length instances)
    (List.fold_left (fun n i -> n + List.length i.vectors) 0 instances);
  List.iter
    (fun vlen ->
      let failures = summary ~sem:Rv32_sem.exec vlen in
      Fmt.pr "vlen %d: %d disagree@." vlen (List.length failures);
      List.iter (Fmt.pr "  %s@.") failures)
    [ 128; 512; 1024 ];
  [%expect
    {|
    77 vector form instances, 3381 vectors
    vlen 128: 0 disagree
    vlen 512: 0 disagree
    vlen 1024: 0 disagree |}]

let with_form f env op =
  match op with
  | O.Vec x -> (
      match f env x with Some r -> r | None -> Rv32_sem.exec env op)
  | _ -> Rv32_sem.exec env op

let%expect_test "vector semantic and mapping mutations are caught" =
  let lanes env x =
    match env.E.get x with D.Lanes l -> l | _ -> raise C.Defect
  in
  List.iter
    (fun (name, f) ->
      Fmt.pr "%s: %d disagree@." name
        (List.length (summary ~sem:(with_form f) 128)))
    [
      ( "vfmax.vv as IEEE maximum",
        fun env -> function
          | Vo.Vfbin (Rv32_kind.Fop.Max, a, b) ->
              let x = lanes env a and y = lanes env b in
              Some
                [
                  D.Lanes
                    (Array.init 4 (fun k ->
                         Rv32_sem.fres
                           (Machine_interp.Mir_numeric.fmax
                              (Machine_interp.Mir_numeric.f32 x.(k))
                              (Machine_interp.Mir_numeric.f32 y.(k)))));
                ]
          | _ -> None );
      ( "a compare's lanes as 1, not all ones",
        fun env op ->
          match op with
          | Vo.Vfcmp _ ->
              Some
                (List.map
                   (function
                     | D.Lanes l ->
                         D.Lanes (Array.map (fun x -> Int64.logand x 1L) l)
                     | d -> d)
                   (Rv32_sem.exec env (O.Vec op)))
          | _ -> None );
      ( "vsext as a zero extension",
        fun env -> function
          | Vo.Vext ({ signed = true; _ } as e) ->
              Some
                (Rv32_sem.exec env (O.Vec (Vo.Vext { e with signed = false })))
          | _ -> None );
      ( "a strided load read contiguously",
        fun env -> function
          | Vo.Vlse32 (t, p, _) ->
              Some (Rv32_sem.exec env (O.Vec (Vo.Vle32 (t, p))))
          | _ -> None );
    ];
  List.iter
    (fun (name, mutation) ->
      Fmt.pr "map %s: %d disagree@." name
        (List.length (summary ~mutation ~sem:Rv32_sem.exec 128)))
    F.Mutation.[ ("commuted sub", Commuted_sub); ("select arms", Select_arms) ];
  [%expect
    {|
    vfmax.vv as IEEE maximum: 1 disagree
    a compare's lanes as 1, not all ones: 6 disagree
    vsext as a zero extension: 6 disagree
    a strided load read contiguously: 3 disagree
    map commuted sub: 1 disagree
    map select arms: 4 disagree |}]
