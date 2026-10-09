(* Per-form native conformance through typed Rivet instructions (the Rivet
   route of test/machine_a64_native's GNU-assembler harness). Every admitted
   scalar form is made from its selected instruction and the locations the
   harness chooses by [Rivet_a64_form], wrapped in a typed probe that seeds the
   operands, NZCV and destination, runs the instruction, and reads the
   destination, NZCV and a canaried buffer back; each vector is one call of the
   loaded probe. The prediction is [A64_model]'s, from the interpreter's
   semantics, over the same boundary and random vectors.

   [--mutate NAME] is the model-side mutation of the GNU harness and must be
   caught; [--map-mutate NAME] is a deliberate defect in the adapter's mapping
   and must be caught too. *)

open Machine_ir
open Machine_interp
open Machine_target_aarch64
open A64_op
open A64_forms
open A64_model
module Form_map = Machine_rivet_aarch64.Rivet_a64_form
module Image = Machine_rivet_aarch64.Rivet_a64_image
module N = Asm_core.Normalized_ast
module D = Asm_core.Directive
module O = Aarch64.Operand
module Op = Aarch64.Opcode
module Loc = Mir_phys.Loc

let origin = Foundation.Origin.synthesized ~pass:"rivet_conformance" ()
let ins op ops = { Aarch64.Instruction.op; ops }
let item i = N.Instruction { insn = i; origin }
let reg n = O.Reg { Aarch64.Reg.num = n; width = 64; is_sp = false }
let freg n = O.Freg { Aarch64.Freg.num = n; double = true }
let qreg n = O.Qreg n

let mem base off =
  O.Mem
    {
      Aarch64.Mem.base = { Aarch64.Reg.num = base; width = 64; is_sp = false };
      offset = Aarch64.Disp.Const (Int64.of_int off);
      writeback = false;
      pre = true;
    }

let imm n = O.Imm (Foundation.Bigint.of_int n)
let io_bytes = 128

(* io words: 0-2 inputs, 3 NZCV, 4 destination seed, 5-8 the buffer, 10-11 a
   vector destination seed; out 12-13 the destination, 14 NZCV. *)
let view_of_input k = function
  | G Sz.X -> A64_reg.x (k + 1)
  | G Sz.W -> A64_reg.w (k + 1)
  | F Fsz.D -> A64_reg.d (k + 1)
  | F Fsz.S -> A64_reg.s (k + 1)

let result_view = function
  | G Sz.X -> A64_reg.x 0
  | G Sz.W -> A64_reg.w 0
  | F Fsz.D -> A64_reg.d 0
  | F Fsz.S -> A64_reg.s 0

let probe_module ?mutation (f : Form.t) =
  Err.Escape.with_escape @@ fun esc ->
  let env =
    {
      Form_map.mutation;
      esc;
      table_slot = (fun _ -> None);
      reference = (fun _ -> None);
    }
  in
  let vs =
    List.mapi
      (fun i r -> Mir_value.{ id = Mir_id.Value.of_int i; ty = ty_of r })
      f.Form.inputs
  in
  let flags_v = Mir_value.{ id = Mir_id.Value.of_int 9; ty = Mir_type.Flags } in
  let loc_of (v : Mir_value.t) =
    match Mir_id.Value.to_int v.Mir_value.id with
    | 9 -> Loc.Reg A64_reg.nzcv
    | 8 -> Loc.Reg (A64_reg.x 4)
    | i -> Loc.Reg (view_of_input i (List.nth f.Form.inputs i))
  in
  let prologue =
    ins Op.Mov [ reg 8; reg 0 ]
    :: List.concat
         (List.mapi
            (fun i r ->
              match r with
              | G _ -> [ ins Op.Ldr [ reg (i + 1); mem 8 (8 * i) ] ]
              | F _ -> [ ins Op.Ldr [ freg (i + 1); mem 8 (8 * i) ] ])
            f.Form.inputs)
    @ [
        ins Op.Ldr [ reg 9; mem 8 24 ];
        ins Op.Msr [ O.Sym (Asm_core.Expr.Symbol "nzcv"); reg 9 ];
      ]
    @ (match f.Form.result with
      | Some (G _) -> [ ins Op.Ldr [ reg 0; mem 8 32 ] ]
      | Some (F _) -> [ ins Op.Ldr [ qreg 0; mem 8 80 ] ]
      | None -> [])
    @ [ ins Op.Add [ reg 4; reg 8; imm 48 ] ]
  in
  let epilogue =
    [
      ins Op.Mrs [ reg 9; O.Sym (Asm_core.Expr.Symbol "nzcv") ];
      ins Op.Str [ reg 9; mem 8 112 ];
    ]
    @ (match f.Form.result with
      | Some (G _) ->
          [
            ins Op.Str [ reg 0; mem 8 96 ];
            ins Op.Movz [ reg 10; imm 0 ];
            ins Op.Str [ reg 10; mem 8 104 ];
          ]
      | Some (F _) -> [ ins Op.Str [ qreg 0; mem 8 96 ] ]
      | None ->
          [
            ins Op.Movz [ reg 10; imm 0 ];
            ins Op.Str [ reg 10; mem 8 96 ];
            ins Op.Str [ reg 10; mem 8 104 ];
          ])
    @ [ ins Op.Ret [] ]
  in
  let body =
    match f.Form.make vs flags_v with
    | `Op op ->
        let uses = List.map loc_of (A64_op.uses op) in
        (* a tied form writes the register it reads: its result is then moved
           to the harness's destination *)
        let tied =
          List.find_map
            (function
              | Mir_target.Constraint.Tied { result = 0; use } -> Some use
              | _ -> None)
            (A64_op.constraints op)
        in
        let defs, move =
          match (f.Form.result, tied) with
          | Some r, Some use -> (
              let l = List.nth uses use in
              ( [ l ],
                match (r, l) with
                | G _, Loc.Reg v ->
                    [
                      ins Op.Mov
                        [
                          reg 0; reg (Mir_id.Unit.to_int v.Mir_target.View.unit);
                        ];
                    ]
                | _ -> [] ))
          | Some r, None -> ([ Loc.Reg (result_view r) ], [])
          | None, _ ->
              ( (if A64_op.writes_flags op then [ Loc.Reg A64_reg.nzcv ] else []),
                [] )
        in
        List.map item
          (prologue @ Form_map.instructions env op ~uses ~defs @ move @ epilogue)
    | `Test t ->
        let uses = List.map loc_of (A64_op.test_uses t) in
        let branch =
          Form_map.branch env t ~uses ~label:".Ltaken" ~inverted:false
        in
        List.map item (prologue @ [ ins Op.Movz [ reg 0; imm 0 ]; branch ])
        @ [
            item (Form_map.jump ".Lend");
            N.Label { name = ".Ltaken"; origin };
            item (ins Op.Movz [ reg 0; imm 1 ]);
            N.Label { name = ".Lend"; origin };
          ]
        @ List.map item epilogue
  in
  {
    N.unit_name = "probe";
    items =
      [
        N.Directive
          {
            directive =
              D.Section
                { name = ".text"; perms = Asm_core.Perms.rx; nobits = false };
            origin;
          };
        N.Directive { directive = D.Align { boundary = 4 }; origin };
        N.Directive { directive = D.Global { name = "probe" }; origin };
        N.Label { name = "probe"; origin };
      ]
      @ body;
  }

let set io w v =
  for b = 0 to 7 do
    io.{(8 * w) + b} <-
      Char.chr
        (Int64.to_int
           (Int64.logand (Int64.shift_right_logical v (8 * b)) 0xFFL))
  done

let word io w =
  let v = ref 0L in
  for b = 7 downto 0 do
    v :=
      Int64.logor (Int64.shift_left !v 8)
        (Int64.of_int (Char.code io.{(8 * w) + b}))
  done;
  !v

type native = { lo : int64; hi : int64; flags : int64; mem : Bytes.t }

let run_vector loaded io (ins_ : int64 list) nz seed (buf : Bytes.t) =
  Bigarray.Array1.fill io '\000';
  List.iteri (fun i x -> set io i x) ins_;
  set io 3 nz;
  set io 4 seed;
  for i = 0 to 3 do
    set io (5 + i) (Bytes.get_int64_le buf (8 * i))
  done;
  set io 10 seed;
  set io 11 seed;
  match Err.payload (Image.call ~io loaded) with
  | Error e -> Error (Fmt.str "%a" Image.Error.pp e)
  | Ok _ ->
      let mem = Bytes.create 32 in
      for i = 0 to 3 do
        Bytes.set_int64_le mem (8 * i) (word io (5 + i))
      done;
      let lo = word io 12 and hi = word io 13 in
      Ok { lo; hi; flags = word io 14; mem }

(* ---- Advanced SIMD forms -------------------------------------------------- *)

module VF = Vector_forms

(* 128-bit contents: two words, lane [k] at [k * width] bits. *)
let lane_width = function
  | VF.V a -> ( match a with Arr.S2 | Arr.S4 -> 32 | Arr.D2 -> 64)
  | VF.M a -> ( match a with Arr.D2 -> 64 | _ -> 32)
  | VF.Sc _ | VF.Gp _ -> 64

let lane_count = function
  | VF.V a | VF.M a -> Arr.lanes a
  | VF.Sc _ | VF.Gp _ -> 1

let unpack_lanes kind lo hi =
  let w = lane_width kind and n = lane_count kind in
  let mask = if w >= 64 then -1L else Int64.pred (Int64.shift_left 1L w) in
  Array.init n (fun k ->
      let pos = k * w in
      let word, shift = if pos < 64 then (lo, pos) else (hi, pos - 64) in
      Int64.logand (Int64.shift_right_logical word shift) mask)

let pack_lanes kind (l : int64 array) =
  let w = lane_width kind in
  let lo = ref 0L and hi = ref 0L in
  Array.iteri
    (fun k x ->
      let x =
        if w >= 64 then x
        else Int64.logand x (Int64.pred (Int64.shift_left 1L w))
      in
      let pos = k * w in
      if pos < 64 then lo := Int64.logor !lo (Int64.shift_left x pos)
      else hi := Int64.logor !hi (Int64.shift_left x (pos - 64)))
    l;
  (!lo, !hi)

let vloc_of_kind kind n =
  match kind with
  | VF.V Arr.S2 -> Loc.Reg (A64_reg.d n)
  | VF.V _ | VF.M _ -> Loc.Reg (A64_reg.q n)
  | VF.Sc Fsz.D -> Loc.Reg (A64_reg.d n)
  | VF.Sc Fsz.S -> Loc.Reg (A64_reg.s n)
  | VF.Gp Sz.X -> Loc.Reg (A64_reg.x n)
  | VF.Gp Sz.W -> Loc.Reg (A64_reg.w n)

let vec_probe ?mutation (f : VF.t) =
  Err.Escape.with_escape @@ fun esc ->
  let env =
    {
      Form_map.mutation;
      esc;
      table_slot = (fun _ -> None);
      reference = (fun _ -> None);
    }
  in
  let vs =
    List.mapi
      (fun i k -> Mir_value.{ id = Mir_id.Value.of_int i; ty = VF.ty_of k })
      f.VF.inputs
  in
  let base_v = Mir_value.{ id = Mir_id.Value.of_int 8; ty = Mir_type.Ptr } in
  let op = f.VF.make vs base_v in
  let loc_of (v : Mir_value.t) =
    match Mir_id.Value.to_int v.Mir_value.id with
    | 8 -> Loc.Reg (A64_reg.x 4)
    | i -> vloc_of_kind (List.nth f.VF.inputs i) (i + 1)
  in
  let uses = List.map loc_of (A64_op.uses op) in
  let qn n = qreg n in
  let prologue =
    ins Op.Mov [ reg 8; reg 0 ]
    :: List.mapi
         (fun i k ->
           match k with
           | VF.Gp _ -> ins Op.Ldr [ reg (i + 1); mem 8 (16 * i) ]
           | _ -> ins Op.Ldr [ qn (i + 1); mem 8 (16 * i) ])
         f.VF.inputs
    @ [ ins Op.Ldr [ qn 0; mem 8 64 ]; ins Op.Add [ reg 4; reg 8; imm 88 ] ]
  in
  let defs, move =
    match (f.VF.result, f.VF.tied) with
    | None, _ -> ([], [])
    | Some _, Some use ->
        let l = List.nth uses use in
        let n =
          match l with
          | Loc.Reg v -> Mir_id.Unit.to_int v.Mir_target.View.unit - 32
          | _ -> 0
        in
        ( [ l ],
          [
            ins Op.Mov
              [ O.Vec (0, Aarch64.Varr.B16); O.Vec (n, Aarch64.Varr.B16) ];
          ] )
    | Some k, None -> ([ vloc_of_kind k 0 ], [])
  in
  let epilogue =
    (match f.VF.result with
      | Some _ -> [ ins Op.Str [ qn 0; mem 8 144 ] ]
      | None -> [])
    @ [ ins Op.Ret [] ]
  in
  {
    N.unit_name = "vprobe";
    items =
      [
        N.Directive
          {
            directive =
              D.Section
                { name = ".text"; perms = Asm_core.Perms.rx; nobits = false };
            origin;
          };
        N.Directive { directive = D.Align { boundary = 4 }; origin };
        N.Directive { directive = D.Global { name = "probe" }; origin };
        N.Label { name = "probe"; origin };
      ]
      @ List.map item
          (prologue @ Form_map.instructions env op ~uses ~defs @ move @ epilogue);
  }

let nan_lane kind x =
  match kind with
  | VF.V a -> (
      match a with
      | Arr.D2 -> Float.is_nan (Int64.float_of_bits x)
      | _ -> Float.is_nan (Int32.float_of_bits (Int64.to_int32 x)))
  | VF.Sc Fsz.D -> Float.is_nan (Int64.float_of_bits x)
  | VF.Sc Fsz.S -> Float.is_nan (Int32.float_of_bits (Int64.to_int32 x))
  | VF.M _ | VF.Gp _ -> false

let run_vector_forms map_mutation =
  let st = Random.State.make [| A64_model.seed + 1 |] in
  let io = Bigarray.Array1.create Bigarray.char Bigarray.c_layout 192 in
  let failures = Hashtbl.create 16 and total = ref 0 in
  let random_lane kind =
    let f32 () =
      if Random.State.int st 10 < 6 then
        List.nth f32_boundary (Random.State.int st (List.length f32_boundary))
      else Random.State.int64 st 0x1_0000_0000L
    and f64 () =
      if Random.State.int st 10 < 6 then
        List.nth f64_boundary (Random.State.int st (List.length f64_boundary))
      else
        Int64.logxor
          (Random.State.int64 st Int64.max_int)
          (Int64.shift_left (Random.State.int64 st 2L) 63)
    in
    match kind with
    | VF.V Arr.D2 | VF.Sc Fsz.D -> f64 ()
    | VF.V _ | VF.Sc Fsz.S -> f32 ()
    | VF.M Arr.D2 -> if Random.State.bool st then -1L else 0L
    | VF.M _ -> if Random.State.bool st then 0xFFFF_FFFFL else 0L
    | VF.Gp Sz.X ->
        List.nth gpr_boundary (Random.State.int st (List.length gpr_boundary))
    | VF.Gp Sz.W ->
        Int64.logand (Random.State.int64 st Int64.max_int) 0xFFFF_FFFFL
  in
  let input_words kind =
    match kind with
    | VF.Gp _ -> (random_lane kind, 0L)
    | VF.Sc _ -> (random_lane kind, 0L)
    | _ ->
        pack_lanes kind
          (Array.init (lane_count kind) (fun _ -> random_lane kind))
  in
  List.iter
    (fun (f : VF.t) ->
      let fail why =
        if not (Hashtbl.mem failures f.VF.name) then
          Hashtbl.replace failures f.VF.name why
      in
      match Err.payload (vec_probe ?mutation:map_mutation f) with
      | Error r ->
          fail
            (Fmt.str "mapping: %a" Machine_rivet_aarch64.Rivet_a64_refusal.pp r)
      | Ok m -> (
          match Err.payload (Image.plan ~entry:"probe" [ m ]) with
          | Error e -> fail (Fmt.str "plan: %a" Image.Error.pp e)
          | Ok laid -> (
              match Err.payload (Image.load laid) with
              | Error e -> fail (Fmt.str "load: %a" Image.Error.pp e)
              | Ok loaded ->
                  for _ = 1 to 300 do
                    incr total;
                    let ins_ = List.map input_words f.VF.inputs in
                    let seed_lo = Random.State.int64 st Int64.max_int
                    and seed_hi = Random.State.int64 st Int64.max_int in
                    let buf =
                      Bytes.init 64 (fun _ ->
                          Char.chr (Random.State.int st 256))
                    in
                    Bigarray.Array1.fill io '\000';
                    List.iteri
                      (fun i (lo, hi) ->
                        set io (2 * i) lo;
                        set io ((2 * i) + 1) hi)
                      ins_;
                    set io 8 seed_lo;
                    set io 9 seed_hi;
                    for i = 0 to 7 do
                      set io (10 + i) (Bytes.get_int64_le buf (8 * i))
                    done;
                    match Err.payload (Image.call ~io loaded) with
                    | Error e -> fail (Fmt.str "%a" Image.Error.pp e)
                    | Ok _ -> (
                        let nlo = word io 18 and nhi = word io 19 in
                        let nmem = Bytes.create 64 in
                        for i = 0 to 7 do
                          Bytes.set_int64_le nmem (8 * i) (word io (10 + i))
                        done;
                        (* the model *)
                        let memory = Mir_memory.create () in
                        let key =
                          Option.get
                            (Mir_memory.alloc memory ~size:64L ~align:16L ())
                        in
                        Mir_memory.write_string memory key ~offset:0L
                          (Bytes.to_string buf);
                        let basep =
                          Option.get
                            (Mir_memory.offset_by
                               (Mir_memory.pointer memory key ~lo:0L ~hi:64L)
                               8L)
                        in
                        let datum kind (lo, hi) =
                          match kind with
                          | VF.Gp _ -> Mir_datum.Bits lo
                          | VF.Sc _ -> Mir_datum.Bits lo
                          | _ -> Mir_datum.Lanes (unpack_lanes kind lo hi)
                        in
                        let get (v : Mir_value.t) =
                          match Mir_id.Value.to_int v.Mir_value.id with
                          | 8 -> Mir_datum.Ptr basep
                          | i ->
                              datum (List.nth f.VF.inputs i) (List.nth ins_ i)
                        in
                        let menv =
                          {
                            Mir_sel_env.get;
                            memory;
                            view = (fun _ -> None);
                            defect = (fun d -> raise (Model_defect d));
                            call =
                              (fun _ _ ->
                                raise
                                  (Model_defect
                                     Mir_observation.Defect.Invalid_program));
                          }
                        in
                        let vs =
                          List.mapi
                            (fun i k ->
                              Mir_value.
                                { id = Mir_id.Value.of_int i; ty = VF.ty_of k })
                            f.VF.inputs
                        in
                        let op =
                          f.VF.make vs
                            Mir_value.
                              { id = Mir_id.Value.of_int 8; ty = Mir_type.Ptr }
                        in
                        match A64_sem.exec menv op with
                        | exception Model_defect _ -> fail "model defect"
                        | rs ->
                            let plo, phi =
                              match (f.VF.result, rs) with
                              | Some k, [ Mir_datum.Lanes l ] -> pack_lanes k l
                              | Some _, [ Mir_datum.Bits b ] -> (b, 0L)
                              | _ -> (0L, 0L)
                            in
                            let pmem =
                              Array.init 64 (fun i ->
                                  Option.value ~default:0
                                    (Mir_memory.read_bytes memory key ~offset:0L
                                       ~n:64).(i))
                            in
                            let result_ok =
                              match f.VF.result with
                              | None -> true
                              | Some k -> (
                                  let n = unpack_lanes k nlo nhi
                                  and p = unpack_lanes k plo phi in
                                  Int64.equal
                                    (if lane_count k = 1 then nhi else nhi)
                                    (if lane_count k = 1 then 0L else nhi)
                                  && (let rec go i =
                                        i >= Array.length n
                                        || (Int64.equal n.(i) p.(i)
                                           || nan_lane k n.(i)
                                              && nan_lane k p.(i))
                                           && go (i + 1)
                                      in
                                      go 0)
                                  &&
                                  (* bits above the lanes stay zero for a 64-bit write *)
                                  match k with
                                  | VF.V Arr.S2 | VF.Sc _ -> Int64.equal nhi 0L
                                  | _ -> true)
                            in
                            let mem_ok =
                              Array.for_all Fun.id
                                (Array.init 64 (fun i ->
                                     Char.code (Bytes.get nmem i) = pmem.(i)))
                            in
                            if not (result_ok && mem_ok) then
                              fail
                                (Printf.sprintf
                                   "inputs %s: native %Lx:%Lx, model %Lx:%Lx%s"
                                   (String.concat " "
                                      (List.map
                                         (fun (a, b) ->
                                           Printf.sprintf "%Lx:%Lx" b a)
                                         ins_))
                                   nhi nlo phi plo
                                   (if mem_ok then "" else " (memory differs)"))
                        )
                  done;
                  Image.close loaded)))
    VF.forms;
  Printf.printf
    "NEON per-form conformance through Rivet: %d forms, %d vectors\n"
    (List.length VF.forms) !total;
  List.iter
    (fun (f : VF.t) ->
      match Hashtbl.find_opt failures f.VF.name with
      | Some why -> Printf.printf "FAIL %s: %s\n" f.VF.name why
      | None -> ())
    VF.forms;
  Printf.printf "%d of %d vector forms agree\n"
    (List.length VF.forms - Hashtbl.length failures)
    (List.length VF.forms);
  Hashtbl.length failures

let () =
  let mutation, map_mutation =
    match Array.to_list Sys.argv with
    | [ _ ] -> (None, None)
    | [ _; "--mutate"; m ] -> (
        match List.assoc_opt m Mutation.all with
        | Some m -> (Some m, None)
        | None ->
            prerr_endline ("unknown mutation " ^ m);
            exit 64)
    | [ _; "--map-mutate"; m ] -> (
        match m with
        | "commuted-sub" -> (None, Some Form_map.Mutation.Commuted_sub)
        | "wrong-lane" -> (None, Some Form_map.Mutation.Wrong_lane)
        | _ ->
            prerr_endline ("unknown mapping mutation " ^ m);
            exit 64)
    | _ ->
        prerr_endline
          "usage: rivet_conformance [--mutate NAME | --map-mutate NAME]";
        exit 64
  in
  let arch =
    String.trim (In_channel.input_all (Unix.open_process_in "uname -m"))
  in
  if arch <> "aarch64" then (
    Printf.printf "unavailable: host %s is not aarch64\n" arch;
    exit 2);
  let all = A64_model.vectors () in
  let by_form = Hashtbl.create 256 in
  List.iter
    (fun ((k, _, _, _, _, _) as v) ->
      Hashtbl.replace by_form k
        (v :: Option.value ~default:[] (Hashtbl.find_opt by_form k)))
    all;
  let io = Bigarray.Array1.create Bigarray.char Bigarray.c_layout io_bytes in
  let failures = Hashtbl.create 16 and total = ref 0 in
  List.iteri
    (fun k (f : Form.t) ->
      let vectors =
        List.rev (Option.value ~default:[] (Hashtbl.find_opt by_form k))
      in
      let fail why =
        if not (Hashtbl.mem failures f.Form.name) then
          Hashtbl.replace failures f.Form.name why
      in
      match Err.payload (probe_module ?mutation:map_mutation f) with
      | Error r ->
          fail
            (Fmt.str "mapping: %a" Machine_rivet_aarch64.Rivet_a64_refusal.pp r)
      | Ok m -> (
          match Err.payload (Image.plan ~entry:"probe" [ m ]) with
          | Error e -> fail (Fmt.str "plan: %a" Image.Error.pp e)
          | Ok laid -> (
              match Err.payload (Image.load laid) with
              | Error e -> fail (Fmt.str "load: %a" Image.Error.pp e)
              | Ok loaded ->
                  List.iter
                    (fun (_, _, ins_, nz, seed, buf) ->
                      incr total;
                      match run_vector loaded io ins_ nz seed buf with
                      | Error e -> fail e
                      | Ok n -> (
                          match predict mutation f ins_ nz seed buf with
                          | exception Model_defect _ -> fail "model defect"
                          | plo, phi, pflags, pmem ->
                              let result_ok =
                                match f.Form.result with
                                | Some (F _ as r)
                                  when float_bits_nan r n.lo
                                       && float_bits_nan r plo ->
                                    Int64.equal n.hi phi
                                | _ ->
                                    Int64.equal n.lo plo && Int64.equal n.hi phi
                              in
                              let mem_ok =
                                Array.for_all Fun.id
                                  (Array.init 32 (fun i ->
                                       Char.code (Bytes.get n.mem i) = pmem.(i)))
                              in
                              let flags_ok =
                                Int64.equal
                                  (Int64.logand n.flags 0xF000_0000L)
                                  (Int64.logand pflags 0xF000_0000L)
                              in
                              if not (result_ok && mem_ok && flags_ok) then
                                fail
                                  (Printf.sprintf
                                     "inputs %s nzcv %Lx seed %Lx: native \
                                      %Lx:%Lx flags %Lx, model %Lx:%Lx flags \
                                      %Lx%s"
                                     (String.concat ","
                                        (List.map (Printf.sprintf "%Lx") ins_))
                                     (Int64.shift_right_logical nz 28)
                                     seed n.hi n.lo
                                     (Int64.shift_right_logical n.flags 28)
                                     phi plo
                                     (Int64.shift_right_logical pflags 28)
                                     (if mem_ok then "" else " (memory differs)"))
                          ))
                    vectors;
                  Image.close loaded)))
    forms;
  Printf.printf
    "aarch64 per-form conformance through Rivet: %d forms, %d vectors, seed %d\n"
    (List.length forms) !total A64_model.seed;
  let failed = Hashtbl.length failures in
  List.iter
    (fun (f : Form.t) ->
      match Hashtbl.find_opt failures f.Form.name with
      | Some why -> Printf.printf "FAIL %s: %s\n" f.Form.name why
      | None -> ())
    forms;
  Printf.printf "%d of %d forms agree\n"
    (List.length forms - failed)
    (List.length forms);
  let failed = failed + run_vector_forms map_mutation in
  match (mutation, map_mutation) with
  | None, None -> exit (if failed = 0 then 0 else 1)
  | _ ->
      if failed > 0 then (
        print_endline "mutation detected";
        exit 0)
      else (
        print_endline "MUTATION SURVIVED";
        exit 1)
