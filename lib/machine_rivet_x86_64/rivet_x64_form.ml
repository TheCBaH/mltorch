(* One physical x86-64 instruction as the Rivet instructions that encode it.
   The selected form says the operation and its immediates; the locations the
   allocator chose say the registers, by operand position. Instructions are made
   through the encoder's surface constructor (an AT&T mnemonic and typed
   operands, source first), so no text is produced or parsed. *)

open Machine_ir
open Machine_target_x86_64
module Loc = Mir_phys.Loc
module R = Rivet_x64_refusal
module X = X64_op
module Fam = X86_family_encode
module Op = Fam.Operand

(* Fault injection for the evidence suite: each is one deliberate mapping
   defect a native comparison must detect. No consumer passes one. *)
module Mutation = struct
  type t =
    | Dropped_disp  (** an address's displacement left out *)
    | Inverted_cond  (** a conditional move or set on the opposite condition *)
end

type env = {
  mutation : Mutation.t option;
  esc : R.t Err.Escape.t;
  reference : Mir_target.Reference.t -> (string * int64) option;
  table_slot : string -> int option;
}

let refuse env r = Err.Escape.throw env.esc r
let origin = Foundation.Origin.synthesized ~pass:"machine_rivet_x86_64" ()

let make env mnemonic ops =
  match X86_64.make_surface_instruction ~mnemonic ~origin ops with
  | Error _ -> refuse env (R.Form mnemonic)
  | Ok s -> (
      match X86_64.simplify_instruction X86_64.default_state s with
      | Ok i -> i
      | Error e ->
          refuse env
            (R.Rivet
               (Fmt.str "%s: %a" mnemonic (Err.Error.pp_kind X86_64.pp_error) e))
      )

(* {1 Registers} *)

let names8 = [| "al"; "cl"; "dl"; "bl"; "spl"; "bpl"; "sil"; "dil" |]
let names16 = [| "ax"; "cx"; "dx"; "bx"; "sp"; "bp"; "si"; "di" |]

let gpr_name ~bits k =
  match bits with
  | 64 -> X64_reg.names64.(k)
  | 32 -> X64_reg.names32.(k)
  | 16 -> if k < 8 then names16.(k) else Printf.sprintf "r%dw" k
  | _ -> if k < 8 then names8.(k) else Printf.sprintf "r%db" k

let find env name =
  match X86_64.find_reg name with
  | Some r -> r
  | None -> refuse env (R.Register name)

let view env = function
  | Loc.Reg v -> v
  | Loc.Slot _ -> refuse env (R.Location "a frame slot")
  | Loc.Mem _ -> refuse env (R.Location "memory")

let unit_of env l = Mir_id.Unit.to_int (view env l).Mir_target.View.unit

(* A general register of [bits] bits. *)
let g env ~bits l =
  let k = unit_of env l in
  if k < 0 || k > 15 then
    refuse env (R.Register (view env l).Mir_target.View.name)
  else Op.Reg (find env (gpr_name ~bits k))

let xmm env l =
  let k = unit_of env l - 16 in
  if k < 0 || k > 15 then
    refuse env (R.Register (view env l).Mir_target.View.name)
  else Op.Reg (find env (Printf.sprintf "xmm%d" k))

let imm n = Op.Imm (Foundation.Bigint.of_int64 n)

let mem_op ?index ~base ~disp () =
  Op.Mem
    {
      Fam.Mem.base = Some base;
      index = Option.map fst index;
      scale = (match index with Some (_, s) -> s | None -> 1);
      disp = Fam.Disp.Const disp;
    }

let sz_bits = X.Sz.bits
let sfx = X.Sz.name

let nth env what l k =
  match List.nth_opt l k with
  | Some x -> x
  | None -> refuse env (R.Location (Fmt.str "a missing %s %d" what k))

let invert (c : X.Cond.t) : X.Cond.t =
  match c with
  | X.Cond.A -> X.Cond.Be
  | X.Cond.Be -> X.Cond.A
  | X.Cond.Ae -> X.Cond.B
  | X.Cond.B -> X.Cond.Ae
  | X.Cond.E -> X.Cond.Ne
  | X.Cond.Ne -> X.Cond.E
  | X.Cond.G -> X.Cond.Le
  | X.Cond.Le -> X.Cond.G
  | X.Cond.Ge -> X.Cond.L
  | X.Cond.L -> X.Cond.Ge
  | X.Cond.Np -> X.Cond.P
  | X.Cond.P -> X.Cond.Np

let cc c = X.Cond.name c

let cc_of env c =
  cc (if env.mutation = Some Mutation.Inverted_cond then invert c else c)

(* {1 Selected forms} *)

let addr env (a : X.Addr.t) uses =
  (* [uses] are the locations of the address's own values, in order *)
  let base = nth env "use" uses 0 in
  let b = find env (gpr_name ~bits:64 (unit_of env base)) in
  let disp =
    if env.mutation = Some Mutation.Dropped_disp then 0L else a.X.Addr.disp
  in
  match a.X.Addr.index with
  | None -> mem_op ~base:b ~disp ()
  | Some (_, scale) ->
      let i =
        find env (gpr_name ~bits:64 (unit_of env (nth env "use" uses 1)))
      in
      mem_op ~base:b ~index:(i, Int64.to_int scale) ~disp ()

let ( ++ ) = List.append

let instructions env (op : X.t) ~(uses : Loc.t list) ~(defs : Loc.t list) :
    X86_64.Instruction.t list =
  let u = nth env "use" uses and d = nth env "def" defs in
  let mk = make env in
  let gq l = g env ~bits:64 l and gl l = g env ~bits:32 l in
  let gs sz l = g env ~bits:(sz_bits sz) l in
  let x l = xmm env l in
  let fs = X.Fsz.name in
  match op with
  | X.Alu (o, sz, _, _) ->
      [ mk (X.Alu.name o ^ sfx sz) [ gs sz (u 1); gs sz (d 0) ] ]
  | X.Alu_imm (o, sz, _, k) ->
      [ mk (X.Alu.name o ^ sfx sz) [ imm k; gs sz (d 0) ] ]
  | X.Bt (sz, _, k) ->
      [ mk ("bt" ^ sfx sz) [ imm (Int64.of_int k); gs sz (u 0) ] ]
  | X.Call { callee; _ } -> (
      match env.reference (Mir_target.Reference.Call callee) with
      | Some (sym, _) -> [ mk "call" [ Op.Sym (Asm_core.Expr.Symbol sym) ] ]
      | None -> refuse env (R.Form "an unrecorded call"))
  | X.Cmov (sz, c, _, _, _) ->
      [ mk ("cmov" ^ cc_of env c) [ gs sz (u 1); gs sz (d 0) ] ]
  | X.Cmp (sz, _, _) -> [ mk ("cmp" ^ sfx sz) [ gs sz (u 1); gs sz (u 0) ] ]
  | X.Cmp_imm (sz, _, k) -> [ mk ("cmp" ^ sfx sz) [ imm k; gs sz (u 0) ] ]
  | X.Cmps (p, fsz, _, _) ->
      let k = match p with X.Cmp_pred.Eq -> 0L | X.Cmp_pred.Unord -> 3L in
      [ mk ("cmp" ^ fs fsz) [ imm k; x (u 1); x (d 0) ] ]
  | X.Cqo_idiv _ -> [ mk "cqto" []; mk "idivq" [ gq (u 1) ] ]
  | X.Cvt (fsz, _) ->
      [
        mk
          (match fsz with X.Fsz.D -> "cvtss2sd" | X.Fsz.S -> "cvtsd2ss")
          [ x (u 0); x (d 0) ];
      ]
  | X.Cvtpd2ps _ -> [ mk "cvtpd2ps" [ x (u 0); x (d 0) ] ]
  | X.Cvtps2pd _ -> [ mk "cvtps2pd" [ x (u 0); x (d 0) ] ]
  | X.Cvtsi2s (fsz, _) -> [ mk ("cvtsi2" ^ fs fsz ^ "q") [ gq (u 0); x (d 0) ] ]
  | X.Cvtts2si (fsz, _) ->
      [ mk ("cvtt" ^ fs fsz ^ "2si") [ x (u 0); gq (d 0) ] ]
  | X.Ext { signed; from; _ } ->
      let bits, w =
        match from with Mir_width.W8 -> (8, "b") | _ -> (16, "w")
      in
      [
        mk
          ((if signed then "movs" else "movz") ^ w ^ "l")
          [ g env ~bits (u 0); gl (d 0) ];
      ]
  | X.Fbin (o, fsz, _, _) -> [ mk (X.Fop.name o ^ fs fsz) [ x (u 1); x (d 0) ] ]
  | X.Flogic (o, fsz, _, _) ->
      [
        mk
          (X.Flogic.name o ^ match fsz with X.Fsz.D -> "d" | X.Fsz.S -> "s")
          [ x (u 1); x (d 0) ];
      ]
  | X.Fmadd231 (fsz, _, _, _) ->
      [ mk ("vfmadd231" ^ fs fsz) [ x (u 1); x (u 0); x (d 0) ] ]
  | X.Imul (sz, _, _) -> [ mk ("imul" ^ sfx sz) [ gs sz (u 1); gs sz (d 0) ] ]
  | X.Imul_imm (sz, _, k) ->
      [ mk ("imul" ^ sfx sz) [ imm k; gs sz (u 0); gs sz (d 0) ] ]
  | X.Lea (_, k) ->
      let b = find env (gpr_name ~bits:64 (unit_of env (u 0))) in
      [ mk "leaq" [ mem_op ~base:b ~disp:k (); gq (d 0) ] ]
  | X.Lea_view view -> (
      match
        env.reference
          (Mir_target.Reference.View
             (view, Mir_target.Reference.Form.Pc_relative))
      with
      | None -> refuse env (R.Form "an unrecorded view")
      | Some (sym, addend) -> (
          match env.table_slot sym with
          | Some slot ->
              (* the region's base from the caller's table, then the view *)
              let rbp = find env "rbp" in
              let load =
                mk "movq"
                  [
                    mem_op ~base:rbp ~disp:(Int64.of_int (8 * slot)) ();
                    gq (d 0);
                  ]
              in
              if Int64.equal addend 0L then [ load ]
              else if Int64.compare addend 0x7FFF_FFFFL <= 0 then
                let b = find env (gpr_name ~bits:64 (unit_of env (d 0))) in
                [ load; mk "leaq" [ mem_op ~base:b ~disp:addend (); gq (d 0) ] ]
              else
                let r11 = Op.Reg (find env "r11") in
                [
                  load;
                  mk "movq" [ imm addend; r11 ];
                  mk "addq" [ r11; gq (d 0) ];
                ]
          | None ->
              let e =
                if Int64.equal addend 0L then Asm_core.Expr.Symbol sym
                else
                  Asm_core.Expr.Binary
                    ( Asm_core.Expr.Add,
                      Asm_core.Expr.Symbol sym,
                      Asm_core.Expr.Const (Foundation.Bigint.of_int64 addend) )
              in
              [
                mk "leaq"
                  [
                    Op.Mem
                      {
                        Fam.Mem.base = Some Fam.rip_reg;
                        index = None;
                        scale = 1;
                        disp = Fam.Disp.Sym e;
                      };
                    gq (d 0);
                  ];
              ]))
  | X.Load (m, a) -> (
      let mem = addr env a uses in
      match m with
      | X.Msz.B -> [ mk "movzbl" [ mem; gl (d 0) ] ]
      | X.Msz.H -> [ mk "movzwl" [ mem; gl (d 0) ] ]
      | X.Msz.L -> [ mk "movl" [ mem; gl (d 0) ] ]
      | X.Msz.Q -> [ mk "movq" [ mem; gq (d 0) ] ]
      | X.Msz.S -> [ mk "movss" [ mem; x (d 0) ] ]
      | X.Msz.D -> [ mk "movsd" [ mem; x (d 0) ] ])
  | X.Mov (sz, _) -> [ mk ("mov" ^ sfx sz) [ gs sz (u 0); gs sz (d 0) ] ]
  | X.Mov_imm (t, k) -> (
      match t with
      | Mir_type.Int Mir_width.W64 -> [ mk "movq" [ imm k; gq (d 0) ] ]
      | _ -> [ mk "movl" [ imm k; gl (d 0) ] ])
  | X.Movap _ -> [ mk "movaps" [ x (u 0); x (d 0) ] ]
  | X.Movlhps _ -> [ mk "movlhps" [ x (u 1); x (d 0) ] ]
  | X.Movq_from_gpr (fsz, _) -> (
      match fsz with
      | X.Fsz.D -> [ mk "movq" [ gq (u 0); x (d 0) ] ]
      | X.Fsz.S -> [ mk "movd" [ gl (u 0); x (d 0) ] ])
  | X.Movq_low _ | X.Movq_widen _ -> [ mk "movq" [ x (u 0); x (d 0) ] ]
  | X.Movq_to_gpr (fsz, _) -> (
      match fsz with
      | X.Fsz.D -> [ mk "movq" [ x (u 0); gq (d 0) ] ]
      | X.Fsz.S -> [ mk "movd" [ x (u 0); gl (d 0) ] ])
  | X.Movup_load (pk, a) ->
      [ mk ("movu" ^ X.Pk.name pk) [ addr env a uses; x (d 0) ] ]
  | X.Movup_store (pk, a, _) ->
      let n = List.length (X.Addr.uses a) in
      [ mk ("movu" ^ X.Pk.name pk) [ x (u n); addr env a uses ] ]
  | X.Movsxd _ -> [ mk "movslq" [ gl (u 0); gq (d 0) ] ]
  | X.Movzx32 _ | X.Trunc32 _ -> [ mk "movl" [ gl (u 0); gl (d 0) ] ]
  | X.Neg (sz, _) -> [ mk ("neg" ^ sfx sz) [ gs sz (d 0) ] ]
  | X.Pbin (o, pk, _, _) ->
      [ mk (X.Fop.name o ^ X.Pk.name pk) [ x (u 1); x (d 0) ] ]
  | X.Pfmadd231 (pk, _, _, _) ->
      [ mk ("vfmadd231" ^ X.Pk.name pk) [ x (u 1); x (u 0); x (d 0) ] ]
  | X.Plogic (o, pk, _, _) ->
      [
        mk
          (X.Flogic.name o ^ match pk with X.Pk.Pd -> "d" | X.Pk.Ps -> "s")
          [ x (u 1); x (d 0) ];
      ]
  | X.Pshufd_half _ -> [ mk "pshufd" [ imm 0xEEL; x (u 0); x (d 0) ] ]
  | X.Pshufd_lane (fsz, k, _) ->
      let i =
        match fsz with
        | X.Fsz.S -> k * 0x55
        | X.Fsz.D -> if k = 0 then 0x44 else 0xEE
      in
      [ mk "pshufd" [ imm (Int64.of_int i); x (u 0); x (d 0) ] ]
  | X.Pshufd_splat (pk, _) ->
      [
        mk "pshufd"
          [
            imm (match pk with X.Pk.Ps -> 0L | X.Pk.Pd -> 0x44L);
            x (u 0);
            x (d 0);
          ];
      ]
  | X.Psqrt (pk, _) -> [ mk ("sqrt" ^ X.Pk.name pk) [ x (u 0); x (d 0) ] ]
  | X.Round_trunc (fsz, _) ->
      [ mk ("round" ^ fs fsz) [ imm 3L; x (u 0); x (d 0) ] ]
  | X.Setcc_zx (c, _) ->
      [
        mk ("set" ^ cc_of env c) [ g env ~bits:8 (d 0) ];
        mk "movzbl" [ g env ~bits:8 (d 0); gl (d 0) ];
      ]
  | X.Shift_imm (o, sz, _, k) ->
      [ mk (X.Shift.name o ^ sfx sz) [ imm (Int64.of_int k); gs sz (d 0) ] ]
  | X.Sqrt (fsz, _) -> [ mk ("sqrt" ^ fs fsz) [ x (u 0); x (d 0) ] ]
  | X.Store (m, a, _) -> (
      let n = List.length (X.Addr.uses a) in
      let mem = addr env a uses and v = u n in
      match m with
      | X.Msz.B -> [ mk "movb" [ g env ~bits:8 v; mem ] ]
      | X.Msz.H -> [ mk "movw" [ g env ~bits:16 v; mem ] ]
      | X.Msz.L -> [ mk "movl" [ gl v; mem ] ]
      | X.Msz.Q -> [ mk "movq" [ gq v; mem ] ]
      | X.Msz.S -> [ mk "movss" [ x v; mem ] ]
      | X.Msz.D -> [ mk "movsd" [ x v; mem ] ])
  | X.Test (sz, _, _) -> [ mk ("test" ^ sfx sz) [ gs sz (u 1); gs sz (u 0) ] ]
  | X.Trunc_zx (w, _) ->
      let bits, n = match w with Mir_width.W8 -> (8, "b") | _ -> (16, "w") in
      [ mk ("movz" ^ n ^ "l") [ g env ~bits (u 0); gl (d 0) ] ]
  | X.Ucomis (fsz, _, _) -> [ mk ("ucomi" ^ fs fsz) [ x (u 1); x (u 0) ] ]

(* {1 Allocation-added forms} *)

let transfer env ~(dst : Loc.t) ~(src : Loc.t) : X86_64.Instruction.t list =
  let mk = make env in
  match (dst, src) with
  | Loc.Reg d, Loc.Reg s -> (
      match (d.Mir_target.View.bank, s.Mir_target.View.bank) with
      | Mir_target.Bank.Gpr, Mir_target.Bank.Gpr ->
          let bits = min d.Mir_target.View.bits s.Mir_target.View.bits in
          [
            mk
              (if bits = 64 then "movq" else "movl")
              [ g env ~bits (Loc.Reg s); g env ~bits (Loc.Reg d) ];
          ]
      | Mir_target.Bank.Fpr, Mir_target.Bank.Fpr ->
          [ mk "movaps" [ xmm env (Loc.Reg s); xmm env (Loc.Reg d) ] ]
      | _ -> refuse env (R.Location "a register transfer across banks"))
  | Loc.Mem { base; offset; bytes }, Loc.Reg r
  | Loc.Reg r, Loc.Mem { base; offset; bytes } -> (
      let load = match dst with Loc.Reg _ -> true | _ -> false in
      let b = find env (gpr_name ~bits:64 (unit_of env (Loc.Reg base))) in
      let m = mem_op ~base:b ~disp:offset () in
      let pair a z = if load then [ mk a [ m; z ] ] else [ mk a [ z; m ] ] in
      match (r.Mir_target.View.bank, bytes) with
      | Mir_target.Bank.Gpr, 8L -> pair "movq" (g env ~bits:64 (Loc.Reg r))
      | Mir_target.Bank.Gpr, 4L -> pair "movl" (g env ~bits:32 (Loc.Reg r))
      | Mir_target.Bank.Fpr, 16L -> pair "movups" (xmm env (Loc.Reg r))
      | Mir_target.Bank.Fpr, 8L -> pair "movsd" (xmm env (Loc.Reg r))
      | Mir_target.Bank.Fpr, 4L -> pair "movss" (xmm env (Loc.Reg r))
      | _ -> refuse env (R.Location "a frame access of this width"))
  | Loc.Slot _, _ | _, Loc.Slot _ -> refuse env (R.Location "a frame slot")
  | Loc.Mem _, Loc.Mem _ -> refuse env (R.Location "memory to memory")

let stack_step env delta : X86_64.Instruction.t list =
  let rsp = Op.Reg (find env "rsp") in
  if Int64.compare delta 0L < 0 then
    [ make env "subq" [ imm (Int64.neg delta); rsp ] ]
  else [ make env "addq" [ imm delta; rsp ] ]

(* {1 Terminators} *)

let branch env (X.Jcc (c, _)) ~label ~inverted =
  let c = if inverted then invert c else c in
  make env ("j" ^ cc c) [ Op.Sym (Asm_core.Expr.Symbol label) ]

let jump env label = make env "jmp" [ Op.Sym (Asm_core.Expr.Symbol label) ]
let ret env = make env "ret" []
