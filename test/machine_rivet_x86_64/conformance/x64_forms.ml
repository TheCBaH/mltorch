(* The x86-64 form instances the conformance harness probes: each admitted
   form with its immediates, conditions and sizes enumerated, the role of each
   of its operands, and its selected-IR counterpart. *)

open Machine_ir
open Machine_target_x86_64
open X64_op

(* What a use of the form is: the condition value, a value of a type, the
   pointer a memory form is addressed from, or an address's index. *)
type role = Flags | Val of Mir_type.t | Base | Index

type t = {
  name : string;
  roles : role list;  (** in the order of [X64_op.uses] *)
  make : Mir_value.t list -> X64_op.t;
}

let ti = function Sz.L -> Mir_type.i32 | Sz.Q -> Mir_type.i64
let fty = function Fsz.D -> Mir_type.F64 | Fsz.S -> Mir_type.F32
let conds = Cond.[ A; Ae; B; Be; E; G; Ge; L; Le; Ne; Np; P ]
let szs = [ Sz.L; Sz.Q ]
let fszs = [ Fsz.D; Fsz.S ]
let pks = [ Pk.Pd; Pk.Ps ]
let nth vs k = List.nth vs k
let form name roles make = { name; roles; make }

let imm_for = function
  | Sz.L ->
      [
        0L;
        1L;
        5L;
        127L;
        128L;
        255L;
        256L;
        0xFFFF_FF80L;
        0xFFFF_FF7FL;
        0xFFFF_FFFFL;
        0x7FFF_FFFFL;
        0x8000_0000L;
      ]
  | Sz.Q ->
      [
        0L;
        1L;
        5L;
        127L;
        128L;
        255L;
        256L;
        -128L;
        -129L;
        -1L;
        0x7FFF_FFFFL;
        -0x8000_0000L;
      ]

let alus = Alu.[ Add; And; Or; Sub; Xor ]
let shifts = Shift.[ Sar; Shl; Shr ]
let fops = Fop.[ Add; Div; Max; Mul; Sub ]
let pops = Fop.[ Add; Div; Mul; Sub ]
let logics = Flogic.[ And; Andn; Or; Xor ]
let sz_name = Sz.name
let fs_name = Fsz.name

(* addresses: base + disp, and base + index * scale + disp; the base points 8
   bytes into a 32-byte buffer *)
let addrs =
  [
    ( "[b]",
      [ Base ],
      fun vs -> { Addr.base = nth vs 0; index = None; disp = 0L } );
    ( "[b+8]",
      [ Base ],
      fun vs -> { Addr.base = nth vs 0; index = None; disp = 8L } );
    ( "[b-8]",
      [ Base ],
      fun vs -> { Addr.base = nth vs 0; index = None; disp = -8L } );
    ( "[b+i]",
      [ Base; Index ],
      fun vs -> { Addr.base = nth vs 0; index = Some (nth vs 1, 1L); disp = 0L }
    );
    ( "[b+2i+4]",
      [ Base; Index ],
      fun vs -> { Addr.base = nth vs 0; index = Some (nth vs 1, 2L); disp = 4L }
    );
    ( "[b+4i-4]",
      [ Base; Index ],
      fun vs ->
        { Addr.base = nth vs 0; index = Some (nth vs 1, 4L); disp = -4L } );
    ( "[b+8i]",
      [ Base; Index ],
      fun vs -> { Addr.base = nth vs 0; index = Some (nth vs 1, 8L); disp = 0L }
    );
  ]

let msz_ty = function
  | Msz.B -> Mir_type.i8
  | Msz.H -> Mir_type.i16
  | Msz.L -> Mir_type.i32
  | Msz.Q -> Mir_type.i64
  | Msz.S -> Mir_type.F32
  | Msz.D -> Mir_type.F64

let msz_name = Msz.name
let ty_i = Mir_type.i64

let all : t list =
  let gpr_forms sz =
    let n = sz_name sz and t = ti sz in
    List.concat
      [
        List.map
          (fun o ->
            form
              (Printf.sprintf "%s%s" (Alu.name o) n)
              [ Val t; Val t ]
              (fun vs -> Alu (o, sz, nth vs 0, nth vs 1)))
          alus;
        List.concat_map
          (fun o ->
            List.map
              (fun k ->
                form
                  (Printf.sprintf "%s%s $%Ld" (Alu.name o) n k)
                  [ Val t ]
                  (fun vs -> Alu_imm (o, sz, nth vs 0, k)))
              (imm_for sz))
          alus;
        List.map
          (fun k ->
            form (Printf.sprintf "bt%s $%d" n k) [ Val t ] (fun vs ->
                Bt (sz, nth vs 0, k)))
          [ 0; 1; 5; Sz.bits sz - 1 ];
        List.map
          (fun c ->
            form
              (Printf.sprintf "cmov%s%s" (Cond.name c) n)
              [ Flags; Val t; Val t ]
              (fun vs -> Cmov (sz, c, nth vs 0, nth vs 1, nth vs 2)))
          conds;
        [
          form ("cmp" ^ n) [ Val t; Val t ] (fun vs ->
              Cmp (sz, nth vs 0, nth vs 1));
          form ("test" ^ n) [ Val t; Val t ] (fun vs ->
              Test (sz, nth vs 0, nth vs 1));
          form ("imul" ^ n) [ Val t; Val t ] (fun vs ->
              Imul (sz, nth vs 0, nth vs 1));
          form ("mov" ^ n) [ Val t ] (fun vs -> Mov (sz, nth vs 0));
          form ("neg" ^ n) [ Val t ] (fun vs -> Neg (sz, nth vs 0));
        ];
        List.map
          (fun k ->
            form (Printf.sprintf "cmp%s $%Ld" n k) [ Val t ] (fun vs ->
                Cmp_imm (sz, nth vs 0, k)))
          (imm_for sz);
        List.map
          (fun k ->
            form (Printf.sprintf "imul%s $%Ld" n k) [ Val t ] (fun vs ->
                Imul_imm (sz, nth vs 0, k)))
          (imm_for sz);
        List.concat_map
          (fun o ->
            List.map
              (fun k ->
                form
                  (Printf.sprintf "%s%s $%d" (Shift.name o) n k)
                  [ Val t ]
                  (fun vs -> Shift_imm (o, sz, nth vs 0, k)))
              [ 0; 1; 7; Sz.bits sz - 1 ])
          shifts;
      ]
  in
  let sse_forms fsz =
    let n = fs_name fsz and t = fty fsz in
    List.concat
      [
        List.map
          (fun p ->
            form
              (Printf.sprintf "cmp%s%s" (Cmp_pred.name p) n)
              [ Val t; Val t ]
              (fun vs -> Cmps (p, fsz, nth vs 0, nth vs 1)))
          Cmp_pred.[ Eq; Unord ];
        List.map
          (fun o ->
            form
              (Printf.sprintf "%s%s" (Fop.name o) n)
              [ Val t; Val t ]
              (fun vs -> Fbin (o, fsz, nth vs 0, nth vs 1)))
          fops;
        List.map
          (fun o ->
            form
              (Printf.sprintf "%s%s" (Flogic.name o)
                 (match fsz with Fsz.D -> "d" | Fsz.S -> "s"))
              [ Val t; Val t ]
              (fun vs -> Flogic (o, fsz, nth vs 0, nth vs 1)))
          logics;
        [
          form ("vfmadd231" ^ n) [ Val t; Val t; Val t ] (fun vs ->
              Fmadd231 (fsz, nth vs 0, nth vs 1, nth vs 2));
          form ("round" ^ n) [ Val t ] (fun vs -> Round_trunc (fsz, nth vs 0));
          form ("sqrt" ^ n) [ Val t ] (fun vs -> Sqrt (fsz, nth vs 0));
          form ("ucomi" ^ n) [ Val t; Val t ] (fun vs ->
              Ucomis (fsz, nth vs 0, nth vs 1));
          form ("cvtsi2" ^ n) [ Val ty_i ] (fun vs -> Cvtsi2s (fsz, nth vs 0));
          form
            ("cvtt" ^ n ^ "2si")
            [ Val t ]
            (fun vs -> Cvtts2si (fsz, nth vs 0));
          form ("movq.from_gpr." ^ n)
            [
              Val
                (match fsz with Fsz.D -> Mir_type.i64 | Fsz.S -> Mir_type.i32);
            ]
            (fun vs -> Movq_from_gpr (fsz, nth vs 0));
          form ("movq.to_gpr." ^ n) [ Val t ] (fun vs ->
              Movq_to_gpr (fsz, nth vs 0));
          form ("movap." ^ n) [ Val t ] (fun vs -> Movap (nth vs 0));
        ];
        [
          form ("cvt." ^ n)
            [ Val (fty (match fsz with Fsz.D -> Fsz.S | Fsz.S -> Fsz.D)) ]
            (fun vs -> Cvt (fsz, nth vs 0));
        ];
      ]
  in
  let packed_forms pk =
    let n = Pk.name pk and t = Pk.ty pk in
    List.concat
      [
        List.map
          (fun o ->
            form
              (Printf.sprintf "%s%s" (Fop.name o) n)
              [ Val t; Val t ]
              (fun vs -> Pbin (o, pk, nth vs 0, nth vs 1)))
          pops;
        List.map
          (fun o ->
            form
              (Printf.sprintf "%s%s" (Flogic.name o)
                 (match pk with Pk.Pd -> "d" | Pk.Ps -> "s"))
              [ Val t; Val t ]
              (fun vs -> Plogic (o, pk, nth vs 0, nth vs 1)))
          logics;
        [
          form ("vfmadd231" ^ n) [ Val t; Val t; Val t ] (fun vs ->
              Pfmadd231 (pk, nth vs 0, nth vs 1, nth vs 2));
          form ("sqrt" ^ n) [ Val t ] (fun vs -> Psqrt (pk, nth vs 0));
          form ("pshufd.splat" ^ n)
            [ Val (fty (Pk.fsz pk)) ]
            (fun vs -> Pshufd_splat (pk, nth vs 0));
        ];
        List.concat_map
          (fun (an, roles, addr) ->
            [
              form (Printf.sprintf "movu%s load %s" n an) roles (fun vs ->
                  Movup_load (pk, addr vs));
              form (Printf.sprintf "movu%s store %s" n an) (roles @ [ Val t ])
                (fun vs ->
                  Movup_store (pk, addr vs, nth vs (List.length roles)));
            ])
          addrs;
        List.init (Pk.lanes pk) (fun k ->
            form (Printf.sprintf "pshufd.lane%s[%d]" n k) [ Val t ] (fun vs ->
                Pshufd_lane (Pk.fsz pk, k, nth vs 0)));
      ]
  in
  let memory_forms =
    List.concat_map
      (fun m ->
        List.concat_map
          (fun (an, roles, addr) ->
            [
              form
                (Printf.sprintf "load.%s %s" (msz_name m) an)
                roles
                (fun vs -> Load (m, addr vs));
              form
                (Printf.sprintf "store.%s %s" (msz_name m) an)
                (roles @ [ Val (msz_ty m) ])
                (fun vs -> Store (m, addr vs, nth vs (List.length roles)));
            ])
          addrs)
      Msz.[ B; D; H; L; Q; S ]
  in
  let misc =
    List.concat
      [
        List.concat_map
          (fun signed ->
            List.map
              (fun from ->
                form
                  (Printf.sprintf "mov%s%sl"
                     (if signed then "s" else "z")
                     (match from with Mir_width.W8 -> "b" | _ -> "w"))
                  [ Val (Mir_type.Int from) ]
                  (fun vs -> Ext { signed; from; src = nth vs 0 }))
              Mir_width.[ W8; W16 ])
          [ false; true ];
        List.map
          (fun c ->
            form
              ("set" ^ Cond.name c)
              [ Flags ]
              (fun vs -> Setcc_zx (c, nth vs 0)))
          conds;
        List.map
          (fun w ->
            form
              (Printf.sprintf "movz%sl.trunc"
                 (match w with Mir_width.W8 -> "b" | _ -> "w"))
              [ Val Mir_type.i32 ]
              (fun vs -> Trunc_zx (w, nth vs 0)))
          Mir_width.[ W8; W16 ];
        [
          form "movl.trunc" [ Val Mir_type.i64 ] (fun vs -> Trunc32 (nth vs 0));
          form "movl.zx" [ Val Mir_type.i32 ] (fun vs -> Movzx32 (nth vs 0));
          form "movslq" [ Val Mir_type.i32 ] (fun vs -> Movsxd (nth vs 0));
          form "cqo;idivq" [ Val Mir_type.i64; Val Mir_type.i64 ] (fun vs ->
              Cqo_idiv (nth vs 0, nth vs 1));
          form "cvtpd2ps" [ Val (Pk.ty Pk.Pd) ] (fun vs -> Cvtpd2ps (nth vs 0));
          form "cvtps2pd" [ Val half_ty ] (fun vs -> Cvtps2pd (nth vs 0));
          form "movlhps"
            [ Val (Pk.ty Pk.Ps); Val half_ty ]
            (fun vs -> Movlhps (nth vs 0, nth vs 1));
          form "movq.low" [ Val (Pk.ty Pk.Ps) ] (fun vs -> Movq_low (nth vs 0));
          form "movq.widen" [ Val half_ty ] (fun vs -> Movq_widen (nth vs 0));
          form "pshufd.half"
            [ Val (Pk.ty Pk.Ps) ]
            (fun vs -> Pshufd_half (nth vs 0));
          form "leaq" [ Base ] (fun vs -> Lea (nth vs 0, 24L));
          form "leaq-8" [ Base ] (fun vs -> Lea (nth vs 0, -8L));
          form "addq ptr" [ Base; Val Mir_type.i64 ] (fun vs ->
              Alu (Alu.Add, Sz.Q, nth vs 0, nth vs 1));
          form "subq ptr" [ Base; Val Mir_type.i64 ] (fun vs ->
              Alu (Alu.Sub, Sz.Q, nth vs 0, nth vs 1));
        ];
        List.map
          (fun k ->
            form (Printf.sprintf "movabs $%Ld" k) [] (fun _ ->
                Mov_imm (Mir_type.i64, k)))
          [
            0L;
            1L;
            -1L;
            0x7FFF_FFFFL;
            0x8000_0000L;
            Int64.min_int;
            0x1234_5678_9ABC_DEF0L;
          ];
        List.map
          (fun k ->
            form (Printf.sprintf "movl $%Ld" k) [] (fun _ ->
                Mov_imm (Mir_type.i32, k)))
          [ 0L; 1L; 0x7FFF_FFFFL; 0x8000_0000L; 0xFFFF_FFFFL ];
        [
          form "movl $0 pred" [] (fun _ -> Mov_imm (Mir_type.Pred, 0L));
          form "movl $1 pred" [] (fun _ -> Mov_imm (Mir_type.Pred, 1L));
        ];
      ]
  in
  List.concat
    [
      List.concat_map gpr_forms szs;
      List.concat_map sse_forms fszs;
      List.concat_map packed_forms pks;
      memory_forms;
      misc;
    ]
