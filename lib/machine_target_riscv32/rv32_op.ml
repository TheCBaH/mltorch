(* The admitted RV32 selected forms over virtual values: RV32I, M and F (the
   RISC-V unprivileged ISA, 20240411). Each constructor is one instruction, or
   a short fixed sequence the emitter spells out (a wide constant add, a select,
   a pair of words); the sequences are marked. Every general register is 32
   bits: an i8 or i16 value is held zero-extended, a predicate is 0 or 1, a
   pointer is a 32-bit address. Every consumer matches this type exhaustively
   and alphabetically. Immediates carry only values the form can encode; the
   typing rule refuses any other. *)

open Machine_ir

(* Register-register integer operations. *)
module Alu = struct
  type t =
    | Add
    | And
    | Div
    | Mul
    | Mulhu
    | Or
    | Rem
    | Sll  (** by the low five bits of the second operand *)
    | Slt
    | Sltu
    | Srl  (** likewise *)
    | Sub
    | Xor

  let name = function
    | Add -> "add"
    | And -> "and"
    | Div -> "div"
    | Mul -> "mul"
    | Mulhu -> "mulhu"
    | Or -> "or"
    | Rem -> "rem"
    | Sll -> "sll"
    | Slt -> "slt"
    | Sltu -> "sltu"
    | Srl -> "srl"
    | Sub -> "sub"
    | Xor -> "xor"

  let compares = function Slt | Sltu -> true | _ -> false
end

(* Register-immediate integer operations: a signed 12-bit immediate, or a
   shift amount below 32. *)
module Alui = struct
  type t = Andi | Ori | Slli | Slti | Sltiu | Srai | Srli | Xori

  let name = function
    | Andi -> "andi"
    | Ori -> "ori"
    | Slli -> "slli"
    | Slti -> "slti"
    | Sltiu -> "sltiu"
    | Srai -> "srai"
    | Srli -> "srli"
    | Xori -> "xori"

  let is_shift = function Slli | Srai | Srli -> true | _ -> false
  let compares = function Slti | Sltiu -> true | _ -> false
end

(* Branch conditions on two registers. *)
module Bcond = struct
  type t = Eq | Ge | Geu | Lt | Ltu | Ne

  let name = function
    | Eq -> "beq"
    | Ge -> "bge"
    | Geu -> "bgeu"
    | Lt -> "blt"
    | Ltu -> "bltu"
    | Ne -> "bne"

  let invert = function
    | Eq -> Ne
    | Ge -> Lt
    | Geu -> Ltu
    | Lt -> Ge
    | Ltu -> Geu
    | Ne -> Eq
end

module Fcmp = Rv32_kind.Fcmp
module Fop = Rv32_kind.Fop
module Sgnj = Rv32_kind.Sgnj

(* A memory access size. A load zero-extends ([lbu], [lhu], [lw]). *)
module Msz = struct
  type t = B | H | W

  let bytes = function B -> 1L | H -> 2L | W -> 4L

  let width = function
    | B -> Mir_width.W8
    | H -> Mir_width.W16
    | W -> Mir_width.W32

  let load = function B -> "lbu" | H -> "lhu" | W -> "lw"
  let store = function B -> "sb" | H -> "sh" | W -> "sw"
end

type v = Mir_value.t

type t =
  | Add_wide of v * int64
      (** sequence: [lui rd, hi; addi rd, rd, lo; add rd, rs, rd], a pointer
          plus any 32-bit constant; the result must not share the operand's
          register *)
  | Addi of v * int64
      (** [addi]: an integer or pointer plus a 12-bit constant *)
  | Alu of Alu.t * v * v
  | Alui of Alui.t * v * int64
  | Call of {
      callee : Mir_op.Callee.t;
      args : v list;
      results : Mir_type.t list;
    }
      (** [call] under ILP32F: arguments and results in their fixed registers,
          then the [i32] status in a0 after them; every caller-saved register
          clobbered *)
  | Fbin of Fop.t * v * v
  | Fcmp of Fcmp.t * v * v
  | Fcvt_s_w of v  (** an i32 to binary32, one rounding *)
  | Fcvt_s_wu of v  (** an i32's bits read unsigned to binary32, one rounding *)
  | Fcvt_w_s of v
      (** binary32 to i32 toward zero, saturating; a NaN gives 2^31 - 1 *)
  | Flw of v * int64  (** base, 12-bit offset *)
  | Fmadd of v * v * v  (** [a * b + c], one rounding *)
  | Fmv_w_x of v  (** an i32's bits as binary32 *)
  | Fmv_x_w of v  (** binary32 bits as an i32 *)
  | Fsel of v * v * v
      (** sequence: a predicate, then, else, as branches around moves *)
  | Fsgnj of Sgnj.t * v * v
  | Fsqrt of v
  | Fsw of v * int64 * v  (** base, offset, value *)
  | La of Mir_id.View.t  (** the view's address *)
  | Li of Mir_type.t * int64
      (** [addi rd, zero, k]: a 12-bit constant, typed as the value it makes
          canonical *)
  | Load of Msz.t * v * int64  (** base, 12-bit offset; zero-extending *)
  | Lui of int64  (** [k << 12] for a 20-bit [k], an i32 *)
  | Mv of v  (** [addi rd, rs, 0]: a whole register *)
  | Sel of v * v * v
      (** sequence: a predicate, then, else, as branches around moves *)
  | Sext of Mir_width.t * v
      (** sequence: [slli] then [srai], an i8 or i16 sign-extended to i32 *)
  | Store of Msz.t * v * int64 * v  (** base, offset, value *)
  | Sw_pair of v * int64 * v * v
      (** sequence: two [sw], base, offset, the low then the high word *)
  | Trunc of Mir_width.t * v
      (** the low byte ([andi]) or halfword ([slli] then [srli]) of an i32 *)
  | Vec of Rv32_vop.t  (** a Zve32f vector form *)
  | Zext of v
      (** [addi rd, rs, 0]: an i8, i16 or predicate as the i32 it already is *)

type test = Bcmp of Bcond.t * v * v | Beqz of v | Bnez of v

let uses = function
  | Alu (_, a, b) | Fbin (_, a, b) | Fcmp (_, a, b) | Fsgnj (_, a, b) ->
      [ a; b ]
  | Add_wide (a, _)
  | Addi (a, _)
  | Alui (_, a, _)
  | Fcvt_s_w a
  | Fcvt_s_wu a
  | Fcvt_w_s a
  | Flw (a, _)
  | Fmv_w_x a
  | Fmv_x_w a
  | Fsqrt a
  | Load (_, a, _)
  | Mv a
  | Sext (_, a)
  | Trunc (_, a)
  | Zext a ->
      [ a ]
  | La _ | Li _ | Lui _ -> []
  | Call { args; _ } -> args
  | Fmadd (a, b, c) | Fsel (a, b, c) | Sel (a, b, c) -> [ a; b; c ]
  | Fsw (base, _, x) | Store (_, base, _, x) -> [ base; x ]
  | Sw_pair (base, _, lo, hi) -> [ base; lo; hi ]
  | Vec x -> Rv32_vop.uses x

let test_uses = function Bcmp (_, a, b) -> [ a; b ] | Beqz a | Bnez a -> [ a ]

let ordered = function
  | Call _ | Flw _ | Fsw _ | Load _ | Store _ | Sw_pair _ -> true
  | Vec x -> Rv32_vop.ordered x
  | _ -> false

(* The forms whose sequence writes v0 on its way (see [Rv32_mir_form]): a
   compare's mask, a slide's temporary, the scalar a lane insert slides up. *)
let vector_scratch (x : Rv32_vop.t) =
  match x with
  | Rv32_vop.Vext { first; _ } when first <> 0 -> [ Rv32_reg.vector_scratch ]
  | Rv32_vop.Vfcmp _ | Rv32_vop.Vinsert _ | Rv32_vop.Vmerge _ ->
      [ Rv32_reg.vector_scratch ]
  | Rv32_vop.Vfmv_f_s (k, _) when k <> 0 -> [ Rv32_reg.vector_scratch ]
  | Rv32_vop.Vcast _ | Rv32_vop.Vext _ | Rv32_vop.Vfbin _ | Rv32_vop.Vfcvt_f_x _
  | Rv32_vop.Vfcvt_x_f _ | Rv32_vop.Vfmacc _ | Rv32_vop.Vfmv_f_s _
  | Rv32_vop.Vfmv_v_f _ | Rv32_vop.Vfsqrt _ | Rv32_vop.Vle32 _
  | Rv32_vop.Vle_int _ | Rv32_vop.Vlogic _ | Rv32_vop.Vlse32 _ | Rv32_vop.Vmv _
  | Rv32_vop.Vmv_v_x _ | Rv32_vop.Vnot _ | Rv32_vop.Vse32 _ | Rv32_vop.Vshift _
  | Rv32_vop.Vsgnj _ | Rv32_vop.Vsse32 _ ->
      []

(* Integers and pointers take the next of a0-a7, floats the next of fa0-fa7;
   the status follows the results in the integer sequence. *)
let sequence tys =
  let gi = ref 0 and fi = ref 0 in
  List.map
    (fun (t : Mir_type.t) ->
      let fp = Mir_type.equal t Mir_type.F32 in
      let k = if fp then fi else gi in
      let r = if fp then Rv32_reg.fa !k else Rv32_reg.a !k in
      incr k;
      if !k > 8 then invalid_arg "Rv32_op: more than eight of one class";
      r)
    tys

let constraints = function
  | Call { args; results; _ } ->
      List.mapi
        (fun use view -> Mir_target.Constraint.Fixed_use { use; view })
        (sequence (List.map (fun (a : v) -> a.Mir_value.ty) args))
      @ List.mapi
          (fun result view ->
            Mir_target.Constraint.Fixed_result { result; view })
          (sequence (results @ [ Mir_type.i32 ]))
  | Add_wide _ -> [ Mir_target.Constraint.Early_clobber 0 ]
  | Vec x ->
      Rv32_vop.constraints x
      @ List.map (fun v -> Mir_target.Constraint.Scratch v) (vector_scratch x)
  | _ -> []

(* Every register is one 32-bit view of its unit. *)
let result_write _ = Mir_target.Write.Zero_upper

let unit_bits = function
  | Mir_target.Bank.Control -> 32
  | Mir_target.Bank.Flags -> 1
  | Mir_target.Bank.Fpr -> 32
  | Mir_target.Bank.Gpr -> 32
  | Mir_target.Bank.Vector -> 128

(* A call leaves undefined every register the psABI does not preserve: ra,
   t0-t6, a0-a7, ft0-ft11, fa0-fa7 and every vector register. *)
let call_clobbers =
  List.map Rv32_reg.x
    ([ 1; 5; 6; 7 ] @ Rv32_reg.range 10 17 @ Rv32_reg.range 28 31)
  @ List.map Rv32_reg.f
      (Rv32_reg.range 0 7 @ Rv32_reg.range 10 17 @ Rv32_reg.range 28 31)
  @ List.map Rv32_reg.v (Rv32_reg.range 0 31)

let clobbers = function
  | Call _ -> call_clobbers
  | Vec x -> vector_scratch x
  | _ -> []

let references = function
  | Call { callee; _ } -> [ Mir_target.Reference.Call callee ]
  | La view ->
      [
        Mir_target.Reference.View (view, Mir_target.Reference.Form.Pc_relative);
      ]
  | _ -> []

let writes_flags _ = false
let flags_defined _ = 0L
let flags_read _ = 0L
let test_flags_read _ = 0L
let op_features = function Vec _ -> [ Mir_target.Feature.Zve32f ] | _ -> []
let imm12 k = Int64.compare k (-2048L) >= 0 && Int64.compare k 2047L <= 0
let imm20 k = Int64.compare k 0L >= 0 && Int64.compare k 0xFFFFFL <= 0
let ( let* ) = Result.bind
let ty (v : v) = v.Mir_value.ty
let need ok why = if ok then Ok () else Error why

(* An integer held in a general register, canonical at its width. *)
let int_like (t : Mir_type.t) =
  match t with
  | Mir_type.Int (Mir_width.W8 | Mir_width.W16 | Mir_width.W32) | Mir_type.Pred
    ->
      true
  | _ -> false

let is t u = Mir_type.equal t u
let i32 t = is t Mir_type.i32
let ptr t = is t Mir_type.Ptr
let f32 t = is t Mir_type.F32
let gpr t = int_like t || ptr t

let typing op =
  match op with
  | Add_wide (a, k) ->
      let* () = need (ptr (ty a) || i32 (ty a)) "wide add operand" in
      let* () =
        need
          (Int64.compare k Int64.(neg 0x8000_0000L) >= 0
          && Int64.compare k 0x7FFF_FFFFL <= 0)
          "wide add constant"
      in
      Ok [ ty a ]
  | Addi (a, k) ->
      let* () = need (ptr (ty a) || i32 (ty a)) "addi operand" in
      let* () = need (imm12 k) "addi immediate" in
      Ok [ ty a ]
  | Alu (o, a, b) -> (
      match (o, ty a, ty b) with
      | Alu.Add, t, u when ptr t && (i32 u || is u Mir_type.Pred) -> Ok [ t ]
      | Alu.Add, t, u when ptr u && i32 t -> Ok [ u ]
      | (Alu.Add | Alu.Sub), t, u when i32 t && is u Mir_type.Pred ->
          (* a carry or borrow: a predicate is 0 or 1 *)
          Ok [ t ]
      | (Alu.Slt | Alu.Sltu), t, u when i32 t && i32 u -> Ok [ Mir_type.Pred ]
      | (Alu.And | Alu.Or | Alu.Xor), t, u
        when is t Mir_type.Pred && is u Mir_type.Pred ->
          Ok [ t ]
      | (Alu.Slt | Alu.Sltu), _, _ -> Error "compare operands"
      | _, t, u when i32 t && i32 u -> Ok [ t ]
      | _ -> Error "alu operands")
  | Alui (o, a, k) -> (
      let* () =
        need
          (if Alui.is_shift o then
             Int64.compare k 0L >= 0 && Int64.compare k 32L < 0
           else imm12 k)
          "immediate"
      in
      match (o, ty a) with
      | (Alui.Slti | Alui.Sltiu), t when i32 t -> Ok [ Mir_type.Pred ]
      | (Alui.Xori | Alui.Andi | Alui.Ori), t when is t Mir_type.Pred ->
          (* the result stays 0 or 1 *)
          let* () =
            need (Int64.equal k 1L || Int64.equal k 0L) "predicate immediate"
          in
          Ok [ t ]
      | (Alui.Slti | Alui.Sltiu), _ -> Error "compare operand"
      | _, t when i32 t -> Ok [ t ]
      | _ -> Error "alui operand")
  | Call { args; results; _ } ->
      let ok t = gpr t || f32 t in
      let* () =
        need (List.for_all ok (results @ List.map ty args)) "call operand types"
      in
      let fits tys =
        match sequence tys with
        | _ -> true
        | exception Invalid_argument _ -> false
      in
      let* () =
        need
          (fits (List.map ty args) && fits (results @ [ Mir_type.i32 ]))
          "more call operands than registers"
      in
      Ok (results @ [ Mir_type.i32 ])
  | Fbin (_, a, b) ->
      let* () = need (f32 (ty a) && f32 (ty b)) "fp operands" in
      Ok [ Mir_type.F32 ]
  | Fcmp (_, a, b) ->
      let* () = need (f32 (ty a) && f32 (ty b)) "fcmp operands" in
      Ok [ Mir_type.Pred ]
  | Fcvt_s_w a | Fcvt_s_wu a ->
      let* () = need (i32 (ty a)) "fcvt source" in
      Ok [ Mir_type.F32 ]
  | Fcvt_w_s a ->
      let* () = need (f32 (ty a)) "fcvt source" in
      Ok [ Mir_type.i32 ]
  | Flw (base, k) ->
      let* () = need (ptr (ty base)) "flw base" in
      let* () = need (imm12 k) "flw offset" in
      Ok [ Mir_type.F32 ]
  | Fmadd (a, b, c) ->
      let* () =
        need (f32 (ty a) && f32 (ty b) && f32 (ty c)) "fmadd operands"
      in
      Ok [ Mir_type.F32 ]
  | Fmv_w_x a ->
      let* () = need (i32 (ty a)) "fmv source" in
      Ok [ Mir_type.F32 ]
  | Fmv_x_w a ->
      let* () = need (f32 (ty a)) "fmv source" in
      Ok [ Mir_type.i32 ]
  | Fsel (p, a, b) ->
      let* () = need (is (ty p) Mir_type.Pred) "select predicate" in
      let* () = need (f32 (ty a) && f32 (ty b)) "select operands" in
      Ok [ Mir_type.F32 ]
  | Fsgnj (_, a, b) ->
      let* () = need (f32 (ty a) && f32 (ty b)) "sign injection operands" in
      Ok [ Mir_type.F32 ]
  | Fsqrt a ->
      let* () = need (f32 (ty a)) "fsqrt operand" in
      Ok [ Mir_type.F32 ]
  | Fsw (base, k, x) ->
      let* () = need (ptr (ty base)) "fsw base" in
      let* () = need (imm12 k) "fsw offset" in
      let* () = need (f32 (ty x)) "fsw value" in
      Ok []
  | La _ -> Ok [ Mir_type.Ptr ]
  | Li (t, k) ->
      let* () = need (int_like t) "li type" in
      let* () = need (imm12 k) "li immediate" in
      let* () =
        need
          (match t with
          | Mir_type.Int w ->
              Mir_width.canonical w (Int64.logand k 0xFFFF_FFFFL)
          | _ -> Int64.equal k 0L || Int64.equal k 1L)
          "li constant outside its type"
      in
      Ok [ t ]
  | Load (m, base, k) ->
      let* () = need (ptr (ty base)) "load base" in
      let* () = need (imm12 k) "load offset" in
      Ok [ Mir_type.Int (Msz.width m) ]
  | Lui k ->
      let* () = need (imm20 k) "lui immediate" in
      Ok [ Mir_type.i32 ]
  | Mv a ->
      let* () = need (gpr (ty a)) "mv operand" in
      Ok [ ty a ]
  | Sel (p, a, b) ->
      let* () = need (is (ty p) Mir_type.Pred) "select predicate" in
      let* () = need (gpr (ty a) && is (ty a) (ty b)) "select operands" in
      Ok [ ty a ]
  | Sext (w, a) ->
      let* () =
        need
          ((w = Mir_width.W8 || w = Mir_width.W16) && is (ty a) (Mir_type.Int w))
          "sign extension source"
      in
      Ok [ Mir_type.i32 ]
  | Store (m, base, k, x) ->
      let* () = need (ptr (ty base)) "store base" in
      let* () = need (imm12 k) "store offset" in
      let* () = need (is (ty x) (Mir_type.Int (Msz.width m))) "store value" in
      Ok []
  | Sw_pair (base, k, lo, hi) ->
      let* () = need (ptr (ty base)) "store base" in
      let* () = need (imm12 k && imm12 (Int64.add k 4L)) "store offset" in
      let* () = need (i32 (ty lo) && i32 (ty hi)) "store words" in
      Ok []
  | Trunc (w, a) ->
      let* () =
        need
          ((w = Mir_width.W8 || w = Mir_width.W16) && i32 (ty a))
          "truncation"
      in
      Ok [ Mir_type.Int w ]
  | Vec x -> Rv32_vop.typing x
  | Zext a ->
      let* () =
        need
          (match ty a with
          | Mir_type.Int (Mir_width.W8 | Mir_width.W16) | Mir_type.Pred -> true
          | _ -> false)
          "zero extension source"
      in
      Ok [ Mir_type.i32 ]

let test_typing = function
  | Bcmp (_, a, b) -> need (i32 (ty a) && i32 (ty b)) "branch compare operands"
  | Beqz a | Bnez a -> need (int_like (ty a)) "branch on a non-integer"

let pp_op pv fmt op =
  let vs = Fmt.(list ~sep:(any ", ") pv) in
  match op with
  | Add_wide (a, k) -> Fmt.pf fmt "add.wide %a, %Ld" pv a k
  | Addi (a, k) -> Fmt.pf fmt "addi %a, %Ld" pv a k
  | Alu (o, a, b) -> Fmt.pf fmt "%s %a" (Alu.name o) vs [ a; b ]
  | Alui (o, a, k) -> Fmt.pf fmt "%s %a, %Ld" (Alui.name o) pv a k
  | Call { callee; args; _ } ->
      Fmt.pf fmt "call %a(%a)" Mir_op.Callee.pp callee vs args
  | Fbin (o, a, b) -> Fmt.pf fmt "%s %a" (Fop.name o) vs [ a; b ]
  | Fcmp (o, a, b) -> Fmt.pf fmt "%s %a" (Fcmp.name o) vs [ a; b ]
  | Fcvt_s_w a -> Fmt.pf fmt "fcvt.s.w %a" pv a
  | Fcvt_s_wu a -> Fmt.pf fmt "fcvt.s.wu %a" pv a
  | Fcvt_w_s a -> Fmt.pf fmt "fcvt.w.s %a, rtz" pv a
  | Flw (base, k) -> Fmt.pf fmt "flw %Ld(%a)" k pv base
  | Fmadd (a, b, c) -> Fmt.pf fmt "fmadd.s %a" vs [ a; b; c ]
  | Fmv_w_x a -> Fmt.pf fmt "fmv.w.x %a" pv a
  | Fmv_x_w a -> Fmt.pf fmt "fmv.x.w %a" pv a
  | Fsel (p, a, b) -> Fmt.pf fmt "fsel %a" vs [ p; a; b ]
  | Fsgnj (o, a, b) -> Fmt.pf fmt "%s %a" (Sgnj.name o) vs [ a; b ]
  | Fsqrt a -> Fmt.pf fmt "fsqrt.s %a" pv a
  | Fsw (base, k, x) -> Fmt.pf fmt "fsw %a, %Ld(%a)" pv x k pv base
  | La view -> Fmt.pf fmt "la %a" Mir_id.View.pp view
  | Li (t, k) -> Fmt.pf fmt "li.%a %Ld" Mir_type.pp t k
  | Load (m, base, k) -> Fmt.pf fmt "%s %Ld(%a)" (Msz.load m) k pv base
  | Lui k -> Fmt.pf fmt "lui 0x%Lx" k
  | Mv a -> Fmt.pf fmt "mv %a" pv a
  | Sel (p, a, b) -> Fmt.pf fmt "sel %a" vs [ p; a; b ]
  | Sext (w, a) -> Fmt.pf fmt "sext.%s %a" (Mir_width.name w) pv a
  | Store (m, base, k, x) ->
      Fmt.pf fmt "%s %a, %Ld(%a)" (Msz.store m) pv x k pv base
  | Sw_pair (base, k, lo, hi) ->
      Fmt.pf fmt "sw.pair %a, %Ld(%a)" vs [ lo; hi ] k pv base
  | Trunc (w, a) -> Fmt.pf fmt "trunc.%s %a" (Mir_width.name w) pv a
  | Vec x -> Rv32_vop.pp pv fmt x
  | Zext a -> Fmt.pf fmt "zext %a" pv a

let pp_test pv fmt = function
  | Bcmp (c, a, b) -> Fmt.pf fmt "%s %a, %a" (Bcond.name c) pv a pv b
  | Beqz a -> Fmt.pf fmt "beqz %a" pv a
  | Bnez a -> Fmt.pf fmt "bnez %a" pv a

let name = "riscv32"

let source =
  {
    Mir_target.Source.document = "The RISC-V Instruction Set Manual, Volume I";
    revision = "20240411";
  }
