module Isa = struct
  type t = Aarch64 | X86_64

  let name = function Aarch64 -> "aarch64" | X86_64 -> "x86_64"
end

module Role = struct
  type t = Arithmetic | Conversion | Memory_move

  let all = [ Arithmetic; Conversion; Memory_move ]

  let name = function
    | Arithmetic -> "arithmetic"
    | Conversion -> "conversion"
    | Memory_move -> "memory_move"
end

module String_map = Map.Make (String)
module String_set = Set.Make (String)

type t = { roles : int array; mnemonics : int String_map.t }
type error = [ `Unknown_fp_mnemonic of Isa.t * string ]

let pp_error ppf : [< error ] -> unit = function
  | `Unknown_fp_mnemonic (isa, m) ->
      Format.fprintf ppf "%s: floating-point mnemonic %S is in no census table"
        (Isa.name isa) m

let role_index = function
  | Role.Arithmetic -> 0
  | Role.Conversion -> 1
  | Role.Memory_move -> 2

let set l = String_set.of_list l

(* {1 AArch64} *)

(* Tables are keyed by mnemonic and kept alphabetical. A mnemonic here is
   reached only when the instruction names a binary64 register or arrangement. *)
let a64_arithmetic =
  set
    [
      "fabs";
      "fabd";
      "fadd";
      "faddp";
      "fccmp";
      "fccmpe";
      "fcmeq";
      "fcmge";
      "fcmgt";
      "fcmle";
      "fcmlt";
      "fcmp";
      "fcmpe";
      "fdiv";
      "fmadd";
      "fmax";
      "fmaxnm";
      "fmaxnmp";
      "fmaxp";
      "fmin";
      "fminnm";
      "fminnmp";
      "fminp";
      "fmla";
      "fmls";
      "fmsub";
      "fmul";
      "fmulx";
      "fneg";
      "fnmadd";
      "fnmsub";
      "fnmul";
      "frecpe";
      "frecps";
      "frecpx";
      "frinta";
      "frinti";
      "frintm";
      "frintn";
      "frintp";
      "frintx";
      "frintz";
      "frsqrte";
      "frsqrts";
      "fsqrt";
      "fsub";
    ]

let a64_conversion =
  set
    [
      "fcvt";
      "fcvtas";
      "fcvtau";
      "fcvtl";
      "fcvtl2";
      "fcvtms";
      "fcvtmu";
      "fcvtn";
      "fcvtn2";
      "fcvtns";
      "fcvtnu";
      "fcvtps";
      "fcvtpu";
      "fcvtxn";
      "fcvtxn2";
      "fcvtzs";
      "fcvtzu";
      "fjcvtzs";
      "scvtf";
      "ucvtf";
    ]

let a64_move =
  set
    [
      "dup";
      "fcsel";
      "fmov";
      "ins";
      "ld1";
      "ldnp";
      "ldp";
      "ldr";
      "ldur";
      "mov";
      "movi";
      "smov";
      "st1";
      "stnp";
      "stp";
      "str";
      "stur";
      "umov";
    ]

let is_digit c = c >= '0' && c <= '9'

(* [d0]..[d31] as a whole operand token, or a [.2d]/[.1d]/[.d] arrangement. *)
let a64_token_is_double tok =
  let n = String.length tok in
  let reg =
    n >= 2
    && tok.[0] = 'd'
    &&
    let rest = String.sub tok 1 (n - 1) in
    String.for_all is_digit rest
    && match int_of_string_opt rest with Some k -> k <= 31 | None -> false
  in
  let ends s =
    n >= String.length s
    && String.sub tok (n - String.length s) (String.length s) = s
  in
  reg || ends ".2d" || ends ".1d" || ends ".d"

let operand_tokens ops =
  let b = Buffer.create 16 and acc = ref [] in
  let flush () =
    if Buffer.length b > 0 then (
      acc := Buffer.contents b :: !acc;
      Buffer.clear b)
  in
  String.iter
    (function
      | ',' | ' ' | '\t' | '[' | ']' | '!' | '{' | '}' | '(' | ')' | '#' ->
          flush ()
      | c -> Buffer.add_char b c)
    ops;
  flush ();
  List.rev !acc

(* {1 x86-64} *)

let x64_arithmetic =
  set
    [
      "addpd";
      "addsd";
      "andnpd";
      "andpd";
      "cmppd";
      "cmpsd";
      "comisd";
      "divpd";
      "divsd";
      "maxpd";
      "maxsd";
      "minpd";
      "minsd";
      "mulpd";
      "mulsd";
      "orpd";
      "roundpd";
      "roundsd";
      "sqrtpd";
      "sqrtsd";
      "subpd";
      "subsd";
      "ucomisd";
      "xorpd";
    ]

let x64_conversion =
  set
    [
      "cvtdq2pd";
      "cvtpd2dq";
      "cvtpd2ps";
      "cvtps2pd";
      "cvtsd2si";
      "cvtsd2sil";
      "cvtsd2siq";
      "cvtsd2ss";
      "cvtsi2sd";
      "cvtsi2sdl";
      "cvtsi2sdq";
      "cvtss2sd";
      "cvttpd2dq";
      "cvttsd2si";
      "cvttsd2sil";
      "cvttsd2siq";
    ]

let x64_move =
  set
    [
      "movapd";
      "movddup";
      "movhpd";
      "movlpd";
      "movq";
      "movsd";
      "movupd";
      "shufpd";
      "unpckhpd";
      "unpcklpd";
    ]

(* Known single-precision and integer SSE forms, so that a mnemonic on an
   [%xmm] register in no table at all is an error rather than assumed benign. *)
let x64_not_double =
  set
    [
      "addps";
      "addss";
      "andnps";
      "andps";
      "comiss";
      "cvtsi2ss";
      "cvtsi2ssl";
      "cvtsi2ssq";
      "cvtss2si";
      "cvtss2sil";
      "cvtss2siq";
      "cvttss2si";
      "cvttss2sil";
      "cvttss2siq";
      "divps";
      "divss";
      "maxps";
      "maxss";
      "minps";
      "minss";
      "movaps";
      "movd";
      "movss";
      "movups";
      "mulps";
      "mulss";
      "orps";
      "roundss";
      "sqrtps";
      "sqrtss";
      "subps";
      "subss";
      "ucomiss";
      "xorps";
    ]

let has_xmm ops =
  let n = String.length ops in
  let rec go i = i + 4 <= n && (String.sub ops i 4 = "%xmm" || go (i + 1)) in
  go 0

(* {1 Lines} *)

let strip_comment isa line =
  let cut s marker =
    let n = String.length s and k = String.length marker in
    let rec go i =
      if i + k > n then s
      else if String.sub s i k = marker then String.sub s 0 i
      else go (i + 1)
    in
    go 0
  in
  match isa with Isa.Aarch64 -> cut line "//" | Isa.X86_64 -> cut line "#"

(* The mnemonic and operand text of an instruction line, [None] for a label,
   a directive or a blank. *)
let split_instruction isa line =
  let line = String.trim (strip_comment isa line) in
  let is_label =
    line <> ""
    && line.[String.length line - 1] = ':'
    && not (String.contains line ' ')
  in
  if line = "" || line.[0] = '.' || is_label then None
  else
    match (String.index_opt line '\t', String.index_opt line ' ') with
    | None, None -> Some (line, "")
    | a, b ->
        let i =
          match (a, b) with
          | Some a, Some b -> min a b
          | Some a, None -> a
          | None, Some b -> b
          | None, None -> 0
        in
        Some
          ( String.sub line 0 i,
            String.trim (String.sub line i (String.length line - i)) )

let role_of_mnemonic ~arith ~conv ~move m =
  if String_set.mem m arith then Some Role.Arithmetic
  else if String_set.mem m conv then Some Role.Conversion
  else if String_set.mem m move then Some Role.Memory_move
  else None

(* [Ok None]: the instruction is not binary64. *)
let classify isa mnemonic ops =
  match isa with
  | Isa.Aarch64 -> (
      if not (List.exists a64_token_is_double (operand_tokens ops)) then Ok None
      else
        match
          role_of_mnemonic ~arith:a64_arithmetic ~conv:a64_conversion
            ~move:a64_move mnemonic
        with
        | Some r -> Ok (Some r)
        | None -> Error (`Unknown_fp_mnemonic (isa, mnemonic)))
  | Isa.X86_64 -> (
      match
        role_of_mnemonic ~arith:x64_arithmetic ~conv:x64_conversion
          ~move:x64_move mnemonic
      with
      | Some Role.Memory_move when mnemonic = "movq" ->
          (* a 64-bit integer move unless it names an [%xmm] register *)
          if has_xmm ops then Ok (Some Role.Memory_move) else Ok None
      | Some r -> Ok (Some r)
      | None ->
          if has_xmm ops && not (String_set.mem mnemonic x64_not_double) then
            Error (`Unknown_fp_mnemonic (isa, mnemonic))
          else Ok None)

let of_text isa text =
  let roles = Array.make (List.length Role.all) 0 in
  let step (mnemonics, ()) line =
    match split_instruction isa line with
    | None -> Ok (mnemonics, ())
    | Some (m, ops) -> (
        match classify isa m ops with
        | Error _ as e -> e
        | Ok None -> Ok (mnemonics, ())
        | Ok (Some r) ->
            let k = role_index r in
            roles.(k) <- roles.(k) + 1;
            let n = Option.value ~default:0 (String_map.find_opt m mnemonics) in
            Ok (String_map.add m (n + 1) mnemonics, ()))
  in
  let rec go acc = function
    | [] -> Ok acc
    | l :: rest -> (
        match step acc l with Ok acc -> go acc rest | Error e -> Error e)
  in
  match go (String_map.empty, ()) (String.split_on_char '\n' text) with
  | Ok (mnemonics, ()) -> Err.return { roles; mnemonics }
  | Error e -> Err.fail e

let count t r = t.roles.(role_index r)
let total t = Array.fold_left ( + ) 0 t.roles
let mnemonics t = String_map.bindings t.mnemonics

let pp ppf t =
  Format.fprintf ppf "arithmetic %d, conversion %d, memory_move %d"
    (count t Role.Arithmetic) (count t Role.Conversion)
    (count t Role.Memory_move)
