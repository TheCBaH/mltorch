open Js_ast

type parens = [ `All | `Minimal ]

(* ECMAScript operator precedence, the one table. Higher binds tighter. *)
let conditional = 3
let unary = 15
let call = 18
let member = 19
let primary = 20

let binop_prec = function
  | Or -> 4
  | And -> 5
  | Bit_or -> 6
  | Bit_and -> 8
  | Eq_strict | Ne_strict -> 9
  | Ge | Gt | Le | Lt -> 10
  | Shl | Shr -> 11
  | Add | Sub -> 12
  | Div | Mul -> 13

let binop_sym = function
  | Add -> "+"
  | And -> "&&"
  | Bit_and -> "&"
  | Bit_or -> "|"
  | Div -> "/"
  | Eq_strict -> "==="
  | Ge -> ">="
  | Gt -> ">"
  | Le -> "<="
  | Lt -> "<"
  | Mul -> "*"
  | Ne_strict -> "!=="
  | Or -> "||"
  | Shl -> "<<"
  | Shr -> ">>"
  | Sub -> "-"

let unop_sym = function Neg -> "-" | Not -> "!" | Plus -> "+"

(* A negative literal is [Unary Neg] of its magnitude, so it takes the unary
   precedence and the operators' spacing rule applies to it too. -0 and -Infinity
   are negative; NaN has no sign. The sign of zero is read off [1. /. x]:
   [Float.sign_bit] goes through [Int64.bits_of_float], which Melange gets wrong
   for -0. *)
let negative x = x < 0. || (x = 0. && 1. /. x < 0.)
let big_negative n = Int64.compare n 0L < 0

let magnitude x =
  if Float.is_nan x then "NaN"
  else if Float.abs x = Float.infinity then "Infinity"
  else if x = 0. then "0"
  else Printf.sprintf "%.17g" (Float.abs x)

(* The digits without going through negation, which [Int64.min_int] cannot take. *)
let big_magnitude n =
  let s = Int64.to_string n in
  if big_negative n then String.sub s 1 (String.length s - 1) else s

let prec = function
  | Binary (op, _, _) -> binop_prec op
  | Bigint n -> if big_negative n then unary else primary
  | Call _ | New _ -> call
  | Cond _ -> conditional
  | Index _ | Member _ -> member
  | Number x -> if negative x then unary else primary
  | Unary _ -> unary
  | Array _ | Bool _ | Global _ | Null | Object _ | String _ | Var _ -> primary

let is_literal = function Bigint _ | Number _ -> true | _ -> false

let escape s =
  let b = Buffer.create (String.length s + 2) in
  let n = String.length s in
  let i = ref 0 in
  while !i < n do
    (match s.[!i] with
    | '"' -> Buffer.add_string b "\\\""
    | '\\' -> Buffer.add_string b "\\\\"
    | '\n' -> Buffer.add_string b "\\n"
    | '\r' -> Buffer.add_string b "\\r"
    | '\t' -> Buffer.add_string b "\\t"
    | c when Char.code c < 0x20 ->
        Buffer.add_string b (Printf.sprintf "\\u%04x" (Char.code c))
    | '\xe2'
      when !i + 2 < n
           && s.[!i + 1] = '\x80'
           && (s.[!i + 2] = '\xa8' || s.[!i + 2] = '\xa9') ->
        (* U+2028 / U+2029 end a line in a JavaScript string literal. *)
        Buffer.add_string b
          (if s.[!i + 2] = '\xa8' then "\\u2028" else "\\u2029");
        i := !i + 2
    | c -> Buffer.add_char b c);
    incr i
  done;
  Buffer.contents b

let starts_with c s = s <> "" && s.[0] = c
let join = String.concat ", "

let rec go mode e =
  match e with
  | Array es -> "[" ^ join (List.map (child mode ~min:conditional) es) ^ "]"
  | Bigint n -> (if big_negative n then "-" else "") ^ big_magnitude n ^ "n"
  | Binary (op, a, b) ->
      let p = binop_prec op in
      child mode ~min:p a ^ " " ^ binop_sym op ^ " " ^ child mode ~min:(p + 1) b
  | Bool b -> string_of_bool b
  | Call (f, args) -> child mode ~min:call f ^ "(" ^ args_text mode args ^ ")"
  | Cond (c, a, b) ->
      child mode ~min:(conditional + 1) c
      ^ " ? "
      ^ child mode ~min:conditional a
      ^ " : "
      ^ child mode ~min:conditional b
  | Global g -> Global.name g
  | Index (a, i) -> object_ mode a ^ "[" ^ child mode ~min:0 i ^ "]"
  | Member (a, id) -> object_ mode a ^ "." ^ Ident.to_string id
  | New (f, args) ->
      "new " ^ child mode ~min:member f ^ "(" ^ args_text mode args ^ ")"
  | Null -> "null"
  | Number x -> (if negative x then "-" else "") ^ magnitude x
  | Object [] -> "{}"
  | Object fields ->
      "{ "
      ^ join
          (List.map
             (fun (k, v) ->
               Ident.to_string k ^ ": " ^ child mode ~min:conditional v)
             fields)
      ^ " }"
  | String s -> "\"" ^ escape s ^ "\""
  | Unary (op, a) ->
      let inner = child mode ~min:unary a in
      let sym = unop_sym op in
      (* [- -x], never [--x]: the two signs would read as a decrement. *)
      let space =
        (op = Neg && starts_with '-' inner)
        || (op = Plus && starts_with '+' inner)
      in
      sym ^ (if space then " " else "") ^ inner
  | Var id -> Ident.to_string id

and args_text mode args = join (List.map (child mode ~min:conditional) args)

(* The object of [.] and [[ ]]: a literal is parenthesised ([(1).x]), and
   nothing below the call level may stand there bare. *)
and object_ mode a =
  let s = child mode ~min:call a in
  if is_literal a && not (starts_with '(' s) then "(" ^ s ^ ")" else s

and child mode ~min e =
  let s = go mode e in
  let compound = prec e < call in
  if prec e < min || (mode = `All && compound) then "(" ^ s ^ ")" else s

let expr ?(parens = `Minimal) e = go parens e

let lvalue mode = function
  | Lvar id -> Ident.to_string id
  | Lindex (a, i) -> object_ mode a ^ "[" ^ child mode ~min:0 i ^ "]"

let assign_sym = function Eq -> "=" | Minus_eq -> "-=" | Plus_eq -> "+="

let rec stmt mode buf ~depth (s : Stmt.t) =
  let pad = String.make (2 * depth) ' ' in
  let line l =
    Buffer.add_string buf pad;
    Buffer.add_string buf l;
    Buffer.add_char buf '\n'
  in
  let block body = List.iter (stmt mode buf ~depth:(depth + 1)) body in
  match s with
  | Stmt.Assign (lv, op, e) ->
      line (lvalue mode lv ^ " " ^ assign_sym op ^ " " ^ go mode e ^ ";")
  | Stmt.Const (id, e) ->
      line ("const " ^ Ident.to_string id ^ " = " ^ go mode e ^ ";")
  | Stmt.Expr e ->
      let t = go mode e in
      (* An object literal at the start of a statement reads as a block. *)
      line ((if starts_with '{' t then "(" ^ t ^ ")" else t) ^ ";")
  | Stmt.For { var; init; test; body } ->
      let v = Ident.to_string var in
      line
        ("for (let " ^ v ^ " = " ^ go mode init ^ "; " ^ go mode test ^ "; " ^ v
       ^ "++) {");
      block body;
      line "}"
  | Stmt.Function f ->
      line
        ("function "
        ^ Ident.to_string f.Func.name
        ^ "("
        ^ join (List.map Ident.to_string f.Func.params)
        ^ ") {");
      block f.Func.body;
      line "}"
  | Stmt.If (c, yes, no) ->
      line ("if (" ^ go mode c ^ ") {");
      block yes;
      if no <> [] then (
        line "} else {";
        block no);
      line "}"
  | Stmt.Let (id, e) ->
      line ("let " ^ Ident.to_string id ^ " = " ^ go mode e ^ ";")
  | Stmt.Return None -> line "return;"
  | Stmt.Return (Some e) -> line ("return " ^ go mode e ^ ";")

let stmts ?(parens = `Minimal) l =
  let buf = Buffer.create 256 in
  List.iter (stmt parens buf ~depth:0) l;
  Buffer.contents buf

let script ?(parens = `Minimal) (p : Program.t) =
  "\"use strict\";\n"
  ^ stmts ~parens (p.Program.prelude @ [ Stmt.Function p.Program.entry ])

let factory_body ?parens (p : Program.t) =
  script ?parens p ^ "return "
  ^ Ident.to_string p.Program.entry.Func.name
  ^ ";\n"
