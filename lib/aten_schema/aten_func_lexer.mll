{
open Aten_func_parser

let kw_or_ident s = match s with
  | "bool"         -> BOOL_TY
  | "Device"       -> DEVICE
  | "DeviceIndex"  -> DEVICE_INDEX
  | "Dimname"      -> DIMNAME
  | "DimVector"    -> DIM_VECTOR
  | "False"        -> FALSE
  | "float"        -> FLOAT_TY
  | "Generator"    -> GENERATOR
  | "GraphModule"  -> GRAPH_MODULE
  | "int"          -> INT_TY
  | "Layout"       -> LAYOUT
  | "MemoryFormat" -> MEMORY_FORMAT
  | "None"         -> NONE
  | "QScheme"      -> QSCHEME
  | "Scalar"       -> SCALAR
  | "ScalarType"   -> SCALAR_TYPE
  | "Storage"      -> STORAGE
  | "str"          -> STR_TY
  | "Stream"       -> STREAM
  | "SymBool"      -> SYM_BOOL
  | "SymInt"       -> SYM_INT
  | "Tensor"       -> TENSOR
  | "True"         -> TRUE
  | s              -> IDENT s
}

let digit = ['0'-'9']
let alpha = ['a'-'z' 'A'-'Z' '_']
let alnum = ['a'-'z' 'A'-'Z' '0'-'9' '_']
let ident = alpha alnum*

rule token = parse
  | [' ' '\t' '\n' '\r'] { token lexbuf }
  | '('    { LPAREN }
  | ')'    { RPAREN }
  | '['    { LBRACKET }
  | ']'    { RBRACKET }
  | ','    { COMMA }
  | '.'    { DOT }
  | '='    { EQ }
  | '!'    { BANG }
  | '*'    { STAR }
  | '?'    { QUESTION }
  | '|'    { PIPE }
  | "->"   { ARROW }
  | '-'? digit+ '.' digit* (['e' 'E'] '-'? digit+)? as s { FLOAT_LIT s }
  | '-'? digit+ ['e' 'E'] '-'? digit+ as s               { FLOAT_LIT s }
  | '-'? digit+ as s { INT_LIT (int_of_string s) }
  | '"'    { read_dqstring (Buffer.create 16) lexbuf }
  | '\''   { read_sqstring (Buffer.create 16) lexbuf }
  | ident as s { kw_or_ident s }
  | eof    { EOF }
  | _ as c { failwith (Printf.sprintf "unexpected char: %c" c) }

and read_dqstring buf = parse
  | '"'              { STR_LIT (Buffer.contents buf) }
  | '\\' ('"' as c)  { Buffer.add_char buf c; read_dqstring buf lexbuf }
  | '\\' ('\\' as c) { Buffer.add_char buf c; read_dqstring buf lexbuf }
  | '\\' (_ as c)    { Buffer.add_char buf '\\'; Buffer.add_char buf c;
                       read_dqstring buf lexbuf }
  | [^ '"' '\\'] as c { Buffer.add_char buf c; read_dqstring buf lexbuf }
  | eof              { failwith "unterminated string literal" }

and read_sqstring buf = parse
  | '\''             { STR_LIT (Buffer.contents buf) }
  | '\\' ('\'' as c) { Buffer.add_char buf c; read_sqstring buf lexbuf }
  | '\\' ('\\' as c) { Buffer.add_char buf c; read_sqstring buf lexbuf }
  | '\\' (_ as c)    { Buffer.add_char buf '\\'; Buffer.add_char buf c;
                       read_sqstring buf lexbuf }
  | [^ '\'' '\\'] as c { Buffer.add_char buf c; read_sqstring buf lexbuf }
  | eof              { failwith "unterminated string literal" }
