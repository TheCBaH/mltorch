{
open Func_parser

let kw_or_ident s = match s with
  | "Generator"    -> GENERATOR
  | "ScalarType"   -> SCALAR_TYPE
  | "Tensor"       -> TENSOR
  | "int"          -> INT_TY
  | "Dimname"      -> DIMNAME
  | "DimVector"    -> DIM_VECTOR
  | "float"        -> FLOAT_TY
  | "str"          -> STR_TY
  | "bool"         -> BOOL_TY
  | "Layout"       -> LAYOUT
  | "Device"       -> DEVICE
  | "DeviceIndex"  -> DEVICE_INDEX
  | "Scalar"       -> SCALAR
  | "MemoryFormat" -> MEMORY_FORMAT
  | "QScheme"      -> QSCHEME
  | "Storage"      -> STORAGE
  | "Stream"       -> STREAM
  | "SymInt"       -> SYM_INT
  | "SymBool"      -> SYM_BOOL
  | "GraphModule"  -> GRAPH_MODULE
  | "None"         -> NONE
  | "True"         -> TRUE
  | "False"        -> FALSE
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
