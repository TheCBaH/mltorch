%{
open Func_ast

let is_out_arg (a : Argument.t) =
  match a.annotation with Some ann -> ann.is_write | None -> false
%}

%token GENERATOR SCALAR_TYPE TENSOR INT_TY DIMNAME DIM_VECTOR FLOAT_TY STR_TY BOOL_TY
%token LAYOUT DEVICE DEVICE_INDEX SCALAR MEMORY_FORMAT QSCHEME STORAGE STREAM
%token SYM_INT SYM_BOOL GRAPH_MODULE
%token NONE TRUE FALSE
%token <string> IDENT
%token <int>    INT_LIT
%token <string> FLOAT_LIT
%token <string> STR_LIT
%token LPAREN RPAREN LBRACKET RBRACKET COMMA DOT EQ BANG STAR QUESTION PIPE ARROW
%token EOF

%start <Func_ast.t> func_schema
%%

(* ---- top level ---- *)

func_schema:
  | op=op_name LPAREN args=arg_list RPAREN ARROW ret=returns EOF
    { let positional, kwarg_only, out = args in
      { name = op;
        arguments = Arguments.{ positional; kwarg_only; out };
        returns = ret } }
;

(* ---- op name: base[.overload] ---- *)

op_name:
  | base=name_part DOT overload=name_part { Op_name.{ base; overload = Some overload } }
  | base=name_part                        { Op_name.{ base; overload = None } }
;

(* op names can include any identifier or type keyword *)
name_part:
  | s=IDENT         { s }
  | s=type_kw_str   { s }
;

type_kw_str:
  | GENERATOR     { "Generator" }
  | SCALAR_TYPE   { "ScalarType" }
  | TENSOR        { "Tensor" }
  | INT_TY        { "int" }
  | DIMNAME       { "Dimname" }
  | DIM_VECTOR    { "DimVector" }
  | FLOAT_TY      { "float" }
  | STR_TY        { "str" }
  | BOOL_TY       { "bool" }
  | LAYOUT        { "Layout" }
  | DEVICE        { "Device" }
  | DEVICE_INDEX  { "DeviceIndex" }
  | SCALAR        { "Scalar" }
  | MEMORY_FORMAT { "MemoryFormat" }
  | QSCHEME       { "QScheme" }
  | STORAGE       { "Storage" }
  | STREAM        { "Stream" }
  | SYM_INT       { "SymInt" }
  | SYM_BOOL      { "SymBool" }
  | GRAPH_MODULE  { "GraphModule" }
  | NONE          { "None" }
  | TRUE          { "True" }
  | FALSE         { "False" }
;

(* ---- argument list: returns (positional, kwarg_only, out) ---- *)

arg_list:
  | (* empty *)              { [], [], [] }
  | args=nonempty_arg_list   { args }
;

nonempty_arg_list:
  | a=argument COMMA rest=nonempty_arg_list
    { let (ps, ks, os) = rest in (a :: ps, ks, os) }
  | a=argument
    { [a], [], [] }
  | STAR COMMA rest=kwarg_args
    { [], fst rest, snd rest }
  | STAR
    { [], [], [] }
;

(* After *, arguments are classified as kwarg_only or out (write-annotated Tensor). *)
kwarg_args:
  | a=argument COMMA rest=kwarg_args
    { let (ks, os) = rest in
      if is_out_arg a then (ks, a :: os) else (a :: ks, os) }
  | a=argument
    { if is_out_arg a then ([], [a]) else ([a], []) }
;

(* ---- single argument ---- *)

argument:
  | TENSOR LPAREN ann=annotation RPAREN name=arg_name dflt=opt_default
    { Argument.{ name; ty = Type.Base Base.Tensor;
                 annotation = Some ann; default = dflt } }
  | TENSOR LPAREN ann=annotation RPAREN QUESTION name=arg_name dflt=opt_default
    { Argument.{ name; ty = Type.Optional (Type.Base Base.Tensor);
                 annotation = Some ann; default = dflt } }
  | TENSOR LPAREN ann=annotation RPAREN LBRACKET RBRACKET name=arg_name dflt=opt_default
    { Argument.{ name; ty = Type.List (Type.Base Base.Tensor, None);
                 annotation = Some ann; default = dflt } }
  | ty=reg_ty name=arg_name dflt=opt_default
    { Argument.{ name; ty; annotation = None; default = dflt } }
;

(* ---- regular (un-annotated) types ---- *)

reg_ty:
  | b=base_ty                              { Type.Base b }
  | t=reg_ty QUESTION                      { Type.Optional t }
  | t=reg_ty LBRACKET RBRACKET             { Type.List (t, None) }
  | t=reg_ty LBRACKET n=INT_LIT RBRACKET   { Type.List (t, Some n) }
;

base_ty:
  | GENERATOR     { Base.Generator }
  | SCALAR_TYPE   { Base.ScalarType }
  | TENSOR        { Base.Tensor }
  | INT_TY        { Base.Int }
  | DIMNAME       { Base.Dimname }
  | DIM_VECTOR    { Base.DimVector }
  | FLOAT_TY      { Base.Float }
  | STR_TY        { Base.Str }
  | BOOL_TY       { Base.Bool }
  | LAYOUT        { Base.Layout }
  | DEVICE        { Base.Device }
  | DEVICE_INDEX  { Base.DeviceIndex }
  | SCALAR        { Base.Scalar }
  | MEMORY_FORMAT { Base.MemoryFormat }
  | QSCHEME       { Base.QScheme }
  | STORAGE       { Base.Storage }
  | STREAM        { Base.Stream }
  | SYM_INT       { Base.SymInt }
  | SYM_BOOL      { Base.SymBool }
  | GRAPH_MODULE  { Base.GraphModule }
;

(* ---- annotation: letter(s) + optional ! + optional -> alias_after ---- *)

annotation:
  | alias=alias_set wr=write_flag after=alias_after
    { Annotation.{ alias_set = alias; is_write = wr; alias_set_after = after } }
;

alias_set:
  | c=IDENT rest=alias_rest { c :: rest }
;

alias_rest:
  | (* empty *)                  { [] }
  | PIPE c=IDENT rest=alias_rest { c :: rest }
;

write_flag:
  | (* empty *) { false }
  | BANG        { true }
;

alias_after:
  | (* empty *)           { [] }
  | ARROW STAR            { ["*"] }
  | ARROW after=alias_set { after }
;

(* ---- argument name (any ident or keyword) ---- *)

arg_name:
  | s=IDENT         { s }
  | s=type_kw_str   { s }
;

opt_default:
  | (* empty *)    { None }
  | EQ v=default_val { Some v }
;

default_val:
  | NONE                              { Default.None }
  | TRUE                              { Default.Bool true }
  | FALSE                             { Default.Bool false }
  | n=INT_LIT                         { Default.Int n }
  | s=FLOAT_LIT                       { Default.Float s }
  | s=STR_LIT                         { Default.Str s }
  | LBRACKET RBRACKET                 { Default.IntList [] }
  | LBRACKET ns=int_list RBRACKET     { Default.IntList ns }
  | s=IDENT                           { Default.Ident s }
;

int_list:
  | n=INT_LIT                         { [n] }
  | n=INT_LIT COMMA rest=int_list     { n :: rest }
;

(* ---- returns ---- *)

returns:
  | LPAREN rs=ret_list RPAREN { rs }
  | r=return_val              { [r] }
;

ret_list:
  | (* empty *)                       { [] }
  | r=return_val                      { [r] }
  | r=return_val COMMA rest=ret_list  { r :: rest }
;

return_val:
  | TENSOR LPAREN ann=annotation RPAREN rname=opt_ret_name
    { Return.{ name = rname; ty = Type.Base Base.Tensor; annotation = Some ann } }
  | TENSOR LPAREN ann=annotation RPAREN QUESTION rname=opt_ret_name
    { Return.{ name = rname; ty = Type.Optional (Type.Base Base.Tensor);
               annotation = Some ann } }
  | TENSOR LPAREN ann=annotation RPAREN LBRACKET RBRACKET rname=opt_ret_name
    { Return.{ name = rname; ty = Type.List (Type.Base Base.Tensor, None);
               annotation = Some ann } }
  | ty=reg_ty rname=opt_ret_name
    { Return.{ name = rname; ty; annotation = None } }
;

opt_ret_name:
  | (* empty *)    { None }
  | s=IDENT        { Some s }
  | s=type_kw_str  { Some s }
;
