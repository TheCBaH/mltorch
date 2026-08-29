(* Generator for the aten_op_config.ml runtime artifact.

   Emits one config record per curated ATen op, indexed by its fully-qualified
   exported node target ("torch.ops.aten.add.Tensor"). The type definitions
   (param_type, default_, param, return_arity, t) are embedded verbatim in the
   generated file so it has no dependency on aten_schema at runtime. Ops with
   unsupported argument/return types are silently omitted; the bridge treats them
   as Skipped. *)

module A = Aten_func_ast

(* Same name/target logic as aten_decode_gen, for consistency across artifacts. *)
let ocaml_name (op : A.t) =
  match op.name.overload with
  | None | Some "" -> op.name.base
  | Some ov -> op.name.base ^ "_" ^ ov

let target (op : A.t) =
  let ov =
    match op.name.overload with None | Some "" -> "default" | Some s -> s
  in
  "torch.ops.aten." ^ op.name.base ^ "." ^ ov

(* Map an AST type to the runtime param_type string, or None for unsupported
   types (e.g. complex, QScheme, GraphModule). Mirrors aten_c_type.map_type but
   produces the config enum name instead of C fragments. *)
let param_type_of (ty : Aten_func_ast.Type.t) =
  match ty with
  | Base Bool -> Some "Bool"
  | Optional (Base Bool) -> Some "Bool_opt"
  | Optional (Base Device) -> Some "Device_opt"
  | Base Float -> Some "Float"
  | Optional (Base Float) -> Some "Float_opt"
  | Base Int | Base SymInt -> Some "Int"
  | List (Base Int, _) | List (Base SymInt, _) -> Some "Int_list"
  | Optional (List (Base Int, _)) | Optional (List (Base SymInt, _)) ->
      Some "Int_list_opt"
  | Optional (Base Int) | Optional (Base SymInt) -> Some "Int_opt"
  | Optional (Base Layout) -> Some "Layout_opt"
  | Optional (Base MemoryFormat) -> Some "Memory_format_opt"
  | Base Scalar -> Some "Scalar"
  | Optional (Base Scalar) -> Some "Scalar_opt"
  | Optional (Base ScalarType) -> Some "Scalar_type_opt"
  | Base Str -> Some "Str"
  | Base Tensor -> Some "Tensor"
  | List (Base Tensor, _) -> Some "Tensor_list"
  | Optional (Base Tensor) -> Some "Tensor_opt"
  | _ -> None

let litf s = if String.length s > 0 && s.[0] = '-' then "(" ^ s ^ ")" else s

(* Emit an OCaml expression for the default value. Float defaults are preserved
   as their original YAML string so no precision is lost in the generated source. *)
let default_expr (d : Aten_func_ast.Default.t) =
  match d with
  | Bool b -> Printf.sprintf "D_bool %b" b
  | Float s -> Printf.sprintf "D_float %s" (litf s)
  | Ident s -> Printf.sprintf "D_ident %S" s
  | Int n ->
      if n < 0 then Printf.sprintf "D_int (%d)" n
      else Printf.sprintf "D_int %d" n
  | IntList ns ->
      Printf.sprintf "D_ints [ %s ]"
        (String.concat "; " (List.map string_of_int ns))
  | None -> "D_none"
  | Str s -> Printf.sprintf "D_str %S" s

(* Emit one param record, or None if the argument type is unsupported. *)
let emit_param (a : Aten_func_ast.Argument.t) ~kwonly =
  param_type_of a.ty
  |> Option.map (fun ty_s ->
      let default_s =
        match a.default with
        | None -> "None"
        | Some d -> Printf.sprintf "Some (%s)" (default_expr d)
      in
      Printf.sprintf "{ name = %S; ty = %s; default = %s; kwonly = %b }" a.name
        ty_s default_s kwonly)

(* Deliberately a second, independent reading of the returns rather than a call
   to Aten_c_type.map_returns: this one names a RUNTIME schema fact (how many
   results a caller should expect), not a C calling convention. [Tensor_list_return]
   is the shape whose count is not knowable here at all -- see
   Aten_spec_run.outputs_for, which has to derive it per target. *)
let returns_arity (returns : Aten_func_ast.Return.t list) =
  let is_tensor (r : Aten_func_ast.Return.t) =
    r.ty = Aten_func_ast.Type.Base Aten_func_ast.Base.Tensor
  in
  match returns with
  | [ { ty = Aten_func_ast.Type.List (Base Tensor, _); _ } ] ->
      Some "Tensor_list_return"
  | _ when List.for_all is_tensor returns -> (
      match List.length returns with
      | 1 -> Some "Single"
      | 2 -> Some "Tuple2"
      | 3 -> Some "Tuple3"
      | _ -> None)
  | _ -> None

(* Emit one [let ocaml_name : t = { ... }] binding, or None if unsupported. *)
let config_record (op : A.t) =
  if op.arguments.out <> [] then None
  else
    Option.bind (returns_arity op.returns) (fun ret_str ->
        let args =
          List.map (fun a -> emit_param a ~kwonly:false) op.arguments.positional
          @ List.map
              (fun a -> emit_param a ~kwonly:true)
              op.arguments.kwarg_only
        in
        if List.exists Option.is_none args then None
        else
          let params_s = List.map Option.get args |> String.concat ";\n    " in
          let overload_s =
            match op.name.overload with
            | None | Some "" -> "None"
            | Some s -> Printf.sprintf "Some %S" s
          in
          Some
            (Printf.sprintf
               {|let %s : t = {
  target = %S;
  base = %S;
  overload = %s;
  returns = %s;
  params = [
    %s;
  ];
}|}
               (ocaml_name op) (target op) op.name.base overload_s ret_str
               params_s))

let banner =
  {|(* @generated by aten_config_gen (bin/aten_ops_gen.ml) from native_functions.yaml.
   DO NOT EDIT. Regenerate via the dune rule in lib/aten_config/dune. *)|}

let header =
  {|type param_type =
  | Bool | Bool_opt
  | Device_opt
  | Float | Float_opt
  | Int | Int_list | Int_list_opt | Int_opt
  | Layout_opt
  | Memory_format_opt
  | Scalar | Scalar_opt | Scalar_type_opt
  | Str
  | Tensor | Tensor_list | Tensor_opt

type default_ =
  | D_bool of bool
  | D_float of float
  | D_ident of string
  | D_int of int
  | D_ints of int list
  | D_none
  | D_str of string

type param = {
  name    : string;
  ty      : param_type;
  default : default_ option;
  kwonly  : bool;
}

(* [Tensor_list_return] is a Tensor[] return: ONE output whose length is a
   runtime property of the call, so unlike the others it does not name a count.
   Not [Tensor_list] -- that constructor is taken by [param_type] above. *)
type return_arity = Single | Tensor_list_return | Tuple2 | Tuple3

type t = {
  target   : string;   (* "torch.ops.aten.add.Tensor" *)
  base     : string;   (* "add" *)
  overload : string option;
  params   : param list;
  returns  : return_arity;
}

let pp_param_type ppf = function
  | Bool -> Format.pp_print_string ppf "Bool"
  | Bool_opt -> Format.pp_print_string ppf "Bool?"
  | Device_opt -> Format.pp_print_string ppf "Device?"
  | Float -> Format.pp_print_string ppf "Float"
  | Float_opt -> Format.pp_print_string ppf "Float?"
  | Int -> Format.pp_print_string ppf "Int"
  | Int_list -> Format.pp_print_string ppf "Int[]"
  | Int_list_opt -> Format.pp_print_string ppf "Int[]?"
  | Int_opt -> Format.pp_print_string ppf "Int?"
  | Layout_opt -> Format.pp_print_string ppf "Layout?"
  | Memory_format_opt -> Format.pp_print_string ppf "MemoryFormat?"
  | Scalar -> Format.pp_print_string ppf "Scalar"
  | Scalar_opt -> Format.pp_print_string ppf "Scalar?"
  | Scalar_type_opt -> Format.pp_print_string ppf "ScalarType?"
  | Str -> Format.pp_print_string ppf "Str"
  | Tensor -> Format.pp_print_string ppf "Tensor"
  | Tensor_list -> Format.pp_print_string ppf "Tensor[]"
  | Tensor_opt -> Format.pp_print_string ppf "Tensor?"

let pp_default_ ppf = function
  | D_bool b -> Format.pp_print_bool ppf b
  | D_float f -> Format.fprintf ppf "%g" f
  | D_ident s -> Format.pp_print_string ppf s
  | D_int n -> Format.pp_print_int ppf n
  | D_ints ns ->
      Format.fprintf ppf "[%s]" (String.concat "," (List.map string_of_int ns))
  | D_none -> Format.pp_print_string ppf "none"
  | D_str s -> Format.fprintf ppf "%S" s

let pp_param ppf p =
  pp_param_type ppf p.ty;
  Format.pp_print_char ppf ' ';
  Format.pp_print_string ppf p.name;
  (match p.default with
   | None -> ()
   | Some d ->
       Format.pp_print_char ppf '=';
       pp_default_ ppf d)

let pp ppf t =
  Format.fprintf ppf "@[<2>%s@ (%a)@ ->@ %s@]"
    t.target
    (Format.pp_print_list
       ~pp_sep:(fun ppf () -> Format.fprintf ppf ",@ ")
       pp_param)
    t.params
    (match t.returns with
     | Single -> "T"
     | Tensor_list_return -> "T[]"
     | Tuple2 -> "(T, T)"
     | Tuple3 -> "(T, T, T)")|}

let file (ops : A.t list) =
  let records = List.filter_map config_record ops in
  let all_entries =
    List.filter_map
      (fun op ->
        config_record op
        |> Option.map (fun _ ->
            Printf.sprintf "  (%S, %s)" (target op) (ocaml_name op)))
      ops
  in
  String.concat "\n\n"
    [
      banner;
      header;
      String.concat "\n\n" records;
      Printf.sprintf
        {|let all : (string * t) list = [
%s
]

let find target = List.assoc_opt target all|}
        (String.concat ";\n" all_entries);
    ]
