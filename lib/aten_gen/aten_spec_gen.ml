(* Generator for the aten_op_spec.ml artifact: a per-operation JSON codec for the
   verification format.

   Driven by the same curated op list (the [Aten_func_ast.t] entries from
   bin/aten_ops_gen) as aten_config_gen, so the JSON surface mirrors the runtime
   op config field-for-field and the two can't drift. For each op it emits a
   module with a record [type t] (one field per supported argument), a
   bidirectional [jsont] codec (defaults come across as [~dec_absent], exactly
   the YAML default aten_config_gen renders into [default_]), and [to_args] which
   flattens the record into the generic [Aten_spec.Arg_value.t] currency the
   runner lowers to a Pytorch_types.Node.

   A trailing registry maps each op target to its decoder, and a top-level
   [jsont] decodes the { target, seed, args } envelope (dispatching "args" to the
   matching per-op codec) and encodes it back generically via Arg_value.to_json. *)

module A = Aten_func_ast

let ocaml_name (op : A.t) =
  match op.name.overload with
  | None | Some "" -> op.name.base
  | Some ov -> op.name.base ^ "_" ^ ov

let target (op : A.t) =
  let ov =
    match op.name.overload with None | Some "" -> "default" | Some s -> s
  in
  "torch.ops.aten." ^ op.name.base ^ "." ^ ov

(* Module names must start uppercase and ATen bases may start with '_'
   (_softmax), so uniformly prefix "Op_". *)
let module_name (op : A.t) = "Op_" ^ ocaml_name op

let kind_string (op : A.t) =
  match op.name.overload with
  | None | Some "" -> op.name.base
  | Some ov -> op.name.base ^ "." ^ ov

(* The supported spec argument shapes, mirroring aten_config_gen's param_type.
   [T_skip] is an always-None optional enum (ScalarType?/Layout?/MemoryFormat?/
   Device?): dropped from the record entirely (the interpreter defaults it to
   None), so it never appears in JSON. *)
type sty =
  | T_tensor
  | T_tensor_opt
  | T_tensor_list
  | T_int
  | T_int_opt
  | T_int_list
  | T_int_list_opt
  | T_float
  | T_float_opt
  | T_bool
  | T_bool_opt
  | T_scalar
  | T_scalar_opt
  | T_str
  | T_skip

let classify (ty : A.Type.t) : sty option =
  match ty with
  | Base Tensor -> Some T_tensor
  | Optional (Base Tensor) -> Some T_tensor_opt
  | List (Base Tensor, _) -> Some T_tensor_list
  | Base Int | Base SymInt -> Some T_int
  | Optional (Base Int) | Optional (Base SymInt) -> Some T_int_opt
  | List (Base Int, _) | List (Base SymInt, _) -> Some T_int_list
  | Optional (List (Base Int, _)) | Optional (List (Base SymInt, _)) ->
      Some T_int_list_opt
  | Base Float -> Some T_float
  | Optional (Base Float) -> Some T_float_opt
  | Base Bool -> Some T_bool
  | Optional (Base Bool) -> Some T_bool_opt
  | Base Scalar -> Some T_scalar
  | Optional (Base Scalar) -> Some T_scalar_opt
  | Base Str -> Some T_str
  | Optional (Base ScalarType)
  | Optional (Base Layout)
  | Optional (Base MemoryFormat)
  | Optional (Base Device) ->
      Some T_skip
  | _ -> None

(* OCaml record-field type for a kept argument. *)
let field_ty = function
  | T_tensor -> "Aten_spec.Tensor_spec.t"
  | T_tensor_opt -> "Aten_spec.Tensor_spec.t option"
  | T_tensor_list -> "Aten_spec.Tensor_spec.t list"
  | T_int -> "int"
  | T_int_opt -> "int option"
  | T_int_list -> "int list"
  | T_int_list_opt -> "int list option"
  | T_float -> "Aten_spec.Float32.t"
  | T_float_opt -> "Aten_spec.Float32.t option"
  | T_bool -> "bool"
  | T_bool_opt -> "bool option"
  | T_scalar -> "Aten_spec.Scalar_value.t"
  | T_scalar_opt -> "Aten_spec.Scalar_value.t option"
  | T_str -> "string"
  | T_skip -> assert false

(* The element codec (for opt fields, opt_mem wraps the option itself). *)
let field_codec = function
  | T_tensor | T_tensor_opt -> "Aten_spec.Tensor_spec.jsont"
  | T_tensor_list -> "(Jsont.list Aten_spec.Tensor_spec.jsont)"
  | T_int | T_int_opt -> "Jsont.int"
  | T_int_list | T_int_list_opt -> "(Jsont.list Jsont.int)"
  | T_float | T_float_opt -> "Aten_spec.Float32.jsont"
  | T_bool | T_bool_opt -> "Jsont.bool"
  | T_scalar | T_scalar_opt -> "Aten_spec.Scalar_value.jsont"
  | T_str -> "Jsont.string"
  | T_skip -> assert false

let arg_ctor = function
  | T_tensor -> "Tensor"
  | T_tensor_opt -> "Tensor_opt"
  | T_tensor_list -> "Tensor_list"
  | T_int -> "Int"
  | T_int_opt -> "Int_opt"
  | T_int_list -> "Int_list"
  | T_int_list_opt -> "Int_list_opt"
  | T_float -> "Float"
  | T_float_opt -> "Float_opt"
  | T_bool -> "Bool"
  | T_bool_opt -> "Bool_opt"
  | T_scalar -> "Scalar"
  | T_scalar_opt -> "Scalar_opt"
  | T_str -> "Str"
  | T_skip -> assert false

let is_opt = function
  | T_tensor_opt | T_int_opt | T_int_list_opt | T_float_opt | T_bool_opt
  | T_scalar_opt ->
      true
  | _ -> false

(* Wrap a negative numeric literal so it parses as an expression. *)
let lit s = if String.length s > 0 && s.[0] = '-' then "(" ^ s ^ ")" else s

(* An OCaml [~dec_absent] expression for a param default, or None if the default
   can't be rendered for this field type (then the field is left required). *)
let dec_absent (sty : sty) (d : A.Default.t) : string option =
  match (sty, d) with
  | T_int, Int n -> Some (lit (string_of_int n))
  | T_float, Float s ->
      Some (Printf.sprintf "(Aten_spec.Float32.to_f32 %s)" (lit s))
  | T_bool, Bool b -> Some (Printf.sprintf "%b" b)
  | T_str, Str s -> Some (Printf.sprintf "%S" s)
  | T_int_list, IntList ns ->
      Some
        (Printf.sprintf "[ %s ]"
           (String.concat "; " (List.map string_of_int ns)))
  | T_scalar, Int n ->
      Some
        (Printf.sprintf "(Aten_spec.Scalar_value.Int %s)"
           (lit (string_of_int n)))
  | T_scalar, Float s ->
      Some
        (Printf.sprintf
           "(Aten_spec.Scalar_value.Float (Aten_spec.Float32.to_f32 %s))"
           (lit s))
  | T_scalar, Bool b ->
      Some (Printf.sprintf "(Aten_spec.Scalar_value.Bool %b)" b)
  | _ -> None

let keywords =
  [
    "and";
    "as";
    "assert";
    "begin";
    "class";
    "constraint";
    "do";
    "done";
    "downto";
    "else";
    "end";
    "exception";
    "external";
    "false";
    "for";
    "fun";
    "function";
    "functor";
    "if";
    "in";
    "include";
    "inherit";
    "initializer";
    "lazy";
    "let";
    "match";
    "method";
    "module";
    "mutable";
    "new";
    "nonrec";
    "object";
    "of";
    "open";
    "or";
    "private";
    "rec";
    "sig";
    "struct";
    "then";
    "to";
    "true";
    "try";
    "type";
    "val";
    "virtual";
    "when";
    "while";
    "with";
    "mod";
    "land";
    "lor";
    "lxor";
    "lsl";
    "lsr";
    "asr";
  ]

(* OCaml identifier for an arg name (the JSON key keeps the real name). *)
let ml_id name = if List.mem name keywords then name ^ "_" else name

type kept = {
  name : string;
  id : string;
  sty : sty;
  default : A.Default.t option;
}

(* Collect the kept (non-skip) args of an op in declared order, or None if any
   argument has an unsupported type (then the whole op is omitted, as in
   aten_config_gen). *)
let kept_args (op : A.t) : kept list option =
  if op.arguments.out <> [] then None
  else
    let all = op.arguments.positional @ op.arguments.kwarg_only in
    let rec go acc = function
      | [] -> Some (List.rev acc)
      | (a : A.Argument.t) :: rest -> (
          match classify a.ty with
          | None -> None
          | Some T_skip -> go acc rest
          | Some sty ->
              go
                ({ name = a.name; id = ml_id a.name; sty; default = a.default }
                :: acc)
                rest)
    in
    go [] all

let buf_add = Buffer.add_string

(* mem / opt_mem line for one field. *)
let emit_mem buf (k : kept) =
  let getter = Printf.sprintf "~enc:(fun (r : t) -> r.%s)" k.id in
  if is_opt k.sty then
    Printf.bprintf buf "  |> Jsont.Object.opt_mem %S %s %s\n" k.name getter
      (field_codec k.sty)
  else
    let dec_absent_s =
      match k.default with
      | Some d -> (
          match dec_absent k.sty d with
          | Some e -> Printf.sprintf "~dec_absent:%s " e
          | None -> "")
      | None -> ""
    in
    Printf.bprintf buf "  |> Jsont.Object.mem %S %s%s %s\n" k.name dec_absent_s
      getter (field_codec k.sty)

let emit_module buf (op : A.t) (kept : kept list) =
  Printf.bprintf buf "module %s = struct\n" (module_name op);
  Printf.bprintf buf "  let target = %S\n\n" (target op);
  (match kept with
  | [] ->
      (* Nullary op: the record is unit. *)
      buf_add buf "  type t = unit\n\n";
      Printf.bprintf buf "  let jsont : t Jsont.t =\n";
      Printf.bprintf buf
        "    Jsont.Object.map ~kind:%S () |> Jsont.Object.finish\n\n"
        (kind_string op);
      buf_add buf
        "  let to_args (_ : t) : (string * Aten_spec.Arg_value.t) list = []\n"
  | _ ->
      buf_add buf "  type t = {\n";
      List.iter
        (fun k -> Printf.bprintf buf "    %s : %s;\n" k.id (field_ty k.sty))
        kept;
      buf_add buf "  }\n\n";
      (* constructor *)
      let params = String.concat " " (List.map (fun k -> k.id) kept) in
      let fields = String.concat "; " (List.map (fun k -> k.id) kept) in
      Printf.bprintf buf "  let jsont : t Jsont.t =\n";
      Printf.bprintf buf
        "    Jsont.Object.map ~kind:%S (fun %s -> ({ %s } : t))\n"
        (kind_string op) params fields;
      List.iter (emit_mem buf) kept;
      buf_add buf "    |> Jsont.Object.finish\n\n";
      (* to_args *)
      buf_add buf
        "  let to_args (r : t) : (string * Aten_spec.Arg_value.t) list =\n";
      buf_add buf "    [\n";
      List.iter
        (fun k ->
          Printf.bprintf buf "      (%S, Aten_spec.Arg_value.%s r.%s);\n" k.name
            (arg_ctor k.sty) k.id)
        kept;
      buf_add buf "    ]\n");
  buf_add buf "end\n\n"

let banner =
  {|(* @generated by aten_spec_gen (bin/aten_ops_gen.ml) from native_functions.yaml.
   DO NOT EDIT. Regenerate via the dune rule in lib/aten_op_spec/dune. *)|}

let trailer ops =
  let buf = Buffer.create 1024 in
  buf_add buf "let args_decoder target :\n";
  buf_add buf
    "    (Jsont.json -> ((string * Aten_spec.Arg_value.t) list, string) \
     result) option =\n";
  buf_add buf "  match target with\n";
  List.iter
    (fun op ->
      Printf.bprintf buf
        "  | %S ->\n\
        \      Some (fun j -> Result.map %s.to_args (Jsont.Json.decode \
         %s.jsont j))\n"
        (target op) (module_name op) (module_name op))
    ops;
  buf_add buf "  | _ -> None\n\n";
  buf_add buf
    {|let jsont : Aten_spec.Op_spec.t Jsont.t =
  Jsont.map ~kind:"Op_spec"
    ~dec:(fun json ->
      match json with
      | Jsont.Object (ms, _) ->
          let find k =
            List.find_map
              (fun ((key, _), v) -> if String.equal key k then Some v else None)
              ms
          in
          let target =
            match find "target" with
            | Some v -> (
                match Jsont.Json.decode Jsont.string v with
                | Ok s -> s
                | Error e -> Jsont.Error.msg Jsont.Meta.none e)
            | None -> Jsont.Error.msgf Jsont.Meta.none "op spec: missing \"target\""
          in
          let args_json =
            match find "args" with Some v -> v | None -> Aten_spec.Spec_util.jobj []
          in
          let args =
            match args_decoder target with
            | None ->
                Jsont.Error.msgf Jsont.Meta.none "op spec: unsupported target %S"
                  target
            | Some f -> (
                match f args_json with
                | Ok a -> a
                | Error e -> Jsont.Error.msg Jsont.Meta.none e)
          in
          ({ target; args } : Aten_spec.Op_spec.t)
      | _ -> Jsont.Error.msgf Jsont.Meta.none "op spec: expected an object")
    ~enc:(fun (s : Aten_spec.Op_spec.t) ->
      let args =
        Aten_spec.Spec_util.jobj
          (List.map (fun (k, v) -> (k, Aten_spec.Arg_value.to_json v)) s.args)
      in
      Aten_spec.Spec_util.jobj
        [
          ("target", Aten_spec.Spec_util.jstr s.target); ("args", args);
        ])
    Jsont.json
|};
  Buffer.contents buf

let file (ops : A.t list) =
  let buf = Buffer.create 8192 in
  buf_add buf banner;
  buf_add buf "\n\n";
  let emitted =
    List.filter
      (fun op ->
        match kept_args op with
        | None -> false
        | Some kept ->
            emit_module buf op kept;
            true)
      ops
  in
  buf_add buf (trailer emitted);
  Buffer.contents buf
