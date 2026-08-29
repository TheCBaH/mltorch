(* Code generator: Type_def.t String_map.t -> OCaml source with jsont decoders *)

module String_map = Pytorch_schema.String_map
module String_set = Set.Make (String)

module Scc_string = Scc.Make (struct
  type t = string

  let compare = String.compare
end)

open Pytorch_schema

let rec refs_of_type = function
  | Type_expr.Str | Type_expr.Int | Type_expr.Float | Type_expr.Bool ->
      String_set.empty
  | Type_expr.List t | Type_expr.Optional t | Type_expr.Dict t -> refs_of_type t
  | Type_expr.Ref name -> String_set.singleton name

let type_module_name name = name ^ "_Type"

let camel_to_snake s =
  let buf = Buffer.create (String.length s + 4) in
  String.iteri
    (fun i c ->
      if i > 0 && c >= 'A' && c <= 'Z' then Buffer.add_char buf '_';
      Buffer.add_char buf (Char.lowercase_ascii c))
    s;
  Buffer.contents buf

let jsont_lazy_name name = camel_to_snake name ^ "_jsont"

(* [group] is the set of names in the current recursive SCC; refs to those
   names use the _Type module qualifier / Lazy.force instead of the bare name. *)
let rec pp_ocaml_type_impl group fmt = function
  | Type_expr.Str -> Format.pp_print_string fmt "string"
  | Type_expr.Int -> Format.pp_print_string fmt "int"
  | Type_expr.Float -> Format.pp_print_string fmt "float"
  | Type_expr.Bool -> Format.pp_print_string fmt "bool"
  | Type_expr.List t ->
      Format.fprintf fmt "%a list" (pp_ocaml_type_impl group) t
  | Type_expr.Optional t ->
      Format.fprintf fmt "%a option" (pp_ocaml_type_impl group) t
  | Type_expr.Dict t ->
      Format.fprintf fmt "%a String_map.t" (pp_ocaml_type_impl group) t
  | Type_expr.Ref name ->
      if String_set.mem name group then
        Format.fprintf fmt "%s.t" (type_module_name name)
      else Format.fprintf fmt "%s.t" name

let pp_ocaml_type = pp_ocaml_type_impl String_set.empty
let pp_ocaml_type_rec = pp_ocaml_type_impl

let rec pp_jsont_expr_impl group fmt = function
  | Type_expr.Str -> Format.pp_print_string fmt "Jsont.string"
  | Type_expr.Int -> Format.pp_print_string fmt "Jsont.int"
  | Type_expr.Float -> Format.pp_print_string fmt "float_jsont"
  | Type_expr.Bool -> Format.pp_print_string fmt "Jsont.bool"
  | Type_expr.List t ->
      Format.fprintf fmt "(Jsont.list %a)" (pp_jsont_expr_impl group) t
  | Type_expr.Optional t -> pp_jsont_expr_impl group fmt t
  | Type_expr.Dict t ->
      Format.fprintf fmt "(Jsont.Object.as_string_map %a)"
        (pp_jsont_expr_impl group) t
  | Type_expr.Ref name ->
      if String_set.mem name group then
        Format.fprintf fmt "(Lazy.force %s)" (jsont_lazy_name name)
      else Format.fprintf fmt "%s.jsont" name

let pp_jsont_expr = pp_jsont_expr_impl String_set.empty
let pp_jsont_expr_rec = pp_jsont_expr_impl

(* Union arms only: an [int]-typed "as_*" case carries an unconstrained
   Python scalar (Argument.as_int, ConstantValue.as_int, SymInt.as_int,
   SymExprHint.as_int), where PyTorch's exporter can legitimately write
   sys.maxsize as an "unbounded" sentinel -- see [Schema_runtime.python_int_jsont].
   A struct's own [int] field (count/index/version) has no such sentinel and
   stays on plain [pp_jsont_expr]/[Jsont.int]. *)
let pp_union_jsont_expr_impl group fmt = function
  | Type_expr.Int -> Format.pp_print_string fmt "python_int_jsont"
  | t -> pp_jsont_expr_impl group fmt t

let pp_union_jsont_expr = pp_union_jsont_expr_impl String_set.empty
let pp_union_jsont_expr_rec = pp_union_jsont_expr_impl

(* ---- Naming ---- *)

let ocaml_keywords =
  [
    "and";
    "as";
    "assert";
    "asr";
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
    "land";
    "lazy";
    "let";
    "lor";
    "lsl";
    "lsr";
    "lxor";
    "match";
    "method";
    "mod";
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
  ]

let sanitize_field name =
  if List.mem name ocaml_keywords then name ^ "_" else name

let capitalize_first s =
  if s = "" then "V_"
  else
    String.make 1 (Char.uppercase_ascii s.[0])
    ^ String.sub s 1 (String.length s - 1)

let enum_ctor_of_name name =
  let i = ref 0 in
  while !i < String.length name && name.[!i] = '_' do
    incr i
  done;
  let rest = String.sub name !i (String.length name - !i) in
  capitalize_first rest

let union_ctor_of_name name =
  let stripped =
    if String.length name > 3 && String.sub name 0 3 = "as_" then
      String.sub name 3 (String.length name - 3)
    else name
  in
  capitalize_first stripped

(* ---- Default value rendering ---- *)

let default_ocaml_expr type_expr default_str =
  match (String.trim default_str, type_expr) with
  | ("None" | "none"), Type_expr.Optional _ -> None
  | ("[]" | "'[]'"), Type_expr.List _ -> Some "[]"
  | ("{}" | "'{}'"), Type_expr.Dict _ -> Some "String_map.empty"
  | ("False" | "'False'"), Type_expr.Bool -> Some "false"
  | ("True" | "'True'"), Type_expr.Bool -> Some "true"
  | s, Type_expr.Str -> (
      let l = String.length s in
      let unquote q =
        if l >= 2 && s.[0] = q && s.[l - 1] = q then
          Some (Printf.sprintf "\"%s\"" (String.sub s 1 (l - 2)))
        else None
      in
      match unquote '\'' with
      | Some e -> Some e
      | None -> (
          match unquote '"' with
          | Some e -> Some e
          | None -> Some (Printf.sprintf "\"%s\"" s)))
  | s, Type_expr.Int -> Some s
  | s, Type_expr.Float -> Some s
  | s, _ ->
      Some (Printf.sprintf "(failwith \"TODO default: %s\")" (String.escaped s))

(* ---- Dependency analysis ---- *)

let fields_refs fields =
  List.fold_left
    (fun acc f ->
      String_set.union acc
        (refs_of_type (Type_expr_parse.of_string (Field.type_ f))))
    String_set.empty fields

let refs_of_typedef = function
  | Type_def.Enum _ -> String_set.empty
  | Type_def.Struct s -> fields_refs (Struct.fields s)
  | Type_def.Union u -> fields_refs (Union.fields u)

let compute_sccs (types : Type_def.t String_map.t) : string list list =
  let nodes = List.map fst (String_map.bindings types) in
  let succ name =
    String_set.elements
      (String_set.filter
         (fun n -> String_map.mem n types)
         (refs_of_typedef (String_map.find name types)))
  in
  Scc_string.run ~succ nodes

(* ---- Field info types ---- *)

type struct_field_info = {
  sf_name : string;
  sf_ocaml : string;
  sf_type : Type_expr.t;
  sf_default : string option;
}

let struct_field_info_of (f : Field.t) =
  {
    sf_name = Field.name f;
    sf_ocaml = sanitize_field (Field.name f);
    sf_type = Type_expr_parse.of_string (Field.type_ f);
    sf_default = Field.default f;
  }

type mem_kind = Mem_default of string | Mem_optional | Mem_required

let classify_field (fi : struct_field_info) : mem_kind =
  match fi.sf_type with
  | Type_expr.Optional _ -> Mem_optional
  | te -> (
      match fi.sf_default with
      | None -> Mem_required
      | Some d -> (
          match default_ocaml_expr te d with
          | None -> Mem_optional
          | Some expr -> Mem_default expr))

type union_field_info = {
  uf_name : string;
  uf_ctor : string;
  uf_type : Type_expr.t;
}

let union_field_info_of (f : Field.t) =
  {
    uf_name = Field.name f;
    uf_ctor = union_ctor_of_name (Field.name f);
    uf_type = Type_expr_parse.of_string (Field.type_ f);
  }

(* ---- Format helpers ---- *)

let pp_sep_sp fmt () = Format.pp_print_char fmt ' '

(* Struct field type: handles Optional unwrapping for the OCaml record type *)
let pp_struct_field_type_impl group fmt fi =
  match fi.sf_type with
  | Type_expr.Optional inner ->
      Format.fprintf fmt "%a option" (pp_ocaml_type_impl group) inner
  | t -> pp_ocaml_type_impl group fmt t

let pp_struct_field_type = pp_struct_field_type_impl String_set.empty
let pp_struct_field_type_rec = pp_struct_field_type_impl

(* ---- Non-recursive module bodies ---- *)
(* Callers wrap each body in "@[<v 0>module %s = struct@;<0 2>@[<v 0>...@]@ end@]".
   Within that outer v0-at-col-2 box, @; breaks to col 2 and @;<0 2> goes to col 4. *)

let pp_enum_body fmt name fields =
  (* [fields] is schema-declared numeric order. The generated constructors and
     conversion arms must preserve it: reordering would change the enum's wire
     encoding, so this is deliberately not an alphabetical presentation. *)
  let ctors =
    List.map
      (fun f -> (enum_ctor_of_name (Enum_field.name f), Enum_field.value f))
      fields
  in
  Format.fprintf fmt
    "type t = %a@\n\
     @;\
     let jsont =@;\
     <0 2>@[<v 0>Jsont.map ~kind:%S@;\
     <0 2>@[<v 0>~dec:(fun n ->@;\
     <0 2>@[<v 0>match n with@;\
     %a@ | n -> Jsont.Error.msgf Jsont.Meta.none \"Unknown %s value: %%d\" \
     n)@]@ ~enc:(fun v ->@;\
     <0 2>@[<v 0>match v with@;\
     %a@])@ Jsont.int@]@]@\n\
     @;\
     let make v = v"
    (Format.pp_print_list
       ~pp_sep:(fun fmt () -> Format.pp_print_string fmt " | ")
       (fun fmt (ctor, _) -> Format.pp_print_string fmt ctor))
    ctors name
    (Format.pp_print_list
       ~pp_sep:(fun fmt () -> Format.fprintf fmt "@ ")
       (fun fmt (ctor, v) -> Format.fprintf fmt "| %d -> %s" v ctor))
    ctors name
    (Format.pp_print_list
       ~pp_sep:(fun fmt () -> Format.fprintf fmt "@ ")
       (fun fmt (ctor, v) -> Format.fprintf fmt "| %s -> %d" ctor v))
    ctors

let pp_struct_type_decl_impl group fmt fis =
  match fis with
  | [ fi ] ->
      Format.fprintf fmt "type t = { %s : %a }" fi.sf_ocaml
        (pp_struct_field_type_impl group)
        fi
  | _ ->
      Format.fprintf fmt "@[<v 0>type t = {@;<0 2>@[<v 0>%a@]@ }@]"
        (Format.pp_print_list
           ~pp_sep:(fun fmt () -> Format.fprintf fmt "@ ")
           (fun fmt fi ->
             Format.fprintf fmt "%s : %a;" fi.sf_ocaml
               (pp_struct_field_type_impl group)
               fi))
        fis

let pp_struct_type_decl = pp_struct_type_decl_impl String_set.empty
let pp_struct_type_decl_rec = pp_struct_type_decl_impl

let pp_struct_make fmt fis =
  Format.fprintf fmt "let make %a =@;<0 2>{ %a }"
    (Format.pp_print_list ~pp_sep:pp_sep_sp (fun fmt fi ->
         Format.pp_print_string fmt fi.sf_ocaml))
    fis
    (Format.pp_print_list
       ~pp_sep:(fun fmt () -> Format.pp_print_string fmt "; ")
       (fun fmt fi -> Format.pp_print_string fmt fi.sf_ocaml))
    fis

let pp_struct_jsont_constructor fmt kinds =
  let needs_lambda =
    List.exists
      (fun (_, k) ->
        match k with Mem_default _ | Mem_optional -> true | _ -> false)
      kinds
  in
  if needs_lambda then
    let lam_params =
      List.map
        (fun (fi, k) ->
          match k with
          | Mem_default _ -> fi.sf_ocaml ^ "_opt"
          | _ -> fi.sf_ocaml)
        kinds
    in
    Format.fprintf fmt "(fun %a ->@;<0 2>@[<v 0>{ %a }@])"
      (Format.pp_print_list ~pp_sep:pp_sep_sp Format.pp_print_string)
      lam_params
      (Format.pp_print_list
         ~pp_sep:(fun fmt () -> Format.fprintf fmt "@ ; ")
         (fun fmt ((fi, k), param) ->
           match k with
           | Mem_default e ->
               Format.fprintf fmt
                 "%s = (match %s with None -> %s | Some v -> v)" fi.sf_ocaml
                 param e
           | Mem_optional ->
               Format.fprintf fmt "%s = Option.join %s" fi.sf_ocaml param
           | Mem_required -> Format.pp_print_string fmt fi.sf_ocaml))
      (List.combine kinds lam_params)
  else Format.pp_print_string fmt "make"

let pp_struct_jsont fmt name fis =
  let kinds = List.map (fun fi -> (fi, classify_field fi)) fis in
  Format.fprintf fmt
    "let jsont =@;\
     <0 2>@[<v 0>Jsont.Object.map ~kind:%S %a%a@;\
     |> Jsont.Object.finish@]"
    name pp_struct_jsont_constructor kinds
    (Format.pp_print_list
       ~pp_sep:(fun _ () -> ())
       (fun fmt (fi, k) ->
         let te = match fi.sf_type with Type_expr.Optional t -> t | t -> t in
         match k with
         | Mem_default _ ->
             Format.fprintf fmt "@;|> Jsont.Object.opt_mem %S %a" fi.sf_name
               pp_jsont_expr te
         | Mem_optional ->
             Format.fprintf fmt "@;|> Jsont.Object.opt_mem %S (Jsont.option %a)"
               fi.sf_name pp_jsont_expr te
         | Mem_required ->
             Format.fprintf fmt "@;|> Jsont.Object.mem %S %a" fi.sf_name
               pp_jsont_expr te))
    kinds

let pp_struct_body fmt name fis =
  pp_struct_type_decl fmt fis;
  Format.fprintf fmt "@\n@;";
  pp_struct_make fmt fis;
  Format.fprintf fmt "@\n@;";
  pp_struct_jsont fmt name fis

let pp_union_case fmt uf =
  Format.fprintf fmt
    "| %S ->@;\
     <0 2>@[<v 0>(match Jsont.Json.decode %a value with@;\
     | Ok v -> %s v@;\
     | Error s -> Jsont.Error.msg Jsont.Meta.none s)@]"
    uf.uf_name pp_union_jsont_expr uf.uf_type uf.uf_ctor

let pp_union_body fmt name fis =
  Format.fprintf fmt
    "type t =@;\
     <0 2>@[<v 0>%a@]@\n\
     @;\
     let jsont =@;\
     <0 2>@[<v 0>Jsont.map ~kind:%S@;\
     <0 2>@[<v 0>~dec:(fun json ->@;\
     <0 2>@[<v 0>match json with@;\
     | Jsont.Object ([ ((key, _), value) ], _) ->@;\
     <0 2>@[<v 0>(match key with@;\
     %a@ | k -> Jsont.Error.msgf Jsont.Meta.none \"Unknown %s case: %%s\" \
     k)@]@ | Jsont.Object _ ->@;\
     <0 2>Jsont.Error.msgf Jsont.Meta.none \"%s must have exactly one \
     member\"@ | _ ->@;\
     <0 2>Jsont.Error.msgf Jsont.Meta.none \"%s must be a JSON object\")@]@ \
     ~enc:(fun _ -> assert false)@ Jsont.json@]@]@\n\
     @;\
     let make v = v"
    (Format.pp_print_list
       ~pp_sep:(fun fmt () -> Format.fprintf fmt "@ ")
       (fun fmt uf ->
         Format.fprintf fmt "| %s of %a" uf.uf_ctor pp_ocaml_type uf.uf_type))
    fis name
    (Format.pp_print_list
       ~pp_sep:(fun fmt () -> Format.fprintf fmt "@ ")
       pp_union_case)
    fis name name name

let pp_single fmt name typedef =
  Format.fprintf fmt "@[<v 0>module %s = struct@;<0 2>@[<v 0>%a@]@ end@]" name
    (fun fmt () ->
      match typedef with
      | Type_def.Enum e -> pp_enum_body fmt name (Enum.fields e)
      | Type_def.Struct s ->
          pp_struct_body fmt name
            (List.map struct_field_info_of (Struct.fields s))
      | Type_def.Union u ->
          pp_union_body fmt name (List.map union_field_info_of (Union.fields u)))
    ()

(* ---- Recursive group: _Type modules + lazy jsnts + facades ---- *)

let pp_type_module fmt group kw name typedef =
  let tmn = type_module_name name in
  Format.fprintf fmt
    "@[<v 0>%s %s : sig@;\
     <0 2>@[<v 0>%a@]@ end = struct@;\
     <0 2>@[<v 0>%a@]@ end@]"
    kw tmn
    (* sig body *)
    (fun fmt () ->
      match typedef with
      | Type_def.Enum e ->
          let ctors =
            List.map
              (fun f -> enum_ctor_of_name (Enum_field.name f))
              (Enum.fields e)
          in
          Format.fprintf fmt "type t = %a@;val make : t -> t"
            (Format.pp_print_list
               ~pp_sep:(fun fmt () -> Format.pp_print_string fmt " | ")
               Format.pp_print_string)
            ctors
      | Type_def.Struct s ->
          let fis = List.map struct_field_info_of (Struct.fields s) in
          pp_struct_type_decl_rec group fmt fis;
          Format.fprintf fmt "@;val make : %a -> t"
            (Format.pp_print_list
               ~pp_sep:(fun fmt () -> Format.pp_print_string fmt " -> ")
               (pp_struct_field_type_rec group))
            fis
      | Type_def.Union u ->
          let fis = List.map union_field_info_of (Union.fields u) in
          Format.fprintf fmt "type t =@;<0 2>@[<v 0>%a@]@;val make : t -> t"
            (Format.pp_print_list
               ~pp_sep:(fun fmt () -> Format.fprintf fmt "@ ")
               (fun fmt uf ->
                 Format.fprintf fmt "| %s of %a" uf.uf_ctor
                   (pp_ocaml_type_rec group) uf.uf_type))
            fis)
    ()
    (* struct body *)
    (fun fmt () ->
      match typedef with
      | Type_def.Enum e ->
          let ctors =
            List.map
              (fun f -> enum_ctor_of_name (Enum_field.name f))
              (Enum.fields e)
          in
          Format.fprintf fmt "type t = %a@\n@;let make v = v"
            (Format.pp_print_list
               ~pp_sep:(fun fmt () -> Format.pp_print_string fmt " | ")
               Format.pp_print_string)
            ctors
      | Type_def.Struct s ->
          let fis = List.map struct_field_info_of (Struct.fields s) in
          pp_struct_type_decl_rec group fmt fis;
          Format.fprintf fmt "@\n@;";
          pp_struct_make fmt fis
      | Type_def.Union u ->
          let fis = List.map union_field_info_of (Union.fields u) in
          Format.fprintf fmt "type t =@;<0 2>@[<v 0>%a@]@\n@;let make v = v"
            (Format.pp_print_list
               ~pp_sep:(fun fmt () -> Format.fprintf fmt "@ ")
               (fun fmt uf ->
                 Format.fprintf fmt "| %s of %a" uf.uf_ctor
                   (pp_ocaml_type_rec group) uf.uf_type))
            fis)
    ()

let pp_struct_jsont_constructor_rec fmt tmn kinds =
  let needs_lambda =
    List.exists
      (fun (_, k) ->
        match k with Mem_default _ | Mem_optional -> true | _ -> false)
      kinds
  in
  if needs_lambda then
    let lam_params =
      List.map
        (fun (fi, k) ->
          match k with
          | Mem_default _ -> fi.sf_ocaml ^ "_opt"
          | _ -> fi.sf_ocaml)
        kinds
    in
    Format.fprintf fmt "(fun %a ->@;<0 2>@[<v 0>({ %a } : %s.t)@])"
      (Format.pp_print_list ~pp_sep:pp_sep_sp Format.pp_print_string)
      lam_params
      (Format.pp_print_list
         ~pp_sep:(fun fmt () -> Format.fprintf fmt "@ ; ")
         (fun fmt ((fi, k), param) ->
           match k with
           | Mem_default e ->
               Format.fprintf fmt
                 "%s = (match %s with None -> %s | Some v -> v)" fi.sf_ocaml
                 param e
           | Mem_optional ->
               Format.fprintf fmt "%s = Option.join %s" fi.sf_ocaml param
           | Mem_required -> Format.pp_print_string fmt fi.sf_ocaml))
      (List.combine kinds lam_params)
      tmn
  else Format.fprintf fmt "%s.make" tmn

let pp_union_case_rec group tmn fmt uf =
  Format.fprintf fmt
    "| %S ->@;\
     <0 2>@[<v 0>(match Jsont.Json.decode %a value with@;\
     | Ok v -> %s.%s v@;\
     | Error s -> Jsont.Error.msg Jsont.Meta.none s)@]"
    uf.uf_name
    (pp_union_jsont_expr_rec group)
    uf.uf_type tmn uf.uf_ctor

let pp_jsont_decl fmt group name typedef kw =
  let tmn = type_module_name name in
  let jln = jsont_lazy_name name in
  Format.fprintf fmt "%s %s : %s.t Jsont.t Lazy.t = lazy (@;<0 2>@[<v 0>%a@])"
    kw jln tmn
    (fun fmt () ->
      match typedef with
      | Type_def.Enum e ->
          let ctors =
            List.map
              (fun f ->
                (enum_ctor_of_name (Enum_field.name f), Enum_field.value f))
              (Enum.fields e)
          in
          Format.fprintf fmt
            "Jsont.map ~kind:%S@;\
             <0 2>@[<v 0>~dec:(fun n ->@;\
             <0 2>@[<v 0>match n with@;\
             %a@ | n -> Jsont.Error.msgf Jsont.Meta.none \"Unknown %s value: \
             %%d\" n)@]@ ~enc:(fun v ->@;\
             <0 2>@[<v 0>match v with@;\
             %a@])@ Jsont.int@]"
            name
            (Format.pp_print_list
               ~pp_sep:(fun fmt () -> Format.fprintf fmt "@ ")
               (fun fmt (ctor, v) ->
                 Format.fprintf fmt "| %d -> %s.%s" v tmn ctor))
            ctors name
            (Format.pp_print_list
               ~pp_sep:(fun fmt () -> Format.fprintf fmt "@ ")
               (fun fmt (ctor, v) ->
                 Format.fprintf fmt "| %s.%s -> %d" tmn ctor v))
            ctors
      | Type_def.Struct s ->
          let fis = List.map struct_field_info_of (Struct.fields s) in
          let kinds = List.map (fun fi -> (fi, classify_field fi)) fis in
          Format.fprintf fmt
            "Jsont.Object.map ~kind:%S %a%a@;|> Jsont.Object.finish" name
            (fun fmt k -> pp_struct_jsont_constructor_rec fmt tmn k)
            kinds
            (Format.pp_print_list
               ~pp_sep:(fun _ () -> ())
               (fun fmt (fi, k) ->
                 let te =
                   match fi.sf_type with Type_expr.Optional t -> t | t -> t
                 in
                 match k with
                 | Mem_default _ ->
                     Format.fprintf fmt "@;|> Jsont.Object.opt_mem %S %a"
                       fi.sf_name (pp_jsont_expr_rec group) te
                 | Mem_optional ->
                     Format.fprintf fmt
                       "@;|> Jsont.Object.opt_mem %S (Jsont.option %a)"
                       fi.sf_name (pp_jsont_expr_rec group) te
                 | Mem_required ->
                     Format.fprintf fmt "@;|> Jsont.Object.mem %S %a" fi.sf_name
                       (pp_jsont_expr_rec group) te))
            kinds
      | Type_def.Union u ->
          let fis = List.map union_field_info_of (Union.fields u) in
          Format.fprintf fmt
            "Jsont.map ~kind:%S@;\
             <0 2>@[<v 0>~dec:(fun json ->@;\
             <0 2>@[<v 0>match json with@;\
             | Jsont.Object ([ ((key, _), value) ], _) ->@;\
             <0 2>@[<v 0>(match key with@;\
             %a@ | k -> Jsont.Error.msgf Jsont.Meta.none \"Unknown %s case: \
             %%s\" k)@]@ | Jsont.Object _ ->@;\
             <0 2>Jsont.Error.msgf Jsont.Meta.none \"%s must have exactly one \
             member\"@ | _ ->@;\
             <0 2>Jsont.Error.msgf Jsont.Meta.none \"%s must be a JSON \
             object\")@]@ ~enc:(fun _ -> assert false)@ Jsont.json@]"
            name
            (Format.pp_print_list
               ~pp_sep:(fun fmt () -> Format.fprintf fmt "@ ")
               (pp_union_case_rec group tmn))
            fis name name name)
    ()

let pp_recursive_group fmt names types =
  let group =
    List.fold_left (fun s n -> String_set.add n s) String_set.empty names
  in

  (* Step 1: module rec Name_Type ... and Name2_Type ... *)
  List.iteri
    (fun i name ->
      let typedef = String_map.find name types in
      let kw = if i = 0 then "module rec" else "and" in
      if i > 0 then Format.fprintf fmt "@\n@;";
      pp_type_module fmt group kw name typedef)
    names;

  (* Step 2: let rec name_jsont = lazy (...) and ... *)
  Format.fprintf fmt "@\n@;";
  List.iteri
    (fun i name ->
      let typedef = String_map.find name types in
      let kw = if i = 0 then "let rec" else "and" in
      if i > 0 then Format.fprintf fmt "@;";
      pp_jsont_decl fmt group name typedef kw)
    names;

  (* Step 3: module facades *)
  Format.fprintf fmt "@\n@;";
  Format.pp_print_list
    ~pp_sep:(fun fmt () -> Format.fprintf fmt "@\n@;")
    (fun fmt name ->
      Format.fprintf fmt
        "@[<v 0>module %s = struct@;\
         <0 2>@[<v 0>include %s@\n\
         @;\
         let jsont = Jsont.rec' %s@]@ end@]"
        name (type_module_name name) (jsont_lazy_name name))
    fmt names

let pp_block fmt scc types =
  match scc with
  | [ name ] ->
      let td = String_map.find name types in
      if String_set.mem name (refs_of_typedef td) then
        pp_recursive_group fmt scc types
      else pp_single fmt name td
  | _ -> pp_recursive_group fmt scc types

let generate (types : Type_def.t String_map.t) : string =
  let buf = Buffer.create 4096 in
  let fmt = Format.formatter_of_buffer buf in
  let sccs = compute_sccs types in
  (* [float_jsont] (PyTorch's string-encoded non-finite floats) comes from the
     opened Schema_runtime; every Float field's decoder refers to it. *)
  Format.fprintf fmt "@[<v 0>open Schema_runtime";
  List.iter
    (fun scc ->
      Format.fprintf fmt "@\n@;";
      pp_block fmt scc types)
    sccs;
  Format.fprintf fmt "@]@.";
  Buffer.contents buf
