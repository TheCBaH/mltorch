module Base = struct
  type t =
    | Generator
    | ScalarType
    | Tensor
    | Int
    | Dimname
    | DimVector
    | Float
    | Str
    | Bool
    | Layout
    | Device
    | DeviceIndex
    | Scalar
    | MemoryFormat
    | QScheme
    | Storage
    | Stream
    | SymInt
    | SymBool
    | GraphModule

  let to_string = function
    | Generator -> "Generator"
    | ScalarType -> "ScalarType"
    | Tensor -> "Tensor"
    | Int -> "int"
    | Dimname -> "Dimname"
    | DimVector -> "DimVector"
    | Float -> "float"
    | Str -> "str"
    | Bool -> "bool"
    | Layout -> "Layout"
    | Device -> "Device"
    | DeviceIndex -> "DeviceIndex"
    | Scalar -> "Scalar"
    | MemoryFormat -> "MemoryFormat"
    | QScheme -> "QScheme"
    | Storage -> "Storage"
    | Stream -> "Stream"
    | SymInt -> "SymInt"
    | SymBool -> "SymBool"
    | GraphModule -> "GraphModule"

  let pp fmt t = Format.pp_print_string fmt (to_string t)
end

module Type = struct
  type t = Base of Base.t | Optional of t | List of t * int option

  let rec pp fmt = function
    | Base b -> Base.pp fmt b
    | Optional t -> Format.fprintf fmt "%a?" pp t
    | List (t, None) -> Format.fprintf fmt "%a[]" pp t
    | List (t, Some n) -> Format.fprintf fmt "%a[%d]" pp t n
end

module Annotation = struct
  (* alias_set_after is non-empty only for "a -> *" / "a -> b" forms *)
  type t = {
    alias_set : string list;
    is_write : bool;
    alias_set_after : string list;
  }

  let pp fmt t =
    let after =
      if t.alias_set_after = [] then ""
      else " -> " ^ String.concat "|" t.alias_set_after
    in
    Format.fprintf fmt "(%s%s%s)"
      (String.concat "|" t.alias_set)
      (if t.is_write then "!" else "")
      after
end

module Default = struct
  type t =
    | None
    | Bool of bool
    | Int of int
    | Float of string (* preserved as-is to keep exact representation *)
    | Str of string
    | IntList of int list
    | Ident of string (* bare identifier, e.g. contiguous_format, Mean *)

  let to_string = function
    | None -> "None"
    | Bool true -> "True"
    | Bool false -> "False"
    | Int n -> string_of_int n
    | Float s -> s
    | Str s -> Printf.sprintf "%S" s
    | IntList ns -> "[" ^ String.concat "," (List.map string_of_int ns) ^ "]"
    | Ident s -> s

  let pp fmt t = Format.pp_print_string fmt (to_string t)
end

module Argument = struct
  type t = {
    name : string;
    ty : Type.t;
    annotation : Annotation.t option;
    default : Default.t option;
  }

  let pp fmt (a : t) =
    (match a.annotation with
    | None -> Type.pp fmt a.ty
    | Some ann -> Format.fprintf fmt "%a%a" Type.pp a.ty Annotation.pp ann);
    Format.fprintf fmt " %s" a.name;
    match a.default with
    | None -> ()
    | Some d -> Format.fprintf fmt "=%a" Default.pp d
end

module Return = struct
  type t = {
    name : string option;
    ty : Type.t;
    annotation : Annotation.t option;
  }

  let pp fmt (r : t) =
    (match r.annotation with
    | None -> Type.pp fmt r.ty
    | Some ann -> Format.fprintf fmt "%a%a" Type.pp r.ty Annotation.pp ann);
    match r.name with None -> () | Some s -> Format.fprintf fmt " %s" s
end

module Op_name = struct
  type t = { base : string; overload : string option }

  let pp fmt t =
    match t.overload with
    | None -> Format.pp_print_string fmt t.base
    | Some ov -> Format.fprintf fmt "%s.%s" t.base ov
end

module Arguments = struct
  type t = {
    positional : Argument.t list;
    kwarg_only : Argument.t list;
    out : Argument.t list;
  }

  let pp fmt (args : t) =
    let sep = ref false in
    let pr a =
      if !sep then Format.pp_print_string fmt ", ";
      Argument.pp fmt a;
      sep := true
    in
    List.iter pr args.positional;
    if args.kwarg_only <> [] || args.out <> [] then begin
      if !sep then Format.pp_print_string fmt ", ";
      Format.pp_print_string fmt "*";
      sep := true;
      List.iter pr args.kwarg_only;
      List.iter pr args.out
    end
end

type t = { name : Op_name.t; arguments : Arguments.t; returns : Return.t list }

let pp fmt (fs : t) =
  let pp_rets fmt = function
    | [ r ] -> Return.pp fmt r
    | rs ->
        Format.pp_print_char fmt '(';
        Format.pp_print_list
          ~pp_sep:(fun fmt () -> Format.pp_print_string fmt ", ")
          Return.pp fmt rs;
        Format.pp_print_char fmt ')'
  in
  Format.fprintf fmt "%a(%a) -> %a" Op_name.pp fs.name Arguments.pp fs.arguments
    pp_rets fs.returns
