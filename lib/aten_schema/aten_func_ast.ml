module Base = struct
  type t =
    | Bool
    | Device
    | DeviceIndex
    | Dimname
    | DimVector
    | Float
    | Generator
    | GraphModule
    | Int
    | Layout
    | MemoryFormat
    | QScheme
    | Scalar
    | ScalarType
    | Storage
    | Stream
    | Str
    | SymBool
    | SymInt
    | Tensor

  let to_string = function
    | Bool -> "bool"
    | Device -> "Device"
    | DeviceIndex -> "DeviceIndex"
    | Dimname -> "Dimname"
    | DimVector -> "DimVector"
    | Float -> "float"
    | Generator -> "Generator"
    | GraphModule -> "GraphModule"
    | Int -> "int"
    | Layout -> "Layout"
    | MemoryFormat -> "MemoryFormat"
    | QScheme -> "QScheme"
    | Scalar -> "Scalar"
    | ScalarType -> "ScalarType"
    | Storage -> "Storage"
    | Stream -> "Stream"
    | Str -> "str"
    | SymBool -> "SymBool"
    | SymInt -> "SymInt"
    | Tensor -> "Tensor"

  let pp fmt t = Format.pp_print_string fmt (to_string t)
end

module Type = struct
  type t = Base of Base.t | List of t * int option | Optional of t

  let rec pp fmt = function
    | Base b -> Base.pp fmt b
    | List (t, None) -> Format.fprintf fmt "%a[]" pp t
    | List (t, Some n) -> Format.fprintf fmt "%a[%d]" pp t n
    | Optional t -> Format.fprintf fmt "%a?" pp t
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
    | Bool of bool
    | Float of string (* preserved as-is to keep exact representation *)
    | Ident of string (* bare identifier, e.g. contiguous_format, Mean *)
    | Int of int
    | IntList of int list
    | None
    | Str of string

  let to_string = function
    | Bool false -> "False"
    | Bool true -> "True"
    | Float s -> s
    | Ident s -> s
    | Int n -> string_of_int n
    | IntList ns -> "[" ^ String.concat "," (List.map string_of_int ns) ^ "]"
    | None -> "None"
    | Str s -> Printf.sprintf "%S" s

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
    Option.iter (fun d -> Format.fprintf fmt "=%a" Default.pp d) a.default
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

  (* Kept as a match: [aten_schema] depends only on jsont/yamlt, and pulling in
     [fmt] to spell this as [Fmt.option] would buy nothing. *)
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
