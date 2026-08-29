module String_map = Map.Make (String)

(* A field in a struct or union type definition.
   The type_ value is kept as a raw string; resolution happens at a higher level. *)
module Field = struct
  type t = { name : string; type_ : string; default : string option }

  let make name type_ default = { name; type_; default }
  let name t = t.name
  let type_ t = t.type_
  let default t = t.default

  let content_jsont : (string * string option) Jsont.t =
    Jsont.Object.map ~kind:"Field" (fun type_ default -> (type_, default))
    |> Jsont.Object.mem "type" Jsont.string
    |> Jsont.Object.opt_mem "default" Jsont.string
    |> Jsont.Object.finish

  (* Decode the fields object for a struct or union, preserving declaration order. *)
  let list_jsont : t list Jsont.t =
    Jsont.map ~kind:"Fields"
      ~dec:(fun json ->
        match json with
        | Jsont.Object (members, _) ->
            List.map
              (fun ((name, _), value) ->
                match Jsont.Json.decode content_jsont value with
                | Ok (type_, default) -> make name type_ default
                | Error s -> Jsont.Error.msg Jsont.Meta.none s)
              members
        | _ -> Jsont.Error.msgf Jsont.Meta.none "Expected object for fields")
      Jsont.json
end

(* A variant in an enum type definition. *)
module Enum_field = struct
  type t = { name : string; value : int }

  let make name value = { name; value }
  let name t = t.name
  let value t = t.value

  (* Decode the fields object for an enum, preserving declaration order. *)
  let list_jsont : t list Jsont.t =
    Jsont.map ~kind:"EnumFields"
      ~dec:(fun json ->
        match json with
        | Jsont.Object (members, _) ->
            List.map
              (fun ((name, _), value) ->
                match Jsont.Json.decode Jsont.int value with
                | Ok v -> make name v
                | Error s -> Jsont.Error.msg Jsont.Meta.none s)
              members
        | _ ->
            Jsont.Error.msgf Jsont.Meta.none "Expected object for enum fields")
      Jsont.json
end

module Struct = struct
  type t = { fields : Field.t list }

  let make fields = { fields }
  let fields t = t.fields
end

module Union = struct
  type t = { fields : Field.t list }

  let make fields = { fields }
  let fields t = t.fields
end

module Enum = struct
  type t = { fields : Enum_field.t list }

  let make fields = { fields }
  let fields t = t.fields
end

(* A schema type definition, dispatched on the "kind" member. *)
module Type_def = struct
  type t = Enum of Enum.t | Struct of Struct.t | Union of Union.t

  let jsont : t Jsont.t =
    let struct_body =
      Jsont.Object.map ~kind:"StructBody" Fun.id
      |> Jsont.Object.mem "fields" Field.list_jsont
      |> Jsont.Object.finish
    in
    let union_body =
      Jsont.Object.map ~kind:"UnionBody" Fun.id
      |> Jsont.Object.mem "fields" Field.list_jsont
      |> Jsont.Object.finish
    in
    let enum_body =
      Jsont.Object.map ~kind:"EnumBody" Fun.id
      |> Jsont.Object.mem "fields" Enum_field.list_jsont
      |> Jsont.Object.finish
    in
    let struct_case =
      Jsont.Object.Case.map
        ~dec:(fun fields -> Struct (Struct.make fields))
        "struct" struct_body
    in
    let union_case =
      Jsont.Object.Case.map
        ~dec:(fun fields -> Union (Union.make fields))
        "union" union_body
    in
    let enum_case =
      Jsont.Object.Case.map
        ~dec:(fun fields -> Enum (Enum.make fields))
        "enum" enum_body
    in
    Jsont.Object.map ~kind:"TypeDef" Fun.id
    |> Jsont.Object.case_mem "kind" Jsont.string
         [
           Jsont.Object.Case.make enum_case;
           Jsont.Object.Case.make struct_case;
           Jsont.Object.Case.make union_case;
         ]
    |> Jsont.Object.finish
end

(* The top-level schema: a map of type names to definitions, plus version fields. *)
module Schema = struct
  type t = {
    types : Type_def.t String_map.t;
    schema_version : int * int;
    treespec_version : int;
  }

  let make types schema_version treespec_version =
    { types; schema_version; treespec_version }

  let types t = t.types
  let schema_version t = t.schema_version
  let treespec_version t = t.treespec_version

  (* The top-level YAML is a heterogeneous object: most keys are type definitions,
     but SCHEMA_VERSION and TREESPEC_VERSION are special. *)
  let jsont : t Jsont.t =
    Jsont.map ~kind:"Schema"
      ~dec:(fun json ->
        match json with
        | Jsont.Object (members, _) ->
            List.fold_left
              (fun acc ((name, _), value) ->
                match name with
                | "SCHEMA_VERSION" -> (
                    match Jsont.Json.decode (Jsont.list Jsont.int) value with
                    | Ok [ major; minor ] ->
                        { acc with schema_version = (major, minor) }
                    | _ -> acc)
                | "TREESPEC_VERSION" -> (
                    match Jsont.Json.decode Jsont.int value with
                    | Ok v -> { acc with treespec_version = v }
                    | Error s -> Jsont.Error.msg Jsont.Meta.none s)
                | type_name -> (
                    match Jsont.Json.decode Type_def.jsont value with
                    | Ok td ->
                        {
                          acc with
                          types = String_map.add type_name td acc.types;
                        }
                    | Error s -> Jsont.Error.msg Jsont.Meta.none s))
              (make String_map.empty (0, 0) 0)
              members
        | _ -> Jsont.Error.msgf Jsont.Meta.none "Expected object for schema")
      Jsont.json

  let of_yaml_string s = Yamlt.of_string jsont s
end
