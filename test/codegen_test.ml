open Pytorch_schema
open Schema_codegen

let sm pairs =
  List.fold_left (fun m (k, v) -> String_map.add k v m) String_map.empty pairs

(* ---- Phase 1: enums ---- *)

let%expect_test "simple enum" =
  let schema =
    sm
      [
        ( "ArgumentKind",
          Type_def.Enum
            (Enum.make
               [
                 Enum_field.make "UNKNOWN" 0;
                 Enum_field.make "POSITIONAL" 1;
                 Enum_field.make "KEYWORD" 2;
               ]) );
      ]
  in
  print_string (generate schema);
  [%expect
    {|
    open Schema_runtime

    module ArgumentKind = struct
      type t = UNKNOWN | POSITIONAL | KEYWORD

      let jsont =
        Jsont.map ~kind:"ArgumentKind"
          ~dec:(fun n ->
            match n with
            | 0 -> UNKNOWN
            | 1 -> POSITIONAL
            | 2 -> KEYWORD
            | n -> Jsont.Error.msgf Jsont.Meta.none "Unknown ArgumentKind value: %d" n)
          ~enc:(fun v ->
            match v with
            | UNKNOWN -> 0
            | POSITIONAL -> 1
            | KEYWORD -> 2)
          Jsont.int

      let make v = v
    end |}]

(* ---- Phase 2: simple structs ---- *)

let%expect_test "single-field struct" =
  let schema =
    sm
      [
        ( "TensorArgument",
          Type_def.Struct (Struct.make [ Field.make "name" "str" None ]) );
      ]
  in
  print_string (generate schema);
  [%expect
    {|
    open Schema_runtime

    module TensorArgument = struct
      type t = { name : string }

      let make name =
        { name }

      let jsont =
        Jsont.Object.map ~kind:"TensorArgument" make
        |> Jsont.Object.mem "name" Jsont.string
        |> Jsont.Object.finish
    end |}]

let%expect_test "two-field struct" =
  let schema =
    sm
      [
        ( "SchemaVersion",
          Type_def.Struct
            (Struct.make
               [ Field.make "major" "int" None; Field.make "minor" "int" None ])
        );
      ]
  in
  print_string (generate schema);
  [%expect
    {|
    open Schema_runtime

    module SchemaVersion = struct
      type t = {
        major : int;
        minor : int;
      }

      let make major minor =
        { major; minor }

      let jsont =
        Jsont.Object.map ~kind:"SchemaVersion" make
        |> Jsont.Object.mem "major" Jsont.int
        |> Jsont.Object.mem "minor" Jsont.int
        |> Jsont.Object.finish
    end |}]

(* ---- Phase 3: Optional fields ---- *)

let%expect_test "struct with Optional fields" =
  let schema =
    sm
      [
        ( "RangeConstraint",
          Type_def.Struct
            (Struct.make
               [
                 Field.make "min_val" "Optional[int]" None;
                 Field.make "max_val" "Optional[int]" None;
               ]) );
      ]
  in
  print_string (generate schema);
  [%expect
    {|
    open Schema_runtime

    module RangeConstraint = struct
      type t = {
        min_val : int option;
        max_val : int option;
      }

      let make min_val max_val =
        { min_val; max_val }

      let jsont =
        Jsont.Object.map ~kind:"RangeConstraint" (fun min_val max_val ->
          { min_val = Option.join min_val
          ; max_val = Option.join max_val })
        |> Jsont.Object.opt_mem "min_val" (Jsont.option Jsont.int)
        |> Jsont.Object.opt_mem "max_val" (Jsont.option Jsont.int)
        |> Jsont.Object.finish
    end |}]

let%expect_test "struct with reserved-word field" =
  let schema =
    sm
      [
        ( "Device",
          Type_def.Struct
            (Struct.make
               [
                 Field.make "type" "str" None;
                 Field.make "index" "Optional[int]" (Some "None");
               ]) );
      ]
  in
  print_string (generate schema);
  [%expect
    {|
    open Schema_runtime

    module Device = struct
      type t = {
        type_ : string;
        index : int option;
      }

      let make type_ index =
        { type_; index }

      let jsont =
        Jsont.Object.map ~kind:"Device" (fun type_ index ->
          { type_
          ; index = Option.join index })
        |> Jsont.Object.mem "type" Jsont.string
        |> Jsont.Object.opt_mem "index" (Jsont.option Jsont.int)
        |> Jsont.Object.finish
    end |}]

(* ---- Phase 4: field defaults ---- *)

let%expect_test "struct with non-optional default fields" =
  let schema =
    sm
      [
        ( "ExportedProgram",
          Type_def.Struct
            (Struct.make
               [
                 Field.make "graph_module" "GraphModule" None;
                 Field.make "schema_version" "SchemaVersion" None;
                 Field.make "verifiers" "List[str]" (Some "'[]'");
                 Field.make "torch_version" "str" (Some "<=2.4");
               ]) );
      ]
  in
  print_string (generate schema);
  [%expect
    {|
    open Schema_runtime

    module ExportedProgram = struct
      type t = {
        graph_module : GraphModule.t;
        schema_version : SchemaVersion.t;
        verifiers : string list;
        torch_version : string;
      }

      let make graph_module schema_version verifiers torch_version =
        { graph_module; schema_version; verifiers; torch_version }

      let jsont =
        Jsont.Object.map ~kind:"ExportedProgram" (fun graph_module schema_version verifiers_opt torch_version_opt ->
          { graph_module
          ; schema_version
          ; verifiers = (match verifiers_opt with None -> [] | Some v -> v)
          ; torch_version = (match torch_version_opt with None -> "<=2.4" | Some v -> v) })
        |> Jsont.Object.mem "graph_module" GraphModule.jsont
        |> Jsont.Object.mem "schema_version" SchemaVersion.jsont
        |> Jsont.Object.opt_mem "verifiers" (Jsont.list Jsont.string)
        |> Jsont.Object.opt_mem "torch_version" Jsont.string
        |> Jsont.Object.finish
    end |}]

(* ---- Phase 5: unions ---- *)

let%expect_test "simple union" =
  let schema =
    sm
      [
        ( "SymIntArgument",
          Type_def.Union
            (Union.make
               [
                 Field.make "as_name" "str" None; Field.make "as_int" "int" None;
               ]) );
      ]
  in
  print_string (generate schema);
  [%expect
    {|
    open Schema_runtime

    module SymIntArgument = struct
      type t =
        | Name of string
        | Int of int

      let jsont =
        Jsont.map ~kind:"SymIntArgument"
          ~dec:(fun json ->
            match json with
            | Jsont.Object ([ ((key, _), value) ], _) ->
              (match key with
              | "as_name" ->
                (match Jsont.Json.decode Jsont.string value with
                | Ok v -> Name v
                | Error s -> Jsont.Error.msg Jsont.Meta.none s)
              | "as_int" ->
                (match Jsont.Json.decode Jsont.int value with
                | Ok v -> Int v
                | Error s -> Jsont.Error.msg Jsont.Meta.none s)
              | k -> Jsont.Error.msgf Jsont.Meta.none "Unknown SymIntArgument case: %s" k)
            | Jsont.Object _ ->
              Jsont.Error.msgf Jsont.Meta.none "SymIntArgument must have exactly one member"
            | _ ->
              Jsont.Error.msgf Jsont.Meta.none "SymIntArgument must be a JSON object")
          ~enc:(fun _ -> assert false)
          Jsont.json

      let make v = v
    end |}]

let%expect_test "union with heterogeneous cases" =
  let schema =
    sm
      [
        ( "SymExprHint",
          Type_def.Union
            (Union.make
               [
                 Field.make "as_int" "int" None;
                 Field.make "as_bool" "bool" None;
                 Field.make "as_float" "float" None;
               ]) );
      ]
  in
  print_string (generate schema);
  [%expect
    {|
    open Schema_runtime

    module SymExprHint = struct
      type t =
        | Int of int
        | Bool of bool
        | Float of float

      let jsont =
        Jsont.map ~kind:"SymExprHint"
          ~dec:(fun json ->
            match json with
            | Jsont.Object ([ ((key, _), value) ], _) ->
              (match key with
              | "as_int" ->
                (match Jsont.Json.decode Jsont.int value with
                | Ok v -> Int v
                | Error s -> Jsont.Error.msg Jsont.Meta.none s)
              | "as_bool" ->
                (match Jsont.Json.decode Jsont.bool value with
                | Ok v -> Bool v
                | Error s -> Jsont.Error.msg Jsont.Meta.none s)
              | "as_float" ->
                (match Jsont.Json.decode float_jsont value with
                | Ok v -> Float v
                | Error s -> Jsont.Error.msg Jsont.Meta.none s)
              | k -> Jsont.Error.msgf Jsont.Meta.none "Unknown SymExprHint case: %s" k)
            | Jsont.Object _ ->
              Jsont.Error.msgf Jsont.Meta.none "SymExprHint must have exactly one member"
            | _ ->
              Jsont.Error.msgf Jsont.Meta.none "SymExprHint must be a JSON object")
          ~enc:(fun _ -> assert false)
          Jsont.json

      let make v = v
    end |}]

(* ---- Phase 6: type references between modules ---- *)

let%expect_test "struct referencing another struct" =
  let schema =
    sm
      [
        ( "TensorArgument",
          Type_def.Struct (Struct.make [ Field.make "name" "str" None ]) );
        ( "BufferMutationSpec",
          Type_def.Struct
            (Struct.make
               [
                 Field.make "arg" "TensorArgument" None;
                 Field.make "buffer_name" "str" None;
               ]) );
      ]
  in
  print_string (generate schema);
  [%expect
    {|
    open Schema_runtime

    module TensorArgument = struct
      type t = { name : string }

      let make name =
        { name }

      let jsont =
        Jsont.Object.map ~kind:"TensorArgument" make
        |> Jsont.Object.mem "name" Jsont.string
        |> Jsont.Object.finish
    end

    module BufferMutationSpec = struct
      type t = {
        arg : TensorArgument.t;
        buffer_name : string;
      }

      let make arg buffer_name =
        { arg; buffer_name }

      let jsont =
        Jsont.Object.map ~kind:"BufferMutationSpec" make
        |> Jsont.Object.mem "arg" TensorArgument.jsont
        |> Jsont.Object.mem "buffer_name" Jsont.string
        |> Jsont.Object.finish
    end |}]

let%expect_test "struct with List and Dict references" =
  let schema =
    sm
      [
        ( "NamedTupleDef",
          Type_def.Struct
            (Struct.make [ Field.make "field_names" "List[str]" None ]) );
        ( "Program",
          Type_def.Struct
            (Struct.make
               [ Field.make "methods" "Dict[str, ExportedProgram]" None ]) );
      ]
  in
  print_string (generate schema);
  [%expect
    {|
    open Schema_runtime

    module NamedTupleDef = struct
      type t = { field_names : string list }

      let make field_names =
        { field_names }

      let jsont =
        Jsont.Object.map ~kind:"NamedTupleDef" make
        |> Jsont.Object.mem "field_names" (Jsont.list Jsont.string)
        |> Jsont.Object.finish
    end

    module Program = struct
      type t = { methods : ExportedProgram.t String_map.t }

      let make methods =
        { methods }

      let jsont =
        Jsont.Object.map ~kind:"Program" make
        |> Jsont.Object.mem "methods" (Jsont.Object.as_string_map ExportedProgram.jsont)
        |> Jsont.Object.finish
    end |}]

(* ---- Phase 7: recursive types ---- *)

(* GraphArgument <-> Graph: two-node cycle *)
let%expect_test "two-node mutual recursion" =
  let schema =
    sm
      [
        ( "GraphArgument",
          Type_def.Struct
            (Struct.make
               [ Field.make "name" "str" None; Field.make "graph" "Graph" None ])
        );
        ( "Graph",
          Type_def.Struct
            (Struct.make [ Field.make "outputs" "List[GraphArgument]" None ]) );
      ]
  in
  print_string (generate schema);
  [%expect
    {|
    open Schema_runtime

    module rec Graph_Type : sig
      type t = { outputs : GraphArgument_Type.t list }
      val make : GraphArgument_Type.t list -> t
    end = struct
      type t = { outputs : GraphArgument_Type.t list }

      let make outputs =
        { outputs }
    end

    and GraphArgument_Type : sig
      type t = {
        name : string;
        graph : Graph_Type.t;
      }
      val make : string -> Graph_Type.t -> t
    end = struct
      type t = {
        name : string;
        graph : Graph_Type.t;
      }

      let make name graph =
        { name; graph }
    end

    let rec graph_jsont : Graph_Type.t Jsont.t Lazy.t = lazy (
      Jsont.Object.map ~kind:"Graph" Graph_Type.make
      |> Jsont.Object.mem "outputs" (Jsont.list (Lazy.force graph_argument_jsont))
      |> Jsont.Object.finish)
    and graph_argument_jsont : GraphArgument_Type.t Jsont.t Lazy.t = lazy (
      Jsont.Object.map ~kind:"GraphArgument" GraphArgument_Type.make
      |> Jsont.Object.mem "name" Jsont.string
      |> Jsont.Object.mem "graph" (Lazy.force graph_jsont)
      |> Jsont.Object.finish)

    module Graph = struct
      include Graph_Type

      let jsont = Jsont.rec' graph_jsont
    end

    module GraphArgument = struct
      include GraphArgument_Type

      let jsont = Jsont.rec' graph_argument_jsont
    end |}]
