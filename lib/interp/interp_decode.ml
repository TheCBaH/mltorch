(* Runtime support for the graph interpreter: the immutable environment, the
   per-argument decoders (export Node.t argument -> ATen-binding value), and the
   output binders. Both the hand-written dispatch in [Interp] and the generated
   dispatch in [Interp_dispatch] are written against these. The decoders are the
   dual of Aten_c_type.map_type: where the binding generator lowers a schema arg
   to its ctypes fragments, these pull those same fragments back out of an
   exported graph node. *)

open Ctypes
open Pytorch_types
open Schema_runtime
module O = Aten_c.Aten_operations

let arr xs = CArray.start (CArray.of_list int64_t (List.map Int64.of_int xs))

(* An absent optional Tensor argument (e.g. a convolution with no bias) crosses
   as a null handle, which the op shim maps to std::nullopt. *)
let null_tensor = coerce (ptr void) Aten_function_description.atc_tensor null

(* The ATen tensor handle: a pointer to the opaque atc_tensor struct, exactly
   what the bound ops take and return. *)
type tensor = [ `atc_tensor_opaque ] Ctypes.structure Ctypes.ptr

(* SSA value name -> tensor. Immutable: each node reads the current bindings and
   the fold produces an extended environment. *)
type env = tensor String_map.t

(* What a decode helper demanded. Closed: exactly the seventeen shapes the
   helpers below accept, each of which used to spell itself as a literal at its
   own call site. The parallel set on the native path is
   [Native_interp.arg_kind] -- deliberately NOT shared, since neither library
   depends on the other and a common module for two closed lists would be
   speculative. *)
type expected_kind =
  [ `Tensor_opt
  | `Tensor_list
  | `Tensor_or_scalar
  | `Int
  | `Int_opt
  | `Int_list
  | `Bool
  | `Bool_opt
  | `Float
  | `Float_opt
  | `Scalar
  | `Scalar_opt
  | `String
  | `Memory_format_opt
  | `Scalar_type_opt
  | `Layout_opt
  | `Device_opt ]

module Wrong_argument_kind = struct
  type t = { arg : string; expected : expected_kind; actual : string }
  (* [actual] stays a string, from [argument_kind_name]: it ranges over the
     GENERATED [Argument.t], whose constructor set is schema-driven, and a
     hand-written variant mirroring it here is exactly the drift the schema
     codegen exists to prevent. *)
end

module Unexpected_output_kind = struct
  type t = { index : int; actual : string }
end

module Tensor_list_output = struct
  type t = { actual : string list }
  (* A Tensor[]-returning node must carry exactly one output, of kind Tensor[].
     [actual] is the kinds actually present, in order -- the whole output shape
     rather than one index, since "too many outputs" and "wrong kind at 0" are
     the same malformation of the same list. *)
end

module Unresolved_sym_arg = struct
  type t = { arg : string; symbol : string }
  (* A [SymInt]-typed argument the exporter wrote as a SYMBOL rather than a
     value ([as_name], [SymIntArgument.Name]). The interpreter has no shape
     environment to evaluate it in, so it is a refusal, not a decode failure of
     the wrong kind -- the argument IS the kind the schema declares.

     Both fields: [arg] is the schema slot ("start"), [symbol] the serialized
     name ("s3"), and a reader needs the pair to tell "this graph is dynamic"
     from "this one slot is". Same distinction the tensor-metadata rule already
     draws with [`Bad_dimension { fault = `Symbolic }] in Native_interp. *)
end

module Tensor_list_arity = struct
  type t = { names : int; tensors : int }
  (* Serialized SSA names vs. tensors ATen actually returned. Both counts, since
     which side is larger says whether names would be left unbound or tensors
     dropped. *)
end

type error =
  [ `Undefined_value of string
  | `Missing_argument of string
  | `Wrong_argument_kind of Wrong_argument_kind.t
  | `Unresolved_sym_arg of Unresolved_sym_arg.t
  | `Unsupported_scalar_type_arg of string
  | `Unsupported_layout_arg of string
  | `Unsupported_device_arg of string
  | `Unknown_memory_format of string
  | `Unexpected_output_kind of Unexpected_output_kind.t
  | `Missing_output of int
    (* Was [`Unexpected_output_kind (i, "missing")]: no output at that index at
       all is a different fact from one of the wrong kind, and a sentinel
       string in the payload is not a way to say so. *)
  | `Expected_tensor_list_output of Tensor_list_output.t
  | `Tensor_list_output_arity_mismatch of Tensor_list_arity.t
    (* NOT [`Output_arity_mismatch]: that tag already exists in Eval_direct and
       Eval_direct4 with a different payload, and sharing the name across
       disjoint rows is a trap for whoever later composes them. *)
  ]

let expected_kind_name : expected_kind -> string = function
  | `Tensor_opt -> "tensor?"
  | `Tensor_list -> "tensor[]"
  | `Tensor_or_scalar -> "tensor|scalar"
  | `Int -> "int"
  | `Int_opt -> "int?"
  | `Int_list -> "int[]"
  | `Bool -> "bool"
  | `Bool_opt -> "bool?"
  | `Float -> "float"
  | `Float_opt -> "float?"
  | `Scalar -> "scalar"
  | `Scalar_opt -> "scalar?"
  | `String -> "string"
  | `Memory_format_opt -> "memory_format?"
  | `Scalar_type_opt -> "scalar_type?"
  | `Layout_opt -> "layout?"
  | `Device_opt -> "device?"

let pp_error ppf : [< error ] -> unit = function
  | `Undefined_value name -> Fmt.pf ppf "undefined SSA value %S" name
  | `Missing_argument name -> Fmt.pf ppf "missing required argument %S" name
  | `Wrong_argument_kind { Wrong_argument_kind.arg; expected; actual } ->
      Fmt.pf ppf "argument %S: expected %s, got %s" arg
        (expected_kind_name expected)
        actual
  | `Unresolved_sym_arg { Unresolved_sym_arg.arg; symbol } ->
      Fmt.pf ppf "argument %S: unresolved symbolic value %S" arg symbol
  | `Unsupported_scalar_type_arg name ->
      Fmt.pf ppf "unsupported scalar_type argument %S" name
  | `Unsupported_layout_arg name ->
      Fmt.pf ppf "unsupported layout argument %S" name
  | `Unsupported_device_arg name ->
      Fmt.pf ppf "unsupported device argument %S" name
  | `Unknown_memory_format name ->
      Fmt.pf ppf "unknown memory_format for argument %S" name
  | `Unexpected_output_kind { Unexpected_output_kind.index; actual } ->
      Fmt.pf ppf "output %d: expected tensor/None, got %s" index actual
  | `Missing_output i -> Fmt.pf ppf "output %d: missing" i
  | `Expected_tensor_list_output { Tensor_list_output.actual } ->
      Fmt.pf ppf "expected one Tensor[] output, got [%s]"
        (String.concat "; " actual)
  | `Tensor_list_output_arity_mismatch { Tensor_list_arity.names; tensors } ->
      Fmt.pf ppf
        "tensor-list output arity: %d serialized name%s, %d tensor%s returned"
        names
        (if names = 1 then "" else "s")
        tensors
        (if tensors = 1 then "" else "s")

let argument_kind_name = function
  | Argument.None _ -> "None"
  | Tensor _ -> "Tensor"
  | Tensors _ -> "Tensor[]"
  | Int _ -> "Int"
  | Ints _ -> "Int[]"
  | Float _ -> "Float"
  | Floats _ -> "Float[]"
  | String _ -> "String"
  | Strings _ -> "String[]"
  | Sym_int _ -> "SymInt"
  | Sym_ints _ -> "SymInt[]"
  | Scalar_type _ -> "ScalarType"
  | Memory_format _ -> "MemoryFormat"
  | Layout _ -> "Layout"
  | Device _ -> "Device"
  | Bool _ -> "Bool"
  | Bools _ -> "Bool[]"
  | Sym_bool _ -> "SymBool"
  | Sym_bools _ -> "SymBool[]"
  | Graph _ -> "Graph"
  | Optional_tensors _ -> "OptionalTensor[]"
  | Custom_obj _ -> "CustomObj"
  | Operator _ -> "Operator"
  | Sym_float _ -> "SymFloat"
  | Sym_floats _ -> "SymFloat[]"
  | Optional_tensor _ -> "OptionalTensor"
  | Complex _ -> "Complex"
  | Nested_tensors _ -> "Tensor[][]"
  | Int_lists _ -> "Int[][]"
  | String_to_argument _ -> "StringMap"
  | Float_lists _ -> "Float[][]"

(* --- argument decoding (by name; absent => default) --- *)

let find_arg (node : Node.t) name =
  List.find_map
    (fun (na : NamedArgument.t) ->
      if String.equal na.name name then Some na.arg else None)
    node.inputs

(* Resolve a value name to a tensor: an earlier node result or a preloaded
   parameter/buffer, all carried by the environment. *)
let resolve (env : env) name =
  String_map.find_opt name env |> Err.of_option (`Undefined_value name)

let wrong_kind arg expected a =
  Err.fail
    (`Wrong_argument_kind
       { Wrong_argument_kind.arg; expected; actual = argument_kind_name a })

let tensor_arg env node name =
  match find_arg node name with
  | Some (Argument.Tensor ta) -> resolve env ta.TensorArgument.name
  | Some (Argument.None _) -> Err.return null_tensor
  | Some (Argument.Optional_tensor (OptionalTensorArgument.Tensor ta)) ->
      resolve env ta.TensorArgument.name
  | Some (Argument.Optional_tensor (OptionalTensorArgument.None _)) ->
      Err.return null_tensor
  | Some arg -> wrong_kind name `Tensor_opt arg
  | None -> Err.fail (`Missing_argument name)

(* A binary op's Tensor-typed argument is normally an earlier node's result,
   but a compile-time scalar (e.g. the +3/relu6/.../6 in hardswish's
   decomposition) is serialized as a bare Int/Float rather than a lifted
   constant tensor; materialise it with full_like, broadcastable against
   [like]. *)
let tensor_or_scalar_arg env node name ~like =
  match find_arg node name with
  | Some (Argument.Tensor ta) -> resolve env ta.TensorArgument.name
  | Some (Argument.Int i) ->
      Err.return
        (O.full_like like
           (Aten_scalar.Int (Int64.of_int i))
           None None None None None)
  | Some (Argument.Float f) ->
      Err.return
        (O.full_like like (Aten_scalar.Float f) None None None None None)
  | Some arg -> wrong_kind name `Tensor_or_scalar arg
  | None -> Err.fail (`Missing_argument name)

(* A schema [SymInt] slot crosses as either [Argument.Int] -- the spelling every
   graph in the corpus uses, and the one [Aten_spec_run] emits -- or as
   [Argument.Sym_int], which carries a RESOLVED value ([Int]) or an unresolved
   SYMBOL ([Name]). Only the last is a refusal, and it is its own row rather
   than a wrong-kind: the argument is exactly the kind the schema declares, and
   what is missing is a shape environment to evaluate it in.

   One rule, applied by every [SymInt]-shaped decoder below, so "does this
   interpreter accept a dynamic bound" has one answer instead of three. *)
let sym_int_value ~arg = function
  | SymIntArgument.Int i -> Err.return i
  | SymIntArgument.Name symbol ->
      Err.fail (`Unresolved_sym_arg { Unresolved_sym_arg.arg; symbol })

let ints_arg ?(default = []) node name =
  match find_arg node name with
  | Some (Argument.Ints xs) -> Err.return xs
  | Some (Argument.Sym_ints xs) ->
      Err.List.map (fun s -> sym_int_value ~arg:name s) xs
  | Some (Argument.None _) | None -> Err.return default
  | Some arg -> wrong_kind name `Int_list arg

let require name = Err.of_option (`Missing_argument name)

let int_arg ?default node name =
  match find_arg node name with
  | Some (Argument.Int i) -> Err.return i
  | Some (Argument.Sym_int s) -> sym_int_value ~arg:name s
  | None -> require name default
  | Some arg -> wrong_kind name `Int arg

let bool_arg ?(default = false) node name =
  match find_arg node name with
  | Some (Argument.Bool b) -> Err.return b
  | None -> Err.return default
  | Some arg -> wrong_kind name `Bool arg

let float_arg ?default node name =
  match find_arg node name with
  | Some (Argument.Float f) -> Err.return f
  | None -> require name default
  | Some arg -> wrong_kind name `Float arg

(* A float? argument crosses as a [double ptr] (map_type lowers float? to a
   [double] pointer): a null pointer encodes None (the C shim maps null to
   std::nullopt), a live cell encodes the present value. The pointer form mirrors
   int?/SymInt?; it is the float counterpart the generator emits for e.g.
   rms_norm's eps. *)
let float_opt_ptr node name =
  match find_arg node name with
  | Some (Argument.Float f) -> Err.return (allocate double f)
  | Some (Argument.None _) | None -> Err.return (from_voidp double null)
  | Some arg -> wrong_kind name `Float_opt arg

(* int? / SymInt?: the integer twin of [float_opt_ptr], and the same encoding --
   [map_type] lowers both to a pointer ([int64_t*]), a null pointer being None
   and a live cell the present value (aten_c_type.ml:151-158). Emitted by the
   generator for e.g. slice.Tensor's [start]/[end] and avg_pool2d's
   [divisor_override].

   [Argument.None] and an ABSENT argument are both None. That is the schema's
   own default ([SymInt? start=None]) rather than a guess: unlike [int_arg],
   there is no other default a caller could want. *)
let int_opt_ptr node name =
  let open Err.Syntax in
  match find_arg node name with
  | Some (Argument.Int i) -> Err.return (allocate int64_t (Int64.of_int i))
  | Some (Argument.Sym_int s) ->
      let+ i = sym_int_value ~arg:name s in
      allocate int64_t (Int64.of_int i)
  | Some (Argument.None _) | None -> Err.return (from_voidp int64_t null)
  | Some arg -> wrong_kind name `Int_opt arg

(* The same rule as [int_opt_ptr], as an OCaml option rather than a C pointer.
   The generated ATen dispatch wants the pointer; [Op_bridge] and
   [Native_interp] want the value, and both go through this so decision 3 of
   op6-impl -- accept [Sym_int (Int n)], refuse [Sym_int (Name _)] -- has one
   implementation across the three decoders instead of three that happen to
   agree. *)
let int_opt_arg node name =
  let open Err.Syntax in
  match find_arg node name with
  | Some (Argument.Int i) -> Err.return (Some i)
  | Some (Argument.Sym_int s) ->
      let+ i = sym_int_value ~arg:name s in
      Some i
  | Some (Argument.None _) | None -> Err.return None
  | Some arg -> wrong_kind name `Int_opt arg

(* A schema Scalar arg crosses as either an Int or a Float argument. *)
let scalar_arg ?default node name =
  match find_arg node name with
  | Some (Argument.Int i) -> Err.return (Aten_scalar.Int (Int64.of_int i))
  | Some (Argument.Float f) -> Err.return (Aten_scalar.Float f)
  | None -> require name default
  | Some arg -> wrong_kind name `Scalar arg

let scalar_opt_arg node name =
  match find_arg node name with
  | Some (Argument.Int i) ->
      Err.return (Some (Aten_scalar.Int (Int64.of_int i)))
  | Some (Argument.Float f) -> Err.return (Some (Aten_scalar.Float f))
  | Some (Argument.None _) | None -> Err.return None
  | Some arg -> wrong_kind name `Scalar_opt arg

let bool_opt_arg node name =
  match find_arg node name with
  | Some (Argument.Bool b) -> Err.return (Some b)
  | Some (Argument.None _) | None -> Err.return None
  | Some arg -> wrong_kind name `Bool_opt arg

let string_arg ~default node name =
  match find_arg node name with
  | Some (Argument.String s) -> Err.return s
  | None -> Err.return default
  | Some arg -> wrong_kind name `String arg

(* The schema MemoryFormat enum crosses as Pytorch_types.MemoryFormat (the
   export schema's JSON encoding) but the op binding expects
   Aten_memory_format.t (c10's enum, used by the C shim). *)
let memory_format_opt_arg node name =
  match find_arg node name with
  | Some (Argument.Memory_format mf) -> (
      match mf with
      | MemoryFormat.ContiguousFormat ->
          Err.return (Some Aten_memory_format.Contiguous)
      | MemoryFormat.PreserveFormat ->
          Err.return (Some Aten_memory_format.Preserve)
      | MemoryFormat.ChannelsLast ->
          Err.return (Some Aten_memory_format.ChannelsLast)
      | MemoryFormat.ChannelsLast3d ->
          Err.return (Some Aten_memory_format.ChannelsLast3d)
      | MemoryFormat.Unknown -> Err.fail (`Unknown_memory_format name))
  | Some (Argument.None _) | None -> Err.return None
  | Some arg -> wrong_kind name `Memory_format_opt arg

(* ScalarType? / Layout? / Device? optionals. In exported inference graphs these
   are always omitted (the op's default None), so we decode the absent case only
   and raise loudly on an unexpected present value rather than guess the export
   enum -> c10 enum translation. *)
let scalar_type_opt_arg node name =
  match find_arg node name with
  | Some (Argument.None _) | None -> Err.return None
  | Some (Argument.Scalar_type _) ->
      Err.fail (`Unsupported_scalar_type_arg name)
  | Some arg -> wrong_kind name `Scalar_type_opt arg

let layout_opt_arg node name =
  match find_arg node name with
  | Some (Argument.None _) | None -> Err.return None
  | Some (Argument.Layout _) -> Err.fail (`Unsupported_layout_arg name)
  | Some arg -> wrong_kind name `Layout_opt arg

let device_opt_arg node name =
  match find_arg node name with
  | Some (Argument.None _) | None -> Err.return None
  | Some (Argument.Device _) -> Err.fail (`Unsupported_device_arg name)
  | Some arg -> wrong_kind name `Device_opt arg

let tensors_arg env node name =
  match find_arg node name with
  | Some (Argument.Tensors tas) ->
      Err.List.map (fun (ta : TensorArgument.t) -> resolve env ta.name) tas
  | Some arg -> wrong_kind name `Tensor_list arg
  | None -> Err.fail (`Missing_argument name)

(* A Tensor[] binding takes a (data, length) pair: build the ctypes array of
   handles for the [data] argument. *)
let tensor_carray tensors =
  CArray.start (CArray.of_list Aten_function_description.atc_tensor tensors)

(* --- output binding --- *)

let out_name (node : Node.t) i =
  match List.nth_opt node.outputs i with
  | Some (Argument.Tensor ta) -> Err.return (Some ta.TensorArgument.name)
  | Some (Argument.None _) -> Err.return None
  | Some arg ->
      Err.fail
        (`Unexpected_output_kind
           { Unexpected_output_kind.index = i; actual = argument_kind_name arg })
  | None -> Err.fail (`Missing_output i)

let bind1 (env : env) node tensor =
  let open Err.Syntax in
  let* name = out_name node 0 in
  match name with
  | Some name ->
      Err.return
        (String_map.add name
           (Aten_tensor.manage (Aten_tensor.check tensor))
           env)
  | None -> Err.return env

(* Every SSA name a node's outputs bind, flattening a Tensor[] output's names in
   place. [Argument.None] outputs stay ignored (a dead result the op computes but
   the graph does not name).

   Callers that filtered for [Argument.Tensor] alone saw ZERO names for a
   list-returning node -- which reads as "nothing to report" rather than as the
   omission it is. *)
let output_names (node : Node.t) : string list =
  List.concat_map
    (function
      | Argument.Tensor ta -> [ ta.TensorArgument.name ]
      | Argument.Tensors tas ->
          List.map (fun (ta : TensorArgument.t) -> ta.name) tas
      | _ -> [])
    node.outputs

(* Bind a Tensor[] result. The serialized shape differs from a fixed tuple's:
   ONE output of kind [Argument.Tensors] carrying every result name, rather than
   N separate [Argument.Tensor] outputs -- hence a distinct path rather than an
   overload of [bind_many], which indexes [node.outputs] positionally.

   [tensors] arrive already checked and managed by [Aten_tensor_list.to_list], so
   failing here cannot leak them: their finalisers are attached and the container
   is already freed.

   The arity check rides on [Err.List.map2], not a separate [List.length] test,
   so it cannot drift from the pairing it guards; map2 validates both lengths
   before the first mapping call, so a mismatch binds nothing at all. *)
let bind_tensor_list (env : env) (node : Node.t) tensors =
  let open Err.Syntax in
  match node.outputs with
  | [ Argument.Tensors tas ] ->
      let* pairs =
        Err.List.map2
          ~unequal_lengths:(fun names tensors ->
            `Tensor_list_output_arity_mismatch
              { Tensor_list_arity.names; tensors })
          (fun (ta : TensorArgument.t) t -> Err.return (ta.name, t))
          tas tensors
      in
      Err.return
        (List.fold_left
           (fun env (name, t) -> String_map.add name t env)
           env pairs)
  | outs ->
      Err.fail
        (`Expected_tensor_list_output
           { Tensor_list_output.actual = List.map argument_kind_name outs })

let bind_many (env : env) node tensors =
  let open Err.Syntax in
  let* _, env =
    Err.List.fold_left
      (fun (i, env) tensor ->
        let* name = out_name node i in
        let env =
          match name with
          | Some name -> String_map.add name (Aten_tensor.manage tensor) env
          | None -> env
        in
        Err.return (i + 1, env))
      (0, env) tensors
  in
  Err.return env

let resolve_result = resolve
let tensor_arg_result = tensor_arg
let tensor_or_scalar_arg_result = tensor_or_scalar_arg
let ints_arg_result = ints_arg
let int_arg_result = int_arg
let int_opt_ptr_result = int_opt_ptr
let int_opt_arg_result = int_opt_arg
let bool_arg_result = bool_arg
let float_arg_result = float_arg
let float_opt_ptr_result = float_opt_ptr
let scalar_arg_result = scalar_arg
let scalar_opt_arg_result = scalar_opt_arg
let bool_opt_arg_result = bool_opt_arg
let string_arg_result = string_arg
let memory_format_opt_arg_result = memory_format_opt_arg
let scalar_type_opt_arg_result = scalar_type_opt_arg
let layout_opt_arg_result = layout_opt_arg
let device_opt_arg_result = device_opt_arg
let tensors_arg_result = tensors_arg
