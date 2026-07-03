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

type error =
  [ `Undefined_value of string
  | `Missing_argument of string
  | `Wrong_argument_kind of string * string * string
  | `Unsupported_scalar_type_arg of string
  | `Unsupported_layout_arg of string
  | `Unsupported_device_arg of string
  | `Unknown_memory_format of string
  | `Unexpected_output_kind of int * string ]

let pp_error ppf : [< error ] -> unit = function
  | `Undefined_value name -> Fmt.pf ppf "undefined SSA value %S" name
  | `Missing_argument name -> Fmt.pf ppf "missing required argument %S" name
  | `Wrong_argument_kind (name, expected, actual) ->
      Fmt.pf ppf "argument %S: expected %s, got %s" name expected actual
  | `Unsupported_scalar_type_arg name ->
      Fmt.pf ppf "unsupported scalar_type argument %S" name
  | `Unsupported_layout_arg name ->
      Fmt.pf ppf "unsupported layout argument %S" name
  | `Unsupported_device_arg name ->
      Fmt.pf ppf "unsupported device argument %S" name
  | `Unknown_memory_format name ->
      Fmt.pf ppf "unknown memory_format for argument %S" name
  | `Unexpected_output_kind (i, actual) ->
      Fmt.pf ppf "output %d: expected tensor/None, got %s" i actual

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
  match String_map.find_opt name env with
  | Some tensor -> Core.return tensor
  | None -> Core.fail (`Undefined_value name)

let wrong_kind name expected arg =
  Core.fail (`Wrong_argument_kind (name, expected, argument_kind_name arg))

let tensor_arg env node name =
  match find_arg node name with
  | Some (Argument.Tensor ta) -> resolve env ta.TensorArgument.name
  | Some (Argument.None _) -> Core.return null_tensor
  | Some (Argument.Optional_tensor (OptionalTensorArgument.Tensor ta)) ->
      resolve env ta.TensorArgument.name
  | Some (Argument.Optional_tensor (OptionalTensorArgument.None _)) ->
      Core.return null_tensor
  | Some arg -> wrong_kind name "tensor?" arg
  | None -> Core.fail (`Missing_argument name)

(* A binary op's Tensor-typed argument is normally an earlier node's result,
   but a compile-time scalar (e.g. the +3/relu6/.../6 in hardswish's
   decomposition) is serialized as a bare Int/Float rather than a lifted
   constant tensor; materialise it with full_like, broadcastable against
   [like]. *)
let tensor_or_scalar_arg env node name ~like =
  match find_arg node name with
  | Some (Argument.Tensor ta) -> resolve env ta.TensorArgument.name
  | Some (Argument.Int i) ->
      Core.return
        (O.full_like like
           (Aten_scalar.Int (Int64.of_int i))
           None None None None None)
  | Some (Argument.Float f) ->
      Core.return
        (O.full_like like (Aten_scalar.Float f) None None None None None)
  | Some arg -> wrong_kind name "tensor|scalar" arg
  | None -> Core.fail (`Missing_argument name)

let ints_arg ?(default = []) node name =
  match find_arg node name with
  | Some (Argument.Ints xs) -> Core.return xs
  | Some (Argument.None _) | None -> Core.return default
  | Some arg -> wrong_kind name "int[]" arg

let require name = function
  | Some d -> Core.return d
  | None -> Core.fail (`Missing_argument name)

let int_arg ?default node name =
  match find_arg node name with
  | Some (Argument.Int i) -> Core.return i
  | None -> require name default
  | Some arg -> wrong_kind name "int" arg

let bool_arg ?(default = false) node name =
  match find_arg node name with
  | Some (Argument.Bool b) -> Core.return b
  | None -> Core.return default
  | Some arg -> wrong_kind name "bool" arg

let float_arg ?default node name =
  match find_arg node name with
  | Some (Argument.Float f) -> Core.return f
  | None -> require name default
  | Some arg -> wrong_kind name "float" arg

(* A float? argument crosses as a [double ptr] (map_type lowers float? to a
   [double] pointer): a null pointer encodes None (the C shim maps null to
   std::nullopt), a live cell encodes the present value. The pointer form mirrors
   int?/SymInt?; it is the float counterpart the generator emits for e.g.
   rms_norm's eps. *)
let float_opt_ptr node name =
  match find_arg node name with
  | Some (Argument.Float f) -> Core.return (allocate double f)
  | Some (Argument.None _) | None -> Core.return (from_voidp double null)
  | Some arg -> wrong_kind name "float?" arg

(* A schema Scalar arg crosses as either an Int or a Float argument. *)
let scalar_arg ?default node name =
  match find_arg node name with
  | Some (Argument.Int i) -> Core.return (Aten_scalar.Int (Int64.of_int i))
  | Some (Argument.Float f) -> Core.return (Aten_scalar.Float f)
  | None -> require name default
  | Some arg -> wrong_kind name "scalar" arg

let scalar_opt_arg node name =
  match find_arg node name with
  | Some (Argument.Int i) ->
      Core.return (Some (Aten_scalar.Int (Int64.of_int i)))
  | Some (Argument.Float f) -> Core.return (Some (Aten_scalar.Float f))
  | Some (Argument.None _) | None -> Core.return None
  | Some arg -> wrong_kind name "scalar?" arg

let bool_opt_arg node name =
  match find_arg node name with
  | Some (Argument.Bool b) -> Core.return (Some b)
  | Some (Argument.None _) | None -> Core.return None
  | Some arg -> wrong_kind name "bool?" arg

let string_arg ~default node name =
  match find_arg node name with
  | Some (Argument.String s) -> Core.return s
  | None -> Core.return default
  | Some arg -> wrong_kind name "string" arg

(* The schema MemoryFormat enum crosses as Pytorch_types.MemoryFormat (the
   export schema's JSON encoding) but the op binding expects
   Aten_memory_format.t (c10's enum, used by the C shim). *)
let memory_format_opt_arg node name =
  match find_arg node name with
  | Some (Argument.Memory_format mf) -> (
      match mf with
      | MemoryFormat.ContiguousFormat ->
          Core.return (Some Aten_memory_format.Contiguous)
      | MemoryFormat.PreserveFormat ->
          Core.return (Some Aten_memory_format.Preserve)
      | MemoryFormat.ChannelsLast ->
          Core.return (Some Aten_memory_format.ChannelsLast)
      | MemoryFormat.ChannelsLast3d ->
          Core.return (Some Aten_memory_format.ChannelsLast3d)
      | MemoryFormat.Unknown -> Core.fail (`Unknown_memory_format name))
  | Some (Argument.None _) | None -> Core.return None
  | Some arg -> wrong_kind name "memory_format?" arg

(* ScalarType? / Layout? / Device? optionals. In exported inference graphs these
   are always omitted (the op's default None), so we decode the absent case only
   and raise loudly on an unexpected present value rather than guess the export
   enum -> c10 enum translation. *)
let scalar_type_opt_arg node name =
  match find_arg node name with
  | Some (Argument.None _) | None -> Core.return None
  | Some (Argument.Scalar_type _) ->
      Core.fail (`Unsupported_scalar_type_arg name)
  | Some arg -> wrong_kind name "scalar_type?" arg

let layout_opt_arg node name =
  match find_arg node name with
  | Some (Argument.None _) | None -> Core.return None
  | Some (Argument.Layout _) -> Core.fail (`Unsupported_layout_arg name)
  | Some arg -> wrong_kind name "layout?" arg

let device_opt_arg node name =
  match find_arg node name with
  | Some (Argument.None _) | None -> Core.return None
  | Some (Argument.Device _) -> Core.fail (`Unsupported_device_arg name)
  | Some arg -> wrong_kind name "device?" arg

let tensors_arg env node name =
  match find_arg node name with
  | Some (Argument.Tensors tas) ->
      Core.List.map (fun (ta : TensorArgument.t) -> resolve env ta.name) tas
  | Some arg -> wrong_kind name "tensor[]" arg
  | None -> Core.fail (`Missing_argument name)

(* A Tensor[] binding takes a (data, length) pair: build the ctypes array of
   handles for the [data] argument. *)
let tensor_carray tensors =
  CArray.start (CArray.of_list Aten_function_description.atc_tensor tensors)

(* --- output binding --- *)

let out_name (node : Node.t) i =
  match List.nth_opt node.outputs i with
  | Some (Argument.Tensor ta) -> Core.return (Some ta.TensorArgument.name)
  | Some (Argument.None _) -> Core.return None
  | Some arg -> Core.fail (`Unexpected_output_kind (i, argument_kind_name arg))
  | None -> Core.fail (`Unexpected_output_kind (i, "missing"))

let bind1 (env : env) node tensor =
  let open Core.Syntax in
  let* name = out_name node 0 in
  match name with
  | Some name ->
      Core.return
        (String_map.add name
           (Aten_tensor.manage (Aten_tensor.check tensor))
           env)
  | None -> Core.return env

let bind_many (env : env) node tensors =
  let open Core.Syntax in
  let* _, env =
    Core.List.fold_left
      (fun (i, env) tensor ->
        let* name = out_name node i in
        let env =
          match name with
          | Some name -> String_map.add name (Aten_tensor.manage tensor) env
          | None -> env
        in
        Core.return (i + 1, env))
      (0, env) tensors
  in
  Core.return env

let resolve_result = resolve
let tensor_arg_result = tensor_arg
let tensor_or_scalar_arg_result = tensor_or_scalar_arg
let ints_arg_result = ints_arg
let int_arg_result = int_arg
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
