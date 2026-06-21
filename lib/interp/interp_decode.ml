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
  | Some tensor -> tensor
  | None -> failwith ("interp: undefined value " ^ name)

let tensor_arg env node name =
  match find_arg node name with
  | Some (Argument.Tensor ta) -> resolve env ta.TensorArgument.name
  | Some (Argument.None _) -> null_tensor
  | Some (Argument.Optional_tensor (OptionalTensorArgument.Tensor ta)) ->
      resolve env ta.TensorArgument.name
  | Some (Argument.Optional_tensor (OptionalTensorArgument.None _)) ->
      null_tensor
  | _ -> failwith ("interp: expected tensor arg " ^ name)

(* A binary op's Tensor-typed argument is normally an earlier node's result,
   but a compile-time scalar (e.g. the +3/relu6/.../6 in hardswish's
   decomposition) is serialized as a bare Int/Float rather than a lifted
   constant tensor; materialise it with full_like, broadcastable against
   [like]. *)
let tensor_or_scalar_arg env node name ~like =
  match find_arg node name with
  | Some (Argument.Tensor ta) -> resolve env ta.TensorArgument.name
  | Some (Argument.Int i) ->
      O.full_like like
        (Aten_scalar.Int (Int64.of_int i))
        None None None None None
  | Some (Argument.Float f) ->
      O.full_like like (Aten_scalar.Float f) None None None None None
  | _ -> failwith ("interp: expected tensor or scalar arg " ^ name)

let ints_arg ?(default = []) node name =
  match find_arg node name with
  | Some (Argument.Ints xs) -> xs
  | None -> default
  | _ -> failwith ("interp: expected int[] arg " ^ name)

let int_arg ?default node name =
  match find_arg node name with
  | Some (Argument.Int i) -> i
  | None -> (
      match default with
      | Some d -> d
      | None -> failwith ("interp: missing int " ^ name))
  | _ -> failwith ("interp: expected int arg " ^ name)

let bool_arg ?(default = false) node name =
  match find_arg node name with
  | Some (Argument.Bool b) -> b
  | None -> default
  | _ -> failwith ("interp: expected bool arg " ^ name)

let float_arg ?default node name =
  match find_arg node name with
  | Some (Argument.Float f) -> f
  | None -> (
      match default with
      | Some d -> d
      | None -> failwith ("interp: missing float " ^ name))
  | _ -> failwith ("interp: expected float arg " ^ name)

(* A schema Scalar arg crosses as either an Int or a Float argument. *)
let scalar_arg ?default node name =
  match find_arg node name with
  | Some (Argument.Int i) -> Aten_scalar.Int (Int64.of_int i)
  | Some (Argument.Float f) -> Aten_scalar.Float f
  | None -> (
      match default with
      | Some d -> d
      | None -> failwith ("interp: missing scalar " ^ name))
  | _ -> failwith ("interp: expected scalar arg " ^ name)

let scalar_opt_arg node name =
  match find_arg node name with
  | Some (Argument.Int i) -> Some (Aten_scalar.Int (Int64.of_int i))
  | Some (Argument.Float f) -> Some (Aten_scalar.Float f)
  | Some (Argument.None _) | None -> None
  | _ -> failwith ("interp: expected scalar arg " ^ name)

let bool_opt_arg node name =
  match find_arg node name with
  | Some (Argument.Bool b) -> Some b
  | Some (Argument.None _) | None -> None
  | _ -> failwith ("interp: expected bool arg " ^ name)

let string_arg ~default node name =
  match find_arg node name with
  | Some (Argument.String s) -> s
  | None -> default
  | _ -> failwith ("interp: expected string arg " ^ name)

(* The schema MemoryFormat enum crosses as Pytorch_types.MemoryFormat (the
   export schema's JSON encoding) but the op binding expects
   Aten_memory_format.t (c10's enum, used by the C shim). *)
let memory_format_opt_arg node name =
  match find_arg node name with
  | Some (Argument.Memory_format mf) -> (
      match mf with
      | MemoryFormat.ContiguousFormat -> Some Aten_memory_format.Contiguous
      | MemoryFormat.PreserveFormat -> Some Aten_memory_format.Preserve
      | MemoryFormat.ChannelsLast -> Some Aten_memory_format.ChannelsLast
      | MemoryFormat.ChannelsLast3d -> Some Aten_memory_format.ChannelsLast3d
      | MemoryFormat.Unknown ->
          failwith ("interp: unknown memory_format " ^ name))
  | Some (Argument.None _) | None -> None
  | _ -> failwith ("interp: expected memory_format arg " ^ name)

(* ScalarType? / Layout? / Device? optionals. In exported inference graphs these
   are always omitted (the op's default None), so we decode the absent case only
   and raise loudly on an unexpected present value rather than guess the export
   enum -> c10 enum translation. *)
let scalar_type_opt_arg node name =
  match find_arg node name with
  | Some (Argument.None _) | None -> None
  | _ -> failwith ("interp: unsupported scalar_type arg " ^ name)

let layout_opt_arg node name =
  match find_arg node name with
  | Some (Argument.None _) | None -> None
  | _ -> failwith ("interp: unsupported layout arg " ^ name)

let device_opt_arg node name =
  match find_arg node name with
  | Some (Argument.None _) | None -> None
  | _ -> failwith ("interp: unsupported device arg " ^ name)

let tensors_arg env node name =
  match find_arg node name with
  | Some (Argument.Tensors tas) ->
      List.map (fun (ta : TensorArgument.t) -> resolve env ta.name) tas
  | _ -> failwith ("interp: expected tensor list arg " ^ name)

(* A Tensor[] binding takes a (data, length) pair: build the ctypes array of
   handles for the [data] argument. *)
let tensor_carray tensors =
  CArray.start (CArray.of_list Aten_function_description.atc_tensor tensors)

(* --- output binding --- *)

let out_name (node : Node.t) i =
  match List.nth node.outputs i with
  | Argument.Tensor ta -> Some ta.TensorArgument.name
  | Argument.None _ -> None
  | _ -> failwith "interp: unexpected output kind"

let bind1 (env : env) node tensor =
  match out_name node 0 with
  | Some name ->
      String_map.add name (Aten_tensor.manage (Aten_tensor.check tensor)) env
  | None -> env

let bind_many (env : env) node tensors =
  List.fold_left
    (fun (i, env) tensor ->
      let env =
        match out_name node i with
        | Some name -> String_map.add name (Aten_tensor.manage tensor) env
        | None -> env
      in
      (i + 1, env))
    (0, env) tensors
  |> snd
