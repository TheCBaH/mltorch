(* Raw (un-interpreted) representation of a native_functions.yaml entry.
   String-valued fields that are enums are decoded into proper OCaml types;
   the func signature string is NOT parsed further. *)

(* ---- enum modules ---- *)

module Variant = struct
  type t = Function | Method

  let of_string = function
    | "function" -> Some Function
    | "method" -> Some Method
    | _ -> None

  let to_string = function Function -> "function" | Method -> "method"
  let pp fmt v = Format.pp_print_string fmt (to_string v)
end

module Backend = struct
  type t =
    | CompositeExplicitAutograd
    | CompositeExplicitAutogradNonFunctional
    | CompositeImplicitAutograd
    | CompositeImplicitAutogradNestedTensor
    | CPU
    | CUDA
    | Generic
    | Meta
    | MkldnnCPU
    | MPS
    | MTIA
    | NestedTensorCPU
    | NestedTensorCUDA
    | NestedTensorHPU
    | NestedTensorMeta
    | QuantizedCPU
    | QuantizedCUDA
    | QuantizedMeta
    | ScalarOnly
    | SparseCPU
    | SparseCsrCPU
    | SparseCsrCUDA
    | SparseCsrMPS
    | SparseCsrMeta
    | SparseCUDA
    | SparseMPS
    | SparseMeta
    | XPU
    | ZeroTensor

  let of_string = function
    | "CompositeExplicitAutograd" -> Some CompositeExplicitAutograd
    | "CompositeExplicitAutogradNonFunctional" ->
        Some CompositeExplicitAutogradNonFunctional
    | "CompositeImplicitAutograd" -> Some CompositeImplicitAutograd
    | "CompositeImplicitAutogradNestedTensor" ->
        Some CompositeImplicitAutogradNestedTensor
    | "CPU" -> Some CPU
    | "CUDA" -> Some CUDA
    | "Generic" -> Some Generic
    | "Meta" -> Some Meta
    | "MkldnnCPU" -> Some MkldnnCPU
    | "MPS" -> Some MPS
    | "MTIA" -> Some MTIA
    | "NestedTensorCPU" -> Some NestedTensorCPU
    | "NestedTensorCUDA" -> Some NestedTensorCUDA
    | "NestedTensorHPU" -> Some NestedTensorHPU
    | "NestedTensorMeta" -> Some NestedTensorMeta
    | "QuantizedCPU" -> Some QuantizedCPU
    | "QuantizedCUDA" -> Some QuantizedCUDA
    | "QuantizedMeta" -> Some QuantizedMeta
    | "ScalarOnly" -> Some ScalarOnly
    | "SparseCPU" -> Some SparseCPU
    | "SparseCsrCPU" -> Some SparseCsrCPU
    | "SparseCsrCUDA" -> Some SparseCsrCUDA
    | "SparseCsrMPS" -> Some SparseCsrMPS
    | "SparseCsrMeta" -> Some SparseCsrMeta
    | "SparseCUDA" -> Some SparseCUDA
    | "SparseMPS" -> Some SparseMPS
    | "SparseMeta" -> Some SparseMeta
    | "XPU" -> Some XPU
    | "ZeroTensor" -> Some ZeroTensor
    | _ -> None

  let to_string = function
    | CompositeExplicitAutograd -> "CompositeExplicitAutograd"
    | CompositeExplicitAutogradNonFunctional ->
        "CompositeExplicitAutogradNonFunctional"
    | CompositeImplicitAutograd -> "CompositeImplicitAutograd"
    | CompositeImplicitAutogradNestedTensor ->
        "CompositeImplicitAutogradNestedTensor"
    | CPU -> "CPU"
    | CUDA -> "CUDA"
    | Generic -> "Generic"
    | Meta -> "Meta"
    | MkldnnCPU -> "MkldnnCPU"
    | MPS -> "MPS"
    | MTIA -> "MTIA"
    | NestedTensorCPU -> "NestedTensorCPU"
    | NestedTensorCUDA -> "NestedTensorCUDA"
    | NestedTensorHPU -> "NestedTensorHPU"
    | NestedTensorMeta -> "NestedTensorMeta"
    | QuantizedCPU -> "QuantizedCPU"
    | QuantizedCUDA -> "QuantizedCUDA"
    | QuantizedMeta -> "QuantizedMeta"
    | ScalarOnly -> "ScalarOnly"
    | SparseCPU -> "SparseCPU"
    | SparseCsrCPU -> "SparseCsrCPU"
    | SparseCsrCUDA -> "SparseCsrCUDA"
    | SparseCsrMPS -> "SparseCsrMPS"
    | SparseCsrMeta -> "SparseCsrMeta"
    | SparseCUDA -> "SparseCUDA"
    | SparseMPS -> "SparseMPS"
    | SparseMeta -> "SparseMeta"
    | XPU -> "XPU"
    | ZeroTensor -> "ZeroTensor"

  let pp fmt b = Format.pp_print_string fmt (to_string b)
end

module Device_check = struct
  (* torchgen DeviceCheckType; default when absent is ExactSame *)
  type t = ExactSame | NoCheck

  let of_string = function
    | "ExactSame" -> Some ExactSame
    | "NoCheck" -> Some NoCheck
    | _ -> None

  let to_string = function ExactSame -> "ExactSame" | NoCheck -> "NoCheck"
  let pp fmt d = Format.pp_print_string fmt (to_string d)
end

module Tag = struct
  (* All valid tags from aten/src/ATen/native/tags.yaml *)
  type t =
    | Core
    | Cudagraph_unsafe
    | Data_dependent_output
    | Dynamic_output_shape
    | Flexible_layout
    | Generated
    | Inplace_view
    | Maybe_aliasing_or_mutating
    | Needs_contiguous_strides
    | Needs_exact_strides
    | Needs_fixed_stride_order
    | Nondeterministic_bitwise
    | Nondeterministic_seeded
    | Pointwise
    | Pt2_compliant_tag
    | Reduction
    | View_copy

  let of_string = function
    | "core" -> Some Core
    | "cudagraph_unsafe" -> Some Cudagraph_unsafe
    | "data_dependent_output" -> Some Data_dependent_output
    | "dynamic_output_shape" -> Some Dynamic_output_shape
    | "flexible_layout" -> Some Flexible_layout
    | "generated" -> Some Generated
    | "inplace_view" -> Some Inplace_view
    | "maybe_aliasing_or_mutating" -> Some Maybe_aliasing_or_mutating
    | "needs_contiguous_strides" -> Some Needs_contiguous_strides
    | "needs_exact_strides" -> Some Needs_exact_strides
    | "needs_fixed_stride_order" -> Some Needs_fixed_stride_order
    | "nondeterministic_bitwise" -> Some Nondeterministic_bitwise
    | "nondeterministic_seeded" -> Some Nondeterministic_seeded
    | "pointwise" -> Some Pointwise
    | "pt2_compliant_tag" -> Some Pt2_compliant_tag
    | "reduction" -> Some Reduction
    | "view_copy" -> Some View_copy
    | _ -> None

  let to_string = function
    | Core -> "core"
    | Cudagraph_unsafe -> "cudagraph_unsafe"
    | Data_dependent_output -> "data_dependent_output"
    | Dynamic_output_shape -> "dynamic_output_shape"
    | Flexible_layout -> "flexible_layout"
    | Generated -> "generated"
    | Inplace_view -> "inplace_view"
    | Maybe_aliasing_or_mutating -> "maybe_aliasing_or_mutating"
    | Needs_contiguous_strides -> "needs_contiguous_strides"
    | Needs_exact_strides -> "needs_exact_strides"
    | Needs_fixed_stride_order -> "needs_fixed_stride_order"
    | Nondeterministic_bitwise -> "nondeterministic_bitwise"
    | Nondeterministic_seeded -> "nondeterministic_seeded"
    | Pointwise -> "pointwise"
    | Pt2_compliant_tag -> "pt2_compliant_tag"
    | Reduction -> "reduction"
    | View_copy -> "view_copy"

  let pp fmt t = Format.pp_print_string fmt (to_string t)
end

module Category = struct
  type t = Dummy | Factory

  let of_string = function
    | "dummy" -> Some Dummy
    | "factory" -> Some Factory
    | _ -> None

  let to_string = function Dummy -> "dummy" | Factory -> "factory"
  let pp fmt c = Format.pp_print_string fmt (to_string c)
end

(* ---- record types ---- *)

type dispatch_entry = {
  backends : Backend.t list; (* one or more comma-separated backends *)
  kernel : string;
}

type t = {
  func : string;
  variants : Variant.t list;
  dispatch : dispatch_entry list;
  structured : bool;
  structured_delegate : string option;
  structured_inherits : string option;
  tags : Tag.t list;
  device_guard : bool;
  device_check : Device_check.t option;
  python_module : string option;
  manual_cpp_binding : bool;
  use_const_ref_for_mutable_tensors : bool;
  category_override : Category.t option;
  cpp_no_default_args : string list;
  autogen : string option; (* comma-separated op names, kept raw *)
  precomputed : string list;
  ufunc_inner_loop : (Backend.t * string) list;
}

(* ---- jsont combinators ---- *)

(* Decode a JSON string to an enum value using the given of_string parser. *)
let string_enum ~kind of_string =
  Jsont.Base.string
    (Jsont.Base.map ~kind
       ~dec:(fun m s ->
         match of_string (String.trim s) with
         | Some v -> v
         | None -> Jsont.Error.msgf m "unknown %s: %S" kind s)
       ())

(* variants: comma-separated "function"/"method" string → Variant.t list *)
let variants_jsont : Variant.t list Jsont.t =
  Jsont.Base.string
    (Jsont.Base.map ~kind:"Variants"
       ~dec:(fun m s ->
         String.split_on_char ',' s
         |> List.filter_map (fun p ->
             let p = String.trim p in
             if p = "" then None
             else
               match Variant.of_string p with
               | Some v -> Some v
               | None -> Jsont.Error.msgf m "unknown variant: %S" p))
       ())

(* comma-separated backend string → Backend.t list (used in dispatch keys) *)
let parse_backends m s =
  String.split_on_char ',' s
  |> List.filter_map (fun p ->
      let p = String.trim p in
      if p = "" then None
      else
        match Backend.of_string p with
        | Some b -> Some b
        | None -> Jsont.Error.msgf m "unknown backend: %S" p)

(* dispatch: object with "CPU, CUDA": kernel_fn entries *)
let dispatch_jsont : dispatch_entry list Jsont.t =
  Jsont.map ~dec:List.rev
    (Jsont.fold_object Jsont.string
       (fun meta keys_str kernel acc ->
         { backends = parse_backends meta keys_str; kernel } :: acc)
       [])

(* tags: scalar string → singleton list;  YAML sequence → list *)
let tags_jsont : Tag.t list Jsont.t =
  let tag_jsont = string_enum ~kind:"Tag" Tag.of_string in
  Jsont.any ()
    ~dec_string:(Jsont.map ~dec:(fun t -> [ t ]) tag_jsont)
    ~dec_array:(Jsont.list tag_jsont)

(* ufunc_inner_loop: object with backend-name keys *)
let ufunc_inner_loop_jsont : (Backend.t * string) list Jsont.t =
  Jsont.map ~dec:List.rev
    (Jsont.fold_object Jsont.string
       (fun meta k v acc ->
         match Backend.of_string k with
         | Some b -> (b, v) :: acc
         | None -> Jsont.Error.msgf meta "unknown ufunc_inner_loop key: %S" k)
       [])

let entry_jsont : t Jsont.t =
  Jsont.Object.(
    finish
      (map
         (fun
           func
           variants
           dispatch
           structured
           structured_delegate
           structured_inherits
           tags
           device_guard
           device_check
           python_module
           manual_cpp_binding
           use_const_ref_for_mutable_tensors
           category_override
           cpp_no_default_args
           autogen
           precomputed
           ufunc_inner_loop
         ->
           {
             func;
             variants;
             dispatch;
             structured;
             structured_delegate;
             structured_inherits;
             tags;
             device_guard;
             device_check;
             python_module;
             manual_cpp_binding;
             use_const_ref_for_mutable_tensors;
             category_override;
             cpp_no_default_args;
             autogen;
             precomputed;
             ufunc_inner_loop;
           })
      |> mem "func" Jsont.string
      |> mem "variants" variants_jsont ~dec_absent:[]
      |> mem "dispatch" dispatch_jsont ~dec_absent:[]
      |> mem "structured" Jsont.bool ~dec_absent:false
      |> opt_mem "structured_delegate" Jsont.string
      |> opt_mem "structured_inherits" Jsont.string
      |> mem "tags" tags_jsont ~dec_absent:[]
      |> mem "device_guard" Jsont.bool ~dec_absent:true
      |> opt_mem "device_check"
           (string_enum ~kind:"DeviceCheck" Device_check.of_string)
      |> opt_mem "python_module" Jsont.string
      |> mem "manual_cpp_binding" Jsont.bool ~dec_absent:false
      |> mem "use_const_ref_for_mutable_tensors" Jsont.bool ~dec_absent:false
      |> opt_mem "category_override"
           (string_enum ~kind:"Category" Category.of_string)
      |> mem "cpp_no_default_args" (Jsont.list Jsont.string) ~dec_absent:[]
      |> opt_mem "autogen" Jsont.string
      |> mem "precomputed" (Jsont.list Jsont.string) ~dec_absent:[]
      |> mem "ufunc_inner_loop" ufunc_inner_loop_jsont ~dec_absent:[]))

let jsont : t list Jsont.t = Jsont.list entry_jsont
let of_yaml_string s = Yamlt.of_string jsont s
