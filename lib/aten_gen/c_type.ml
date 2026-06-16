(* Type mapping: ATen schema types (Func_ast) -> C-ABI representation.

   This is the gated core of the binding generator. Only a controlled set of
   types is supported; an unsupported type makes [map] return [None] and the
   whole op is skipped (handled manually / later). Conventions mirror
   janestreet/torch: tensors cross the boundary as opaque [void*] handles
   (a heap [at::Tensor*]), scalars/ints/floats by value, lists as
   (pointer, length) pairs. *)

(* The C representation of one schema argument. A single schema arg can expand
   to several C parameters (e.g. a list -> data ptr + length). *)
type arg = {
  c_params : string list; (* C parameter declarations *)
  call_expr : string; (* C++ expression passed to the at:: call *)
  ctypes : string list; (* ctypes type fragments, one per c_param *)
}

(* The C representation of the return. *)
type ret = Tensor_ret (* single Tensor -> owning void* handle *)

let unsupported = None

(* Map an argument [name] of schema type [ty] to its C representation, or
   [None] if any constituent type is outside the supported set. *)
let map_type ~name (ty : Func_ast.Type.t) =
  match ty with
  | Base Tensor ->
      Some
        {
          c_params = [ Printf.sprintf "atc_tensor %s" name ];
          call_expr = Printf.sprintf "*atc_to_ptr(%s)" name;
          ctypes = [ "atc_tensor" ];
        }
  (* SymInt is the symbolic-shape int; calling the non-_symint [at::<op>]
     frontend overload accepts a plain int64_t, so we bind it identically to
     a concrete int. (Symbolic values are never produced on this CPU path.) *)
  | Base Int | Base SymInt ->
      Some
        {
          c_params = [ Printf.sprintf "int64_t %s" name ];
          call_expr = name;
          ctypes = [ "int64_t" ];
        }
  | Base Float ->
      Some
        {
          c_params = [ Printf.sprintf "double %s" name ];
          call_expr = name;
          ctypes = [ "double" ];
        }
  | Base Bool ->
      Some
        {
          c_params = [ Printf.sprintf "int %s" name ];
          call_expr = Printf.sprintf "(bool)%s" name;
          ctypes = [ "bool" ];
        }
  | Base Scalar ->
      (* c10::Scalar is a tagged union; it crosses as a pointer to a
         [struct atc_scalar] (see atg_shim.h) so the int/float category — which
         drives ATen type promotion — survives. (By pointer, not by value: the
         ctypes stub generator re-emits a conflicting definition for a by-value
         struct arg.) [scalar] is the ctypes view over the typed [Scalar.t]. *)
      Some
        {
          c_params = [ Printf.sprintf "const struct atc_scalar* %s" name ];
          call_expr = Printf.sprintf "atc_to_c10_scalar(%s)" name;
          ctypes = [ "scalar" ];
        }
  | Base ScalarType ->
      (* the c10 enum crosses as its int code; [scalar_type] is a ctypes view
         that translates the OCaml [Scalar_type.t] enum (see emit.ml). *)
      Some
        {
          c_params = [ Printf.sprintf "int %s" name ];
          call_expr = Printf.sprintf "static_cast<at::ScalarType>(%s)" name;
          ctypes = [ "scalar_type" ];
        }
  | Base MemoryFormat ->
      (* c10 enum int code; [memory_format] is the ctypes view over
         [Memory_format.t] (e.g. contiguous's memory_format arg). *)
      Some
        {
          c_params = [ Printf.sprintf "int %s" name ];
          call_expr = Printf.sprintf "static_cast<at::MemoryFormat>(%s)" name;
          ctypes = [ "memory_format" ];
        }
  | Optional (Base Tensor) ->
      (* null handle (0) -> nullopt *)
      Some
        {
          c_params = [ Printf.sprintf "atc_tensor %s" name ];
          call_expr =
            Printf.sprintf
              "%s ? std::make_optional(*atc_to_ptr(%s)) : std::nullopt" name
              name;
          ctypes = [ "atc_tensor" ];
        }
  (* Scalar?: same [struct atc_scalar] pointer as [Base Scalar], with the NONE
     tag encoding None (-> nullopt). [scalar_opt] is the ctypes view over
     [Scalar.t option]. *)
  | Optional (Base Scalar) ->
      Some
        {
          c_params = [ Printf.sprintf "const struct atc_scalar* %s" name ];
          call_expr = Printf.sprintf "atc_to_c10_scalar_opt(%s)" name;
          ctypes = [ "scalar_opt" ];
        }
  (* int? / SymInt?: pass a pointer; null -> nullopt, else the pointee. The
     non-_symint [at::<op>] overload takes std::optional<int64_t>, so SymInt?
     binds identically to int? (mirrors the Int/SymInt scalar case). *)
  | Optional (Base Int) | Optional (Base SymInt) ->
      Some
        {
          c_params = [ Printf.sprintf "int64_t* %s" name ];
          call_expr =
            Printf.sprintf "%s ? std::make_optional(*%s) : std::nullopt" name
              name;
          ctypes = [ "ptr int64_t" ];
        }
  (* ScalarType?: the enum int code with a negative sentinel for None (-> the
     [at::<op>] std::optional<ScalarType> param, e.g. mean.dim's dtype kwarg).
     [scalar_type_opt] is the matching ctypes view over [Scalar_type.t option]. *)
  | Optional (Base ScalarType) ->
      Some
        {
          c_params = [ Printf.sprintf "int %s" name ];
          call_expr =
            Printf.sprintf
              "%s < 0 ? std::nullopt : \
               std::make_optional(static_cast<at::ScalarType>(%s))"
              name name;
          ctypes = [ "scalar_type_opt" ];
        }
  (* Layout?: like ScalarType?, the enum int code with a negative sentinel for
     None. [layout_opt] is the matching ctypes view over [Layout.t option]. *)
  | Optional (Base Layout) ->
      Some
        {
          c_params = [ Printf.sprintf "int %s" name ];
          call_expr =
            Printf.sprintf
              "%s < 0 ? std::nullopt : \
               std::make_optional(static_cast<at::Layout>(%s))"
              name name;
          ctypes = [ "layout_opt" ];
        }
  (* MemoryFormat?: like ScalarType?, the enum int code with a negative sentinel
     for None (e.g. clone's memory_format kwarg). [memory_format_opt] is the
     matching ctypes view over [Memory_format.t option]. *)
  | Optional (Base MemoryFormat) ->
      Some
        {
          c_params = [ Printf.sprintf "int %s" name ];
          call_expr =
            Printf.sprintf
              "%s < 0 ? std::nullopt : \
               std::make_optional(static_cast<at::MemoryFormat>(%s))"
              name name;
          ctypes = [ "memory_format_opt" ];
        }
  (* Int[] and SymInt[] both bind to the non-_symint [at::<op>] overload, which
     takes an at::IntArrayRef; pass it as a (data, length) pair. *)
  | List (Base Int, _) | List (Base SymInt, _) ->
      Some
        {
          c_params =
            [
              Printf.sprintf "int64_t* %s_data" name;
              Printf.sprintf "int %s_len" name;
            ];
          call_expr =
            Printf.sprintf "at::IntArrayRef(%s_data, %s_len)" name name;
          ctypes = [ "ptr int64_t"; "int" ];
        }
  (* int[]? / SymInt[]? (incl. sized, e.g. int[1]?): an optional IntArrayRef.
     null data ptr -> nullopt, else a (data, length) view. The non-_symint
     [at::<op>] overload takes at::OptionalIntArrayRef (e.g. mean.dim's dim). *)
  | Optional (List (Base Int, _)) | Optional (List (Base SymInt, _)) ->
      Some
        {
          c_params =
            [
              Printf.sprintf "int64_t* %s_data" name;
              Printf.sprintf "int %s_len" name;
            ];
          call_expr =
            Printf.sprintf
              "%s_data ? at::OptionalIntArrayRef(at::IntArrayRef(%s_data, \
               %s_len)) : at::OptionalIntArrayRef(std::nullopt)"
              name name name;
          ctypes = [ "ptr int64_t"; "int" ];
        }
  (* NB: any c10 enum added here (e.g. QScheme, or DeviceType inside Device)
     must cross as a ctypes [view] over its OCaml enum, like ScalarType / Layout
     / MemoryFormat above — never a bare int, which would leak an untyped code to
     callers. QScheme is omitted on purpose: it never appears as a schema arg. *)
  | Base _ | Optional _ | List _ -> unsupported

(* The supported return shapes. Initially: exactly one Tensor. *)
let map_returns (returns : Func_ast.Return.t list) =
  match returns with
  | [ { ty = Base Tensor; _ } ] -> Some Tensor_ret
  | _ -> unsupported
