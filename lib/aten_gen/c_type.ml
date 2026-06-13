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
      (* Simplification: float-valued scalar passed as a double. Sufficient for
         the initial gated op set (e.g. add's alpha); revisit for int scalars. *)
      Some
        {
          c_params = [ Printf.sprintf "double %s" name ];
          call_expr = Printf.sprintf "c10::Scalar(%s)" name;
          ctypes = [ "double" ];
        }
  | Base ScalarType ->
      Some
        {
          c_params = [ Printf.sprintf "int %s" name ];
          call_expr = Printf.sprintf "static_cast<at::ScalarType>(%s)" name;
          ctypes = [ "int" ];
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
  | Base _ | Optional _ | List _ -> unsupported

(* The supported return shapes. Initially: exactly one Tensor. *)
let map_returns (returns : Func_ast.Return.t list) =
  match returns with
  | [ { ty = Base Tensor; _ } ] -> Some Tensor_ret
  | _ -> unsupported
