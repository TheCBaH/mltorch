(* Decode the `data.pkl` that torch.save writes for one tensor into a rebuild
   descriptor. The pickle stack machine is handled by the opickle library, which
   keeps GLOBAL / REDUCE / persistent-id nodes structurally; here we interpret
   torch's [_rebuild_tensor_v2(storage, storage_offset, size, stride,
   requires_grad, backward_hooks)] reduction, where [storage] is a persistent id
   referencing a typed storage in the archive's `data/<key>` entry. *)

module V = Opickle.Value

type rebuild = {
  storage_key : string;
  dtype : Pt2_dtype.t;
  storage_offset : Core.Storage_units.Element_offset.t;
  sizes : Pt2_tensor.Size.t list;
  strides : Pt2_tensor.Stride.t list;
}

type error =
  [ `Duplicate_tensor_map_key of string
  | `Expected_int of string
  | `Expected_tensor_map
  | `Malformed_pickle of string
  | `Negative_value of string * int64
  | `No_tensor_found
  | Pt2_dtype.error
  | `Tensor_map_key_not_string
  | `Tensor_map_value_not_tensor of string
  | `Unexpected_rebuild_args
  | `Unexpected_storage_persistent_id ]

let pp_error ppf : error -> unit = function
  | `Duplicate_tensor_map_key key ->
      Fmt.pf ppf "tensor map contains duplicate key %S" key
  | `Expected_int what -> Fmt.pf ppf "expected int for %s" what
  | `Expected_tensor_map -> Fmt.string ppf "pickle did not yield a tensor map"
  | `Malformed_pickle msg -> Fmt.pf ppf "pickle decode failed: %s" msg
  | `Negative_value (what, i) ->
      Fmt.pf ppf "%s is %Ld, which is negative" what i
  | `No_tensor_found -> Fmt.string ppf "pickle did not yield a tensor"
  | #Pt2_dtype.error as e -> Pt2_dtype.pp_error ppf e
  | `Tensor_map_key_not_string ->
      Fmt.string ppf "tensor map key is not a string"
  | `Tensor_map_value_not_tensor key ->
      Fmt.pf ppf "tensor map value for %S is not a tensor" key
  | `Unexpected_rebuild_args ->
      Fmt.string ppf "unexpected _rebuild_tensor_v2 arguments"
  | `Unexpected_storage_persistent_id ->
      Fmt.string ppf "unexpected storage persistent id"

(* The pickle's integers are signed 64-bit and stay [int64] here: a size or
   offset past this backend's [int] (32 bits under js_of_ocaml) is a legal
   value, and narrowing it is [Pt2_tensor.Cells]'s decision, made after a
   bound. A negative size or offset is [`Negative_value], distinct from
   [`Expected_int]: the value IS an integer, just outside the field's domain.
   See [[js_backends_design]]. *)
let int64_of what = function
  | V.Int i -> Err.return i
  | _ -> Err.fail (`Expected_int what)

let size_of what v =
  let open Err.Syntax in
  let* i = int64_of what v in
  Pt2_tensor.Size.of_int64 i

let sizes_of what a = Err.List.map (size_of what) (Array.to_list a)

let strides_of what a =
  Err.List.map
    (fun v -> Err.map (fun i -> Pt2_tensor.Stride.of_int64 i) (int64_of what v))
    (Array.to_list a)

let offset_of what v =
  let open Err.Syntax in
  let* i = int64_of what v in
  match Core.Storage_units.Element_offset.of_int64 i with
  | Ok o -> Err.return o
  | Error _ -> Err.fail (`Negative_value (what, i))

(* A storage persistent id is ('storage', <TypeStorage>, key, device, numel). *)
let storage_of_persid = function
  | V.Tuple [| V.Str "storage"; V.Global { name; _ }; V.Str key; _; _ |] ->
      let open Err.Syntax in
      let+ dtype =
        Pt2_dtype.of_storage_name name |> Err.map_error (fun e -> (e :> error))
      in
      (dtype, key)
  | _ -> Err.fail `Unexpected_storage_persistent_id

let rebuild_of_args = function
  | V.Tuple a when Array.length a >= 4 -> (
      match (a.(0), a.(1), a.(2), a.(3)) with
      | V.Persistent pid, offset, V.Tuple size, V.Tuple stride ->
          let open Err.Syntax in
          let* dtype, storage_key = storage_of_persid pid in
          let* storage_offset = offset_of "storage_offset" offset in
          let* sizes = sizes_of "size" size in
          let+ strides = strides_of "stride" stride in
          { storage_key; dtype; storage_offset; sizes; strides }
      | _ -> Err.fail `Unexpected_rebuild_args)
  | _ -> Err.fail `Unexpected_rebuild_args

let rec find_in_list = function
  | [] -> Err.return None
  | v :: vs -> (
      let open Err.Syntax in
      let* found = find_tensor v in
      match found with Some _ -> Err.return found | None -> find_in_list vs)

and find_in_dict = function
  | [] -> Err.return None
  | (k, v) :: rest -> (
      let open Err.Syntax in
      let* found = find_tensor k in
      match found with
      | Some _ -> Err.return found
      | None -> (
          let* found = find_tensor v in
          match found with
          | Some _ -> Err.return found
          | None -> find_in_dict rest))

(* Find the first _rebuild_tensor_v2 reduction anywhere in the value tree: a
   top-level tensor is wrapped by torch as ((tensor,), {}). *)
and find_tensor (v : V.t) =
  let open Err.Syntax in
  match v with
  | V.Reduce { func = V.Global { name = "_rebuild_tensor_v2"; _ }; args } ->
      let+ rb = rebuild_of_args args in
      Some rb
  | V.Tuple a -> find_in_list (Array.to_list a)
  | V.List r | V.Set r -> find_in_list !r
  | V.Frozenset l -> find_in_list l
  | V.Dict r -> find_in_dict !r
  | V.Reduce { func; args } -> (
      let* found = find_tensor func in
      match found with Some _ -> Err.return found | None -> find_tensor args)
  | V.Object { cls; args; state } -> (
      let* found = find_tensor cls in
      match found with
      | Some _ -> Err.return found
      | None -> (
          let* found = find_tensor args in
          match (found, state) with
          | Some _, _ -> Err.return found
          | None, None -> Err.return None
          | None, Some state -> find_tensor state))
  | V.Persistent id -> find_tensor id
  | _ -> Err.return None

let parse_tensor s =
  match Opickle.of_string s with
  | Error e -> Err.fail (`Malformed_pickle (Opickle.Error.to_string e))
  | Ok v ->
      let open Err.Syntax in
      let* rb = find_tensor v in
      rb |> Err.of_option `No_tensor_found

(* External release files are [torch.save] maps keyed by sample name.  This is
   deliberately not a general tree search: accepting a tensor hidden in an
   arbitrary pickle object makes a malformed oracle look valid and loses the
   association the caller must check. *)
let tensor_of_value key = function
  | V.Reduce { func = V.Global { name = "_rebuild_tensor_v2"; _ }; args } ->
      rebuild_of_args args
  | _ -> Err.fail (`Tensor_map_value_not_tensor key)

let parse_tensor_map s =
  match Opickle.of_string s with
  | Error e -> Err.fail (`Malformed_pickle (Opickle.Error.to_string e))
  | Ok (V.Dict entries) ->
      let seen = Hashtbl.create 16 in
      let open Err.Syntax in
      let* pairs =
        Err.List.map
          (fun (key, value) ->
            match key with
            | V.Str key ->
                if Hashtbl.mem seen key then
                  Err.fail (`Duplicate_tensor_map_key key)
                else begin
                  Hashtbl.add seen key ();
                  let+ tensor = tensor_of_value key value in
                  (key, tensor)
                end
            | _ -> Err.fail `Tensor_map_key_not_string)
          !entries
      in
      Err.return (List.sort (fun (a, _) (b, _) -> String.compare a b) pairs)
  | Ok _ -> Err.fail `Expected_tensor_map
