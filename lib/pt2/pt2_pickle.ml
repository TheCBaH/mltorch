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
let int64_of what v =
  match V.view v with
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
let storage_of_persid pid =
  match V.view pid with
  | V.Tuple [| tag; cls; key; _; _ |] -> (
      match (V.view tag, V.view cls, V.view key) with
      | V.Str "storage", V.Global { name; _ }, V.Str key ->
          let open Err.Syntax in
          let+ dtype =
            Pt2_dtype.of_storage_name name
            |> Err.map_error (fun e -> (e :> error))
          in
          (dtype, key)
      | _ -> Err.fail `Unexpected_storage_persistent_id)
  | _ -> Err.fail `Unexpected_storage_persistent_id

let rebuild_of_args args =
  match V.view args with
  | V.Tuple a when Array.length a >= 4 -> (
      match (V.view a.(0), a.(1), V.view a.(2), V.view a.(3)) with
      | V.Persistent pid, offset, V.Tuple size, V.Tuple stride ->
          let open Err.Syntax in
          let* dtype, storage_key = storage_of_persid pid in
          let* storage_offset = offset_of "storage_offset" offset in
          let* sizes = sizes_of "size" size in
          let+ strides = strides_of "stride" stride in
          { storage_key; dtype; storage_offset; sizes; strides }
      | _ -> Err.fail `Unexpected_rebuild_args)
  | _ -> Err.fail `Unexpected_rebuild_args

(* The arguments of a [_rebuild_tensor_v2(...)] reduction, whatever operations
   follow it. *)
let rebuild_args v =
  match V.view v with
  | V.Object { constructor = V.Reduce { func; args }; _ } -> (
      match V.view func with
      | V.Global { name = "_rebuild_tensor_v2"; _ } -> Some args
      | _ -> None)
  | _ -> None

let children v =
  let constructor = function
    | V.Reduce { func; args } -> [ func; args ]
    | V.New { cls; args; kwargs } -> cls :: args :: Option.to_list kwargs
    | V.Legacy_instantiate { cls; args } -> [ cls; args ]
  in
  let operation = function
    | V.Append x | V.Add_item x | V.Build x -> [ x ]
    | V.Set_item (k, x) -> [ k; x ]
  in
  match V.view v with
  | V.Tuple a -> Array.to_list a
  | V.List l | V.Set l | V.Frozenset l -> l
  | V.Dict d -> List.concat_map (fun (k, x) -> [ k; x ]) d
  | V.Object { constructor = c; operations } ->
      constructor c @ List.concat_map operation operations
  | V.Persistent x | V.Readonly_buffer x -> [ x ]
  | _ -> []

(* Find the first _rebuild_tensor_v2 reduction anywhere in the value graph: a
   top-level tensor is wrapped by torch as ((tensor,), {}). The graph keeps
   pickle memo identity, so shared and self-referential nodes are visited once. *)
let find_tensor (v : V.t) =
  let seen = Hashtbl.create 64 in
  let rec go v =
    if Hashtbl.mem seen (V.id v) then Err.return None
    else begin
      Hashtbl.add seen (V.id v) ();
      match rebuild_args v with
      | Some args -> Err.map Option.some (rebuild_of_args args)
      | None -> first (children v)
    end
  and first = function
    | [] -> Err.return None
    | v :: vs -> (
        let open Err.Syntax in
        let* found = go v in
        match found with Some _ -> Err.return found | None -> first vs)
  in
  go v

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
let tensor_of_value key v =
  match rebuild_args v with
  | Some args -> rebuild_of_args args
  | None -> Err.fail (`Tensor_map_value_not_tensor key)

let parse_tensor_map s =
  match Opickle.of_string s with
  | Error e -> Err.fail (`Malformed_pickle (Opickle.Error.to_string e))
  | Ok v -> (
      match V.view v with
      | V.Dict entries ->
          let seen = Hashtbl.create 16 in
          let open Err.Syntax in
          let* pairs =
            Err.List.map
              (fun (key, value) ->
                match V.view key with
                | V.Str key ->
                    if Hashtbl.mem seen key then
                      Err.fail (`Duplicate_tensor_map_key key)
                    else begin
                      Hashtbl.add seen key ();
                      let+ tensor = tensor_of_value key value in
                      (key, tensor)
                    end
                | _ -> Err.fail `Tensor_map_key_not_string)
              entries
          in
          Err.return (List.sort (fun (a, _) (b, _) -> String.compare a b) pairs)
      | _ -> Err.fail `Expected_tensor_map)
