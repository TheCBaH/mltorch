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
  storage_offset : int;
  sizes : int list;
  strides : int list;
}

let fail msg = failwith ("Pt2_pickle: " ^ msg)
let int_of = function V.Int i -> Int64.to_int i | _ -> fail "expected int"
let ints_of a = Array.to_list (Array.map int_of a)

(* A storage persistent id is ('storage', <TypeStorage>, key, device, numel). *)
let storage_of_persid = function
  | V.Tuple [| V.Str "storage"; V.Global { name; _ }; V.Str key; _; _ |] ->
      (Pt2_dtype.of_storage_name name, key)
  | _ -> fail "unexpected storage persistent id"

let rebuild_of_args = function
  | V.Tuple a when Array.length a >= 4 -> (
      match (a.(0), a.(1), a.(2), a.(3)) with
      | V.Persistent pid, offset, V.Tuple size, V.Tuple stride ->
          let dtype, storage_key = storage_of_persid pid in
          {
            storage_key;
            dtype;
            storage_offset = int_of offset;
            sizes = ints_of size;
            strides = ints_of stride;
          }
      | _ -> fail "unexpected _rebuild_tensor_v2 arguments")
  | _ -> fail "unexpected _rebuild_tensor_v2 arguments"

(* Find the first _rebuild_tensor_v2 reduction anywhere in the value tree: a
   top-level tensor is wrapped by torch as ((tensor,), {}). *)
let rec find_tensor (v : V.t) =
  match v with
  | V.Reduce { func = V.Global { name = "_rebuild_tensor_v2"; _ }; args } ->
      Some (rebuild_of_args args)
  | V.Tuple a -> Array.find_map find_tensor a
  | V.List r | V.Set r -> List.find_map find_tensor !r
  | V.Frozenset l -> List.find_map find_tensor l
  | V.Dict r ->
      List.find_map
        (fun (k, v) ->
          match find_tensor k with Some _ as s -> s | None -> find_tensor v)
        !r
  | V.Reduce { func; args } -> (
      match find_tensor func with Some _ as s -> s | None -> find_tensor args)
  | V.Object { cls; args; state } -> (
      match find_tensor cls with
      | Some _ as s -> s
      | None -> (
          match find_tensor args with
          | Some _ as s -> s
          | None -> Option.bind state find_tensor))
  | V.Persistent id -> find_tensor id
  | _ -> None

let parse_tensor s =
  match Opickle.of_string s with
  | Error e -> fail (Opickle.Error.to_string e)
  | Ok v -> (
      match find_tensor v with
      | Some rb -> rb
      | None -> fail "pickle did not yield a tensor")
