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

type error =
  [ Pt2_dtype.error
  | `Malformed_pickle of string
  | `Expected_int of string
  | `Int_out_of_range of string * int64
  | `Unexpected_storage_persistent_id
  | `Unexpected_rebuild_args
  | `No_tensor_found ]

let pp_error ppf : error -> unit = function
  | #Pt2_dtype.error as e -> Pt2_dtype.pp_error ppf e
  | `Malformed_pickle msg -> Fmt.pf ppf "pickle decode failed: %s" msg
  | `Expected_int what -> Fmt.pf ppf "expected int for %s" what
  | `Int_out_of_range (what, i) ->
      Fmt.pf ppf "%s is %Ld, outside the representable int range [%d, %d]" what
        i min_int max_int
  | `Unexpected_storage_persistent_id ->
      Fmt.string ppf "unexpected storage persistent id"
  | `Unexpected_rebuild_args ->
      Fmt.string ppf "unexpected _rebuild_tensor_v2 arguments"
  | `No_tensor_found -> Fmt.string ppf "pickle did not yield a tensor"

(* A distinct tag from [`Expected_int]: the value here IS an integer, it just
   does not fit this backend's [int]. Under js_of_ocaml that is 32 bits, so a
   size or stride outside [min_int, max_int] would wrap silently -- and the
   wrap is two-sided, because pickle integers are signed. Reporting "expected
   an int" would send a reader hunting a type error in the pickle instead of a
   width limit. See [[js_backends_design]]. *)
let int_of what = function
  | V.Int i ->
      if
        Int64.compare i (Int64.of_int min_int) < 0
        || Int64.compare i (Int64.of_int max_int) > 0
      then Err.fail (`Int_out_of_range (what, i))
      else Err.return (Int64.to_int i)
  | _ -> Err.fail (`Expected_int what)

let ints_of what a = Err.List.map (int_of what) (Array.to_list a)

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
          let* storage_offset = int_of "storage_offset" offset in
          let* sizes = ints_of "size" size in
          let+ strides = ints_of "stride" stride in
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
