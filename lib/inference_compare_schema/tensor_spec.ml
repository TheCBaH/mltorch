module W = Wire
module Layout = W.Text (Id.Layout)

type t = {
  dtype : Dtype.t;
  logical_shape : int64 list;
  layout_id : Id.Layout.t;
  value_semantics : bool;
}

let compare a b =
  match Dtype.compare a.dtype b.dtype with
  | 0 -> (
      match List.compare Int64.compare a.logical_shape b.logical_shape with
      | 0 -> (
          match Id.Layout.compare a.layout_id b.layout_id with
          | 0 -> Bool.compare a.value_semantics b.value_semantics
          | c -> c)
      | c -> c)
  | c -> c

let equal a b = compare a b = 0

(* The product of the extents, or [None] when it exceeds [Int64.max_int].
   Dividing the running product into the ceiling before multiplying keeps the
   test on the exact value: a check on a wrapped product is not a bound. *)
let element_count spec =
  let rec go acc = function
    | [] -> Some acc
    | d :: rest ->
        if Int64.compare d 0L = 0 then
          (* A zero extent makes the count zero whatever the other extents
             are, but every extent must still be nonnegative. *)
          if List.for_all (fun x -> Int64.compare x 0L >= 0) rest then Some 0L
          else None
        else if Int64.compare d 0L < 0 then None
        else if Int64.compare acc (Int64.div Int64.max_int d) > 0 then None
        else go (Int64.mul acc d) rest
  in
  go 1L spec.logical_shape

let pp ppf s =
  Fmt.pf ppf "%a%a@%a" Dtype.pp s.dtype
    Fmt.(brackets (list ~sep:comma int64))
    s.logical_shape Id.Layout.pp s.layout_id

let decode_shape : int64 list W.decoder =
 fun j ->
  let ( let* ) = W.( let* ) in
  let* dims = W.list W.int64 j in
  if List.exists (fun d -> Int64.compare d 0L < 0) dims then
    W.fail (W.Constraint Negative_value)
  else Ok dims

let decode : t W.decoder =
  W.record (fun o ->
      let ( let* ) = W.( let* ) in
      let* dtype = W.Obj.required o "dtype" Dtype.decode in
      let* logical_shape = W.Obj.required o "logical_shape" decode_shape in
      let* layout_id = W.Obj.required o "layout_id" Layout.decode in
      let* value_semantics = W.Obj.required o "value_semantics" W.bool in
      Ok { dtype; logical_shape; layout_id; value_semantics })

let encode s =
  W.obj
    [
      W.field "dtype" Dtype.encode s.dtype;
      W.field "logical_shape" (W.enc_list W.enc_int64) s.logical_shape;
      W.field "layout_id" Layout.encode s.layout_id;
      W.field "value_semantics" W.enc_bool s.value_semantics;
    ]
