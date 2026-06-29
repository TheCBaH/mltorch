(* A tensor is just a 6D shape plus a typed payload — dense, channels-last, no
   strides (native_tensor_design.md §1c). The element at a coord is
   [Vec6.in_bounds] (strict) -> [Vec6.offset] -> [Payload.get_float]. *)

type ('elt, 'ba, 'q) t = {
  shape : Vec6.shape;
  payload : ('elt, 'ba, 'q) Payload.payload;
}

(* The concrete format is hidden; tensors flow through the graph as [packed]. *)
type packed = Tensor : ('elt, 'ba, 'q) t -> packed

let channel c = Dim.to_int (Vec6.get c Axis.C)

(* Strict read of one element as a float: every axis index must lie within the
   source extent, else [Invalid_argument]. This is the one read primitive — it
   does NOT broadcast or pad. A caller that needs an extent-1 axis to fan out
   reduces the coordinate first ([Pointwise.broadcast_coord]); one that needs a pad value
   outside the source takes a guarded tap ([shift_in_bounds]) and supplies its
   own. See .ai/native_tensor_design.md §1b. *)
let read (Tensor t) (coord : Vec6.coord) =
  if not (Vec6.in_bounds t.shape coord) then
    invalid_arg
      (Format.asprintf "Tensor.read: coord %a out of bounds for shape %a"
         Vec6.pp_coord coord Vec6.pp_shape t.shape);
  let i = (Vec6.offset t.shape coord :> int) in
  Payload.get_float t.payload ~c:(channel coord) ~i

(* Strict read at an [idx a] index per axis — the primitive ops reach through the
   semantics' [load]. Builds the coord and reads via [read]; an out-of-range index
   raises rather than silently broadcasting or padding. *)
let read_at packed (idx : Axis.t -> int) =
  read packed
    (Vec6.coord ~n:(idx N) ~t:(idx T) ~d:(idx D) ~h:(idx H) ~w:(idx W)
       ~c:(idx C))

(* tap helper: the source coord = [base] + per-axis signed [deltas], guarded into
   the source extents. [None] is the pad region. *)
let shift_in_bounds (Tensor t) (base : Vec6.coord) (deltas : Vec6.deltas) =
  let idx a = Dim.to_int (Vec6.get base a) + Dim.to_int (Vec6.get deltas a) in
  let in_range a = idx a >= 0 && idx a < Dim.to_int (Vec6.get t.shape a) in
  if List.for_all in_range Axis.all then
    Some
      (Vec6.coord ~n:(idx N) ~t:(idx T) ~d:(idx D) ~h:(idx H) ~w:(idx W)
         ~c:(idx C))
  else None

(* Build a fresh dense float32 tensor by running [f] over every output coord. *)
let materialize (shape : Vec6.shape) (f : Vec6.coord -> float) =
  let n = (Vec6.numel shape :> int) in
  let data = Bigarray.(Array1.create float32 c_layout n) in
  let payload = { Payload.fmt = Payload.F32; quant = Payload.No_quant; data } in
  Vec6.iter shape (fun c ->
      let i = (Vec6.offset shape c :> int) in
      Payload.set_float payload ~c:(channel c) ~i (f c));
  Tensor { shape; payload }

let pp fmt (Tensor t) =
  Format.fprintf fmt "tensor %a %a {" Payload.pp t.payload Vec6.pp_shape t.shape;
  let n = (Vec6.numel t.shape :> int) in
  let k = min n 8 in
  let count = ref 0 in
  (try
     Vec6.iter t.shape (fun c ->
         if !count >= k then raise Exit;
         if !count > 0 then Format.fprintf fmt ", ";
         let i = (Vec6.offset t.shape c :> int) in
         Format.fprintf fmt "%g" (Payload.get_float t.payload ~c:(channel c) ~i);
         incr count)
   with Exit -> ());
  if n > k then Format.fprintf fmt ", ...";
  Format.fprintf fmt "}"
