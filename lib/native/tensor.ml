(* A tensor is just a 6D shape plus a typed payload — dense, channels-last, no
   strides (native_tensor_design.md §1c). The element at a coord is
   [read_coord] (broadcast) -> [Vec6.offset] -> [Payload.get_float]. *)

type ('elt, 'ba, 'q) t = {
  shape : Vec6.shape;
  payload : ('elt, 'ba, 'q) Payload.payload;
}

(* The concrete format is hidden; tensors flow through the graph as [packed]. *)
type packed = Tensor : ('elt, 'ba, 'q) t -> packed

let channel c = Dim.to_int (Vec6.get c Axis.C)

(* Read one element as a float, broadcasting extent-1 axes of the source. *)
let read (Tensor t) (coord : Vec6.coord) : float =
  let c = Vec6.read_coord t.shape coord in
  let i = (Vec6.offset t.shape c :> int) in
  Payload.get_float t.payload ~c:(channel c) ~i

(* Read at the coord given by [idx a] per axis, with broadcast (a source axis of
   extent 1 is read at index 0) and padding (an index outside the source makes the
   whole read 0). This is the one read primitive ops use, via the semantics. *)
let read_guarded (Tensor t as packed) (idx : Axis.t -> int) : float =
  let extent a = Dim.to_int (Vec6.get t.shape a) in
  let in_range a = extent a = 1 || (idx a >= 0 && idx a < extent a) in
  if not (List.for_all in_range Axis.all) then 0.
  else
    let comp a = if extent a = 1 then 0 else idx a in
    read packed
      (Vec6.coord ~n:(comp N) ~t:(comp T) ~d:(comp D) ~h:(comp H) ~w:(comp W)
         ~c:(comp C))

(* tap helper: the source coord = [base] + per-axis signed deltas (0 where an axis
   is absent), guarded into the source extents. [None] is the pad region. *)
let shift_in_bounds (Tensor t) (base : Vec6.coord)
    (deltas : (Axis.t * Dim.delta Dim.t) list) : Vec6.coord option =
  let idx a =
    let d =
      match List.assoc_opt a deltas with Some d -> Dim.to_int d | None -> 0
    in
    Dim.to_int (Vec6.get base a) + d
  in
  let in_range a = idx a >= 0 && idx a < Dim.to_int (Vec6.get t.shape a) in
  if List.for_all in_range Axis.all then
    Some
      (Vec6.coord ~n:(idx N) ~t:(idx T) ~d:(idx D) ~h:(idx H) ~w:(idx W)
         ~c:(idx C))
  else None

(* Build a fresh dense float32 tensor by running [f] over every output coord. *)
let materialize (shape : Vec6.shape) (f : Vec6.coord -> float) : packed =
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
