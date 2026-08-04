(* See direct.mli. Indices are ['role Dim.t] (not raw [int]): [position index
   = Dim.index Dim.t] carries its non-negativity proof through to
   [Tensor.read_at]/[Vec6.offset_of], which then need no re-validation on
   their (millions-of-calls) hot path — see .ai/pt2_inference_perf.md. [delta
   index = Dim.delta Dim.t] stays unchecked/signed, since window arithmetic
   goes negative before [clamp_low]/[assume_index] convert it back. *)

type t = float
type 'role index = 'role Dim.t
type input = Tensor.packed

let const x = x
let add = ( +. )
let sub = ( -. )
let mul = ( *. )
let div = ( /. )
let exp = Stdlib.exp
let sqrt = Stdlib.sqrt

type b = bool

let lt a b = a < b
let select c a b = if c then a else b
let index_zero : Semantics.position index = Dim.index 0

let index_extent (e : Dim.extent Dim.t) : Semantics.delta index =
  Dim.delta (e :> int)

let index_const (n : int) : Semantics.delta index = Dim.delta n

let of_index (i : Semantics.position index) : Semantics.delta index =
  Dim.to_delta i

let index_add (a : Semantics.delta index) (b : Semantics.delta index) :
    Semantics.delta index =
  Dim.delta ((a :> int) + (b :> int))

let index_scale (k : int) (i : Semantics.delta index) : Semantics.delta index =
  Dim.delta (k * (i :> int))

let floor_div_pos n d =
  let d = (d : Op_config.Pos.t :> int) in
  if n >= 0 then n / d else -((-n + d - 1) / d)

let index_floor_div_pos (n : Semantics.delta index) (d : Op_config.Pos.t) :
    Semantics.delta index =
  Dim.delta (floor_div_pos (n :> int) d)

let index_ceil_div_pos (n : Semantics.delta index) (d : Op_config.Pos.t) :
    Semantics.delta index =
  Dim.delta (-(floor_div_pos (-(n :> int)) d))

let index_min (a : Semantics.delta index) (b : Semantics.delta index) :
    Semantics.delta index =
  Dim.delta (Stdlib.min (a :> int) (b :> int))

let index_eq (x : Semantics.delta index) (y : Semantics.delta index) : bool =
  Int.equal (x :> int) (y :> int)

let clamp_low (x : Semantics.delta index) : Semantics.position index =
  Dim.index (Stdlib.max 0 (x :> int))

let assume_index (x : Semantics.delta index) : Semantics.position index =
  Dim.index (x :> int)

let value_of_index (x : Semantics.delta index) : t = float_of_int (x :> int)

let load inp (v : Semantics.position index Vec6.t) =
  Tensor.read_at6 inp ~n:v.Vec6.n ~t:v.Vec6.t ~d:v.Vec6.d ~h:v.Vec6.h
    ~w:v.Vec6.w ~c:v.Vec6.c

let load6 inp ~n ~t ~d ~h ~w ~c = Tensor.read_at6 inp ~n ~t ~d ~h ~w ~c

let pool_bounds ~out ~kernel ~stride ~pad ~extent =
  let start = (out * stride) - pad in
  (Stdlib.max 0 start, Stdlib.min extent (start + kernel))

let max_pool2d inp ~(x_shape : Vec6.shape)
    ~(kernel : Dim.extent Dim.t Op_config.Hw.t)
    ~(stride : Op_config.Pos.t Op_config.Hw.t)
    ~(pad : Op_config.Nonneg.t Op_config.Hw.t)
    (out : Semantics.position index Vec6.t) =
  let kh, kw = ((kernel.Op_config.Hw.h :> int), (kernel.w :> int)) in
  let sh, sw = ((stride.Op_config.Hw.h :> int), (stride.w :> int)) in
  let ph, pw = ((pad.Op_config.Hw.h :> int), (pad.w :> int)) in
  let h, w =
    ((Vec6.get x_shape Axis.H :> int), (Vec6.get x_shape Axis.W :> int))
  in
  let hlo, hhi =
    pool_bounds ~out:(out.Vec6.h :> int) ~kernel:kh ~stride:sh ~pad:ph ~extent:h
  in
  let wlo, whi =
    pool_bounds ~out:(out.Vec6.w :> int) ~kernel:kw ~stride:sw ~pad:pw ~extent:w
  in
  let rec loop_w ih iw best =
    if iw >= whi then loop_h (ih + 1) best
    else
      let v =
        load6 inp ~n:out.n ~t:out.t ~d:out.d ~h:(Dim.index ih) ~w:(Dim.index iw)
          ~c:out.c
      in
      loop_w ih (iw + 1) (Expr.Max_op.apply Pool_max best v)
  and loop_h ih best = if ih >= hhi then best else loop_w ih wlo best in
  loop_h hlo neg_infinity

let max_pool2d_index inp ~(x_shape : Vec6.shape)
    ~(kernel : Dim.extent Dim.t Op_config.Hw.t)
    ~(stride : Op_config.Pos.t Op_config.Hw.t)
    ~(pad : Op_config.Nonneg.t Op_config.Hw.t)
    (out : Semantics.position index Vec6.t) =
  let kh, kw = ((kernel.Op_config.Hw.h :> int), (kernel.w :> int)) in
  let sh, sw = ((stride.Op_config.Hw.h :> int), (stride.w :> int)) in
  let ph, pw = ((pad.Op_config.Hw.h :> int), (pad.w :> int)) in
  let h, w =
    ((Vec6.get x_shape Axis.H :> int), (Vec6.get x_shape Axis.W :> int))
  in
  let hlo, hhi =
    pool_bounds ~out:(out.Vec6.h :> int) ~kernel:kh ~stride:sh ~pad:ph ~extent:h
  in
  let wlo, whi =
    pool_bounds ~out:(out.Vec6.w :> int) ~kernel:kw ~stride:sw ~pad:pw ~extent:w
  in
  let rec loop_w ih iw best best_index =
    if iw >= whi then loop_h (ih + 1) best best_index
    else
      let v =
        load6 inp ~n:out.n ~t:out.t ~d:out.d ~h:(Dim.index ih) ~w:(Dim.index iw)
          ~c:out.c
      in
      (* One predicate updates value and index together, so they cannot fall out
         of step (a NaN taken as the value but not as the index). Ties keep the
         incumbent, preserving the first (smallest flat) index. *)
      let best, best_index =
        if Expr.Max_op.pool_better ~best ~value:v then (v, (ih * w) + iw)
        else (best, best_index)
      in
      loop_w ih (iw + 1) best best_index
  and loop_h ih best best_index =
    if ih >= hhi then best_index else loop_w ih wlo best best_index
  in
  float_of_int (loop_h hlo neg_infinity 0)

let sum ~(lo : Semantics.position index) ~(hi : Semantics.delta index)
    (f : Semantics.position index -> t) =
  let rec loop (i : Semantics.position index) acc =
    if (i :> int) >= (hi :> int) then acc else loop (Dim.succ i) (acc +. f i)
  in
  loop lo 0.

let max_reduce ~(lo : Semantics.position index) ~(hi : Semantics.delta index)
    (f : Semantics.position index -> t) =
  let rec loop (i : Semantics.position index) acc =
    if (i :> int) >= (hi :> int) then acc
    else loop (Dim.succ i) (Expr.Max_op.apply Float_max acc (f i))
  in
  loop lo neg_infinity
