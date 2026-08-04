(* See symbolic.mli. *)

type t = Expr.Value.t Expr.Builder.t
type 'role index = 'role Expr.Index.t
type input = Tensor_sig.t
type b = Expr.Bool.t Expr.Builder.t

(* Lifting helpers. A symbolic value is a construction COMPUTATION, so every
   combinator taking more than one of them threads the supply through its
   operands -- left to right, which is what makes construction deterministic
   rather than merely correct. Local rather than added to [Expr.Builder]:
   nothing outside this adapter needs them. *)
let map2 f a b =
  let open Expr.Builder.Syntax in
  let* x = a in
  let+ y = b in
  f x y

let map3 f a b c =
  let open Expr.Builder.Syntax in
  let* x = a in
  let* y = b in
  let+ z = c in
  f x y z

(* Everything below that does not combine two computations is pure, and says so
   by going through [return] rather than through the monad. Index operations do
   not appear here at all: ['role index] is an [Expr.Index.t], not a
   computation, because indices bind nothing and so carry no supply. *)
let const x = Expr.Builder.return (Expr.Value.const x)
let add a b = map2 Expr.Value.add a b
let sub a b = map2 Expr.Value.sub a b
let mul a b = map2 Expr.Value.mul a b
let div a b = map2 Expr.Value.div a b
let exp a = Expr.Builder.map Expr.Value.exp a
let sqrt a = Expr.Builder.map Expr.Value.sqrt a
let lt a b = map2 Expr.Bool.value_lt a b
let select c a b = map3 Expr.Value.select c a b
let index_zero = Expr.Index.zero
let index_extent (e : Dim.extent Dim.t) = Expr.Index.const (e :> int)
let index_const = Expr.Index.const
let of_index = Expr.Index.of_position
let index_add = Expr.Index.add
let index_scale = Expr.Index.scale
let index_min = Expr.Index.min
let index_eq a b = Expr.Builder.return (Expr.Bool.index_eq a b)
let clamp_low = Expr.Index.clamp_low
let assume_index = Expr.Index.assume_position
let value_of_index i = Expr.Builder.return (Expr.Value.value_of_index i)

(* [SEMANTICS] returns a plain index while the smart constructor is
   result-returning, so the wrapper is dropped here -- deliberately, and
   through the repository's NAMED boundary rather than an open-coded match, so
   it stays distinguishable from the defect that silently discards a
   backtrace. It is unreachable in practice: [Op_config.Pos.t] is already
   validated positive, which is the whole point of the type. *)
let to_index r = Core.or_raise Expr.Index.pp_error r

let index_floor_div_pos a (d : Op_config.Pos.t) =
  to_index (Expr.Index.floor_div_pos a (d :> int))

let index_ceil_div_pos a (d : Op_config.Pos.t) =
  to_index (Expr.Index.ceil_div_pos a (d :> int))

let load (s : input) v =
  Expr.Builder.return
    (Expr.Value.load
       (Expr_bridge.source_of_id s.Tensor_sig.id)
       (Expr_bridge.coord_of_vec6 v))

let load6 (s : input) ~n ~t ~d ~h ~w ~c =
  Expr.Builder.return
    (Expr.Value.load
       (Expr_bridge.source_of_id s.Tensor_sig.id)
       (Expr.Coord.make ~n ~t ~d ~h ~w ~c))

(* The descriptor carries [in_h]/[in_w] explicitly, because an opaque source
   cannot be asked for its shape the way the embedded [Tensor_sig.t] could.
   They come from [~x_shape], which the older instance ignores precisely
   because it stashed the signature instead. *)
let max_pool (input : input) ~(x_shape : Vec6.shape) ~kernel ~stride ~pad out
    result =
  (* Extracted per field rather than through a shared helper: the three [Hw.t]s
     carry different element types ([Dim.extent Dim.t], [Pos.t], [Nonneg.t]),
     so one helper would be monomorphised at whichever came first. *)
  let kernel_h = (kernel.Op_config.Hw.h : Dim.extent Dim.t :> int)
  and kernel_w = (kernel.Op_config.Hw.w : Dim.extent Dim.t :> int)
  and stride_h = (stride.Op_config.Hw.h : Op_config.Pos.t :> int)
  and stride_w = (stride.Op_config.Hw.w : Op_config.Pos.t :> int)
  and pad_h = (pad.Op_config.Hw.h : Op_config.Nonneg.t :> int)
  and pad_w = (pad.Op_config.Hw.w : Op_config.Nonneg.t :> int) in
  Expr.Builder.return
    (Expr.Value.intrinsic
       (Core.or_raise Expr.Intrinsic.pp_error
          (Expr.Intrinsic.max_pool
             ~source:(Expr_bridge.source_of_id input.Tensor_sig.id)
             ~in_h:(Dim.to_int (Vec6.get x_shape Axis.H))
             ~in_w:(Dim.to_int (Vec6.get x_shape Axis.W))
             ~kernel_h ~kernel_w ~stride_h ~stride_w ~pad_h ~pad_w
             ~out:(Expr_bridge.coord_of_vec6 out)
             ~result)))

let max_pool2d input ~x_shape ~kernel ~stride ~pad out =
  max_pool input ~x_shape ~kernel ~stride ~pad out Expr.Intrinsic.Max_pool.Value

let max_pool2d_index input ~x_shape ~kernel ~stride ~pad out =
  max_pool input ~x_shape ~kernel ~stride ~pad out Expr.Intrinsic.Max_pool.Index

(* [Builder.reduction] mints the variable, hands its index to the body and
   threads the supply -- the scope is correct by construction, and the adapter
   never touches a [Reduce_var.t]. The body type [SEMANTICS] demands,
   [position index -> t], is already the type [reduction] expects. *)
let reduce ~kind ~lo ~hi f = Expr.Builder.reduction ~kind ~lo ~hi f
let sum ~lo ~hi f = reduce ~kind:Expr.Reduction.Sum ~lo ~hi f
let max_reduce ~lo ~hi f = reduce ~kind:Expr.Reduction.Max ~lo ~hi f

let out_vec : Semantics.position Expr.Index.t Vec6.t =
  Vec6.make ~n:(Expr.Index.output Axis.N) ~t:(Expr.Index.output Axis.T)
    ~d:(Expr.Index.output Axis.D) ~h:(Expr.Index.output Axis.H)
    ~w:(Expr.Index.output Axis.W) ~c:(Expr.Index.output Axis.C)
