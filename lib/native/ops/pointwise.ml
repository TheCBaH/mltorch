(* Pointwise ops: each output pixel reads the input(s) at the same coord. The
   functor is over SEMANTICS, so the same definition serves Direct and Symbolic.
   [out] is the output coord itself, one index expression per axis.
   See .ai/native_compute_design.md §2. *)

(* Output shape of a binary broadcasting op. Per axis the two extents must be
   equal, or one of them must be 1 (the broadcast axis, which takes the other's
   extent); any other mismatch is incompatible and an error — NOT silently the
   larger of the two. Computed in extent-space (no [:> int] round-trips); start
   from [a_shape] and overwrite every axis. See .ai/native_compute_design.md §2b. *)
let broadcast_output_shape (a_shape : Vec6.shape) (b_shape : Vec6.shape) =
  let open Core.Syntax in
  Core.List.fold_left
    (fun s axis ->
      let a = Vec6.get a_shape axis and b = Vec6.get b_shape axis in
      let* out =
        if Dim.equal a b then Core.return a
        else if Dim.equal a Dim.one then Core.return b
        else if Dim.equal b Dim.one then Core.return a
        else
          Core.fail
            (`Broadcast Shape_error.Broadcast.{ axis; lhs = a; rhs = b })
      in
      Core.return (Vec6.set s axis out))
    a_shape Axis.all

(* Broadcasting for a binary op. [load] is strict — an out-of-bounds index is an
   error, never a silent fan-out — so an operand with an extent-1 (broadcast) axis
   must have the output coord reduced to a valid read of it FIRST: [broadcast_coord
   shape out_vec] maps every axis whose source extent is 1 to [index_zero] and
   keeps [out_vec]'s value elsewhere, so a single stored value is read at index 0
   on a broadcast axis regardless of where the output iterates. The decision is a
   static per-axis shape test, independent of the index value, so the same helper
   serves [Direct] (int indices) and [Symbolic] (index expressions) — it only
   needs [index_zero]. [Vec6.mapi], not a closure: the caller already has (or can
   share) a materialized [Vec6.t], and this returns one directly, ready for
   [SEMANTICS.load]. See .ai/native_tensor_design.md §1b. *)
let broadcast_coord ~(index_zero : 'i) (shape : Vec6.shape)
    (out_vec : 'i Vec6.t) : 'i Vec6.t =
  Vec6.mapi
    (fun a i -> if Dim.equal (Vec6.get shape a) Dim.one then index_zero else i)
    out_vec

module Relu = struct
  type t = { x : Tensor_ref.t }

  let name = "Relu"

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        { x = Json_util.req_field ms "x" Tensor_ref.jsont name })
      ~enc:(fun t ->
        Json_util.jobj [ ("x", Json_util.enc Tensor_ref.jsont t.x) ])
      Jsont.json

  let operands (t : t) = [ t.x ]
  let map_operands f (t : t) = { x = f t.x }

  let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>relu@ x=%a@]" pp_ref t.x

  (* Identity: relu doesn't change shape. See .ai/native_compute_design.md §2b. *)
  let output_shape (x_shape : Vec6.shape) = Core.return x_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    (* relu x = (x < 0 ? 0 : x) — derived from [select]+[lt], not a primitive *)
    let pixel x (out : Semantics.position S.index Vec6.t) =
      let v = S.load x out in
      S.select (S.lt v (S.const 0.)) (S.const 0.) v
  end

  (* This op's own random-walk config space: a single tensor of any shape (relu
     imposes no constraint, so [cascade] is identity). A functor over the global
     Limits; the shape is one compound entry mutated within the limits' budget.
     Lives with the op so it can diverge from any other backend's walk. *)
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = { shape : Walk_core.Shape.t }

    let initial =
      { shape = { Walk_core.Shape.n = 1; t = 1; d = 1; h = 4; w = 4; c = 3 } }

    let cascade c = c
    let shape (c : cfg) = Walk_bridge.vec6 c.shape

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun _ s -> { shape = s });
        ]

    let pp fmt (c : cfg) = Walk_core.Shape.pp fmt c.shape
  end
end

(* A binary elementwise op: read each operand at the output coord reduced against
   that operand's own shape ([broadcast_coord]), so an extent-1 axis fans out
   without ever handing [load] an out-of-bounds index. [combine] is the scalar op
   (S.add, S.mul, …). *)
module Binary (S : Semantics.SEMANTICS) = struct
  let pixel ~combine ~a_shape ~b_shape a b
      (out : Semantics.position S.index Vec6.t) =
    let read shape t =
      S.load t (broadcast_coord ~index_zero:S.index_zero shape out)
    in
    combine (read a_shape a) (read b_shape b)
end

(* Payload shared by the binary pointwise ops [Add]/[Mul]: two operand refs. Each
   op aliases its [t] to this so the [a]/[b] labels are defined once (avoiding
   cross-op label ambiguity) and the serialise/dataflow/pp boilerplate is written
   once, parameterised only by the JSON case [name] and the printed [op] keyword. *)
module Bin = struct
  type t = { a : Tensor_ref.t; b : Tensor_ref.t }

  let jsont ~name : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        let get k = Json_util.req_field ms k Tensor_ref.jsont name in
        { a = get "a"; b = get "b" })
      ~enc:(fun t ->
        let ref_ = Json_util.enc Tensor_ref.jsont in
        Json_util.jobj [ ("a", ref_ t.a); ("b", ref_ t.b) ])
      Jsont.json

  let operands (t : t) = [ t.a; t.b ]
  let map_operands f (t : t) = { a = f t.a; b = f t.b }

  let pp ~op (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>%s@ a=%a@ b=%a@]" op pp_ref t.a pp_ref t.b

  (* Walk config space shared by the binary pointwise ops (Add/Mul): one shape
     used for both operands (equal-shape; broadcast not exercised). A functor
     over the global Limits; no constraint, so [cascade] is identity. *)
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = { shape : Walk_core.Shape.t }

    let initial =
      { shape = { Walk_core.Shape.n = 1; t = 1; d = 1; h = 4; w = 4; c = 3 } }

    let cascade c = c
    let shape (c : cfg) = Walk_bridge.vec6 c.shape

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun _ s -> { shape = s });
        ]

    let pp fmt (c : cfg) = Walk_core.Shape.pp fmt c.shape
  end
end

module Add = struct
  type t = Bin.t

  let name = "Add"
  let jsont = Bin.jsont ~name
  let operands = Bin.operands
  let map_operands = Bin.map_operands
  let pp pp_ref fmt t = Bin.pp ~op:"add" pp_ref fmt t
  let output_shape = broadcast_output_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    module B = Binary (S)

    let pixel ~a_shape ~b_shape a b out =
      B.pixel ~combine:S.add ~a_shape ~b_shape a b out
  end
end

module Mul = struct
  type t = Bin.t

  let name = "Mul"
  let jsont = Bin.jsont ~name
  let operands = Bin.operands
  let map_operands = Bin.map_operands
  let pp pp_ref fmt t = Bin.pp ~op:"mul" pp_ref fmt t
  let output_shape = broadcast_output_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    module B = Binary (S)

    let pixel ~a_shape ~b_shape a b out =
      B.pixel ~combine:S.mul ~a_shape ~b_shape a b out
  end
end
