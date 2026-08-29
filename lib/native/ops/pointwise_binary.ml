(* Binary pointwise ops split out of pointwise.ml see pointwise.ml (the facade). Includes the broadcasting helpers and the
   Binary/Scalar_binary functors, since only the concrete ops
   (Add/Sub/Mul/Div/.../Pow) are part of pointwise.ml's external surface. *)

(* Output shape of a binary broadcasting op. Per axis the two extents must be
   equal, or one of them must be 1 (the broadcast axis, which takes the other's
   extent); any other mismatch is incompatible and an error — NOT silently the
   larger of the two. Computed in extent-space (no [:> int] round-trips); start
   from [a_shape] and overwrite every axis. See .ai/native_compute_design.md §2b. *)
let broadcast_output_shape (a_shape : Vec6.shape) (b_shape : Vec6.shape) =
  let open Err.Syntax in
  Err.List.fold_left
    (fun s axis ->
      let a = Vec6.get a_shape axis and b = Vec6.get b_shape axis in
      let* out =
        if Dim.equal a b then Err.return a
        else if Dim.equal a Dim.one then Err.return b
        else if Dim.equal b Dim.one then Err.return a
        else
          Err.fail (`Broadcast Shape_error.Broadcast.{ axis; lhs = a; rhs = b })
      in
      Err.return (Vec6.set s axis out))
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

(* Payload shared by the scalar-operand pointwise ops [Add_scalar]/[Div_scalar]:
   one operand ref and one compile-time scalar. The exporter serialises a
   compile-time scalar straight into a Tensor-typed slot (MobileNet-v3's
   hardsigmoid is `add(x, 3)` / `div(x, 6)` with `as_int` arguments), and the
   ATen interp materialises those with [full_like]. Here the scalar stays a
   graph parameter instead: no extra edge to bind, and the symbolic path gets a
   [const] leaf rather than a load, which is what a later fusion pass wants.

   [scalar] is f32-canonical by construction (the builder narrows it), because
   the engine's tensors are F32 and a scalar that is not f32-exact would
   otherwise round differently here than under the ATen path's [full_like] on
   an F32 tensor. *)
module Scalar_bin = struct
  type t = { x : Tensor_ref.t; scalar : float }

  let jsont ~name : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        {
          x = Json_util.req_field ms "x" Tensor_ref.jsont name;
          scalar = Json_util.req_field ms "scalar" Json_util.f32_jsont name;
        })
      ~enc:(fun t ->
        Json_util.jobj
          [
            ("x", Json_util.enc Tensor_ref.jsont t.x);
            ("scalar", Json_util.enc Json_util.f32_jsont t.scalar);
          ])
      Jsont.json

  let operands (t : t) = [ t.x ]
  let map_operands f (t : t) = { t with x = f t.x }

  let pp ~op (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>%s@ x=%a@ scalar=%a@]" op pp_ref t.x Fmt.float t.scalar

  (* Walk config shared by the scalar pointwise ops: one shape plus a scalar
     drawn from a finite candidate set. Zero is deliberately absent so the same
     space is safe for [Div_scalar] — unlike [Div], whose random *tensor*
     divisor is what makes it unwalkable. Explicit zero behaviour is pinned by
     the direct unit tests instead. The list mixes integer-like values (what
     MobileNet actually serialises), a negative, and an f32-inexact value. *)
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = { shape : Walk_core.Shape.t; scalar : float }

    let candidates = [ 3.; 6.; -2.; 0.5; 0.1 ]

    let initial =
      {
        shape = { Walk_core.Shape.n = 1; t = 1; d = 1; h = 4; w = 4; c = 3 };
        scalar = 3.;
      }

    let cascade c = c
    let shape (c : cfg) = Walk_bridge.vec6 c.shape

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun c s -> { c with shape = s });
          field_axis "scalar" candidates (fun c v -> { c with scalar = v });
        ]

    let pp fmt (c : cfg) =
      Fmt.pf fmt "%a scalar=%a" Walk_core.Shape.pp c.shape Fmt.float c.scalar
  end
end

(* A scalar elementwise op: the operand already has the output shape, so it is
   read at [out] directly (no broadcast reduction), and the scalar becomes a
   [const] in the value domain. *)
module Scalar_binary (S : Semantics.SEMANTICS) = struct
  let pixel ~combine ~scalar x (out : Semantics.position S.index Vec6.t) =
    combine (S.load x out) (S.const scalar)
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

module Add_scalar = struct
  type t = Scalar_bin.t

  let name = "Add_scalar"
  let jsont = Scalar_bin.jsont ~name
  let operands = Scalar_bin.operands
  let map_operands = Scalar_bin.map_operands
  let pp pp_ref fmt t = Scalar_bin.pp ~op:"add_scalar" pp_ref fmt t
  let output_shape (x_shape : Vec6.shape) = Err.return x_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    module B = Scalar_binary (S)

    let pixel ~scalar x out = B.pixel ~combine:S.add ~scalar x out
  end
end

module Div = struct
  type t = Bin.t

  let name = "Div"
  let jsont = Bin.jsont ~name
  let operands = Bin.operands
  let map_operands = Bin.map_operands
  let pp pp_ref fmt t = Bin.pp ~op:"div" pp_ref fmt t
  let output_shape = broadcast_output_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    module B = Binary (S)

    (* A zero divisor is the caller's business, exactly as it is for the
       underlying float division: no guard here, so the result is the IEEE
       infinity or NaN rather than a silently substituted value. *)
    let pixel ~a_shape ~b_shape a b out =
      B.pixel ~combine:S.div ~a_shape ~b_shape a b out
  end
end

module Div_scalar = struct
  type t = Scalar_bin.t

  let name = "Div_scalar"
  let jsont = Scalar_bin.jsont ~name
  let operands = Scalar_bin.operands
  let map_operands = Scalar_bin.map_operands
  let pp pp_ref fmt t = Scalar_bin.pp ~op:"div_scalar" pp_ref fmt t
  let output_shape (x_shape : Vec6.shape) = Err.return x_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    module B = Scalar_binary (S)

    (* As in [Div], a zero divisor is the caller's business: the result is the
       IEEE infinity or NaN, not a silently substituted value. Unlike [Div] this
       op is still walkable, because its divisor comes from a fixed candidate
       list that omits zero rather than from a random tensor. *)
    let pixel ~scalar x out = B.pixel ~combine:S.div ~scalar x out
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

module Mul_scalar = struct
  type t = Scalar_bin.t

  let name = "Mul_scalar"
  let jsont = Scalar_bin.jsont ~name
  let operands = Scalar_bin.operands
  let map_operands = Scalar_bin.map_operands
  let pp pp_ref fmt t = Scalar_bin.pp ~op:"mul_scalar" pp_ref fmt t
  let output_shape (x_shape : Vec6.shape) = Err.return x_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    module B = Scalar_binary (S)

    let pixel ~scalar x out = B.pixel ~combine:S.mul ~scalar x out
  end
end

module Pow = struct
  (* [x ** scalar] -- ATen's `aten.pow.Tensor_Scalar`. [scalar] is a
     compile-time constant: an ATen [Scalar] argument is never wired from
     another node's output (a tensor-valued exponent traces to
     `pow.Tensor_Tensor`, a different node this op does not cover), and the
     bridge/importer reject anything else as a decode error. So the branch
     below runs once per graph node, in OCaml, not once per pixel -- mirroring
     ATen's own [PowKernel.cpp], which special-cases exactly these six
     exponents (sqrt/rsqrt/reciprocal/square/cube/reciprocal-of-square) via
     [sqrt_kernel]/[rsqrt_kernel]/[reciprocal_kernel]/multiplication rather
     than [std::pow], for the same accuracy reason -- not a per-element
     runtime check either engine has to make. Any other exponent falls back
     to [exp(scalar * log x)], which -- unlike the six special cases -- is
     only [Equivalent] to ATen's [std::pow], not bit-identical, and (like
     [Sqrt]) is only correct for x > 0: [log] of a non-positive base is not
     what [std::pow] returns for a negative base with a non-integer exponent
     (NaN either way) or a base of exactly 0 ([std::pow(0, e)] is
     well-defined; [log 0] is not). *)
  type t = Scalar_bin.t

  let name = "Pow"
  let jsont = Scalar_bin.jsont ~name
  let operands = Scalar_bin.operands
  let map_operands = Scalar_bin.map_operands
  let pp pp_ref fmt t = Scalar_bin.pp ~op:"pow" pp_ref fmt t
  let output_shape (x_shape : Vec6.shape) = Err.return x_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    let pixel ~scalar x (out : Semantics.position S.index Vec6.t) =
      let v = S.load x out in
      let reciprocal v = S.div (S.const 1.) v in
      if scalar = 2.0 then S.mul v v
      else if scalar = 3.0 then S.mul (S.mul v v) v
      else if scalar = -2.0 then reciprocal (S.mul v v)
      else if scalar = 0.5 then S.sqrt v
      else if scalar = -0.5 then reciprocal (S.sqrt v)
      else if scalar = -1.0 then reciprocal v
      else S.exp (S.mul (S.const scalar) (S.log v))
  end
end

module Sub = struct
  type t = Bin.t

  let name = "Sub"
  let jsont = Bin.jsont ~name
  let operands = Bin.operands
  let map_operands = Bin.map_operands
  let pp pp_ref fmt t = Bin.pp ~op:"sub" pp_ref fmt t
  let output_shape = broadcast_output_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    module B = Binary (S)

    let pixel ~a_shape ~b_shape a b out =
      B.pixel ~combine:S.sub ~a_shape ~b_shape a b out
  end
end
