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

module Clamp = struct
  (* [aten.clamp.default]: both bounds are `Scalar?`, and at least one must be
     given — ATen's meta function rejects the both-absent spelling rather than
     treating it as the identity, so [output_shape] does too. *)
  type params = { min : float option; max : float option }
  type t = { params : params; x : Tensor_ref.t }

  let name = "Clamp"

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"clamp_params" (fun min max -> { min; max })
    |> Jsont.Object.opt_mem "min" Json_util.f32_jsont ~enc:(fun p -> p.min)
    |> Jsont.Object.opt_mem "max" Json_util.f32_jsont ~enc:(fun p -> p.max)
    |> Jsont.Object.finish

  let pp_bound = Fmt.option ~none:(Fmt.any "none") Fmt.float

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{min=%a;@ max=%a}@]" pp_bound p.min pp_bound p.max

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        {
          params = Json_util.req_field ms "params" params_jsont name;
          x = Json_util.req_field ms "x" Tensor_ref.jsont name;
        })
      ~enc:(fun t ->
        Json_util.jobj
          [
            ("params", Json_util.enc params_jsont t.params);
            ("x", Json_util.enc Tensor_ref.jsont t.x);
          ])
      Jsont.json

  let operands (t : t) = [ t.x ]
  let map_operands f (t : t) = { t with x = f t.x }

  let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>clamp@ x=%a@ params=%a@]" pp_ref t.x pp_params t.params

  (* Shape-preserving, but this is also where the both-absent configuration is
     rejected: [Graph_shape] is the only per-op validation channel the builder
     consults, so returning the error here means the node can never be built.
     [Convolution.Same_padding_requires_stride_one] is the same pattern — a pure
     parameter check reported as a shape error. *)
  let output_shape (p : params) (x_shape : Vec6.shape) =
    match (p.min, p.max) with
    | None, None -> Core.fail (`Clamp Shape_error.Clamp.No_bounds)
    | _ -> Core.return x_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    (* ATen's public path is [TORCH_IMPL_FUNC(clamp_out)]: it tests each scalar
       bound for NaN FIRST and fills the whole result with NaN if either is,
       only then dispatching to the kernel's [min(max(a, lo), hi)]. Both halves
       are reproduced here.

       The NaN test reads only graph parameters, so it is loop-invariant, but
       [Eval_op] hands the whole pixel function one output coord at a time and
       so re-tests it per pixel. That is the same per-pixel granularity every op
       here pays (see .ai/native_inference_design.md's note on it); hoisting it
       would mean changing the calling convention for all ops, and the symbolic
       path folds the branch away anyway — it picks one arm while building the
       expression, so nothing reaches the staged program.

       Bound order is load-bearing: lower bound first, then upper, so a reversed
       [min > max] yields [max] everywhere exactly as the ATen kernel does. *)
    let pixel (p : params) x (out : Semantics.position S.index Vec6.t) =
      let is_nan = Option.fold ~none:false ~some:Float.is_nan in
      if is_nan p.min || is_nan p.max then S.const Float.nan
      else
        let v = S.load x out in
        (* max (v, lo) = if v < lo then lo else v *)
        let v =
          match p.min with
          | None -> v
          | Some lo -> S.select (S.lt v (S.const lo)) (S.const lo) v
        in
        (* min (v, hi) = if hi < v then hi else v *)
        match p.max with
        | None -> v
        | Some hi -> S.select (S.lt (S.const hi) v) (S.const hi) v
  end

  (* Walk config: the bound pair is ONE axis, not two independent ones, because
     the invalid (None, None) state must be unreachable by mutation — an axis
     per bound could step [min] to None while [max] already was. The candidate
     list therefore enumerates whole valid configurations, including reversed
     and NaN bounds (both of which ATen defines). *)
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = { shape : Walk_core.Shape.t; bounds : params }

    let candidates =
      [
        { min = Some 0.; max = Some 6. };
        { min = Some 0.; max = None };
        { min = None; max = Some 6. };
        { min = Some (-1.); max = Some 1. };
        { min = Some 1.; max = Some (-1.) };
        { min = Some Float.nan; max = Some 1. };
        { min = Some 0.; max = Some Float.nan };
      ]

    let initial =
      {
        shape = { Walk_core.Shape.n = 1; t = 1; d = 1; h = 4; w = 4; c = 3 };
        bounds = { min = Some 0.; max = Some 6. };
      }

    let cascade c = c
    let shape (c : cfg) = Walk_bridge.vec6 c.shape

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun c s -> { c with shape = s });
          field_axis "bounds" candidates (fun c b -> { c with bounds = b });
        ]

    let pp fmt (c : cfg) =
      Fmt.pf fmt "%a %a" Walk_core.Shape.pp c.shape pp_params c.bounds
  end
end

module Clone = struct
  (* [aten.clone.default] with no [memory_format] is the identity on a value.
     The native IR is SSA over values with no aliasing, so there is nothing to
     copy — but the node is kept rather than folded away at import so the PT2
     graph and the native graph stay one-to-one and the clone's own SSA name
     survives in the provenance sidecar. *)
  type t = { x : Tensor_ref.t }

  let name = "Clone"

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
    Fmt.pf fmt "@[<hv 2>clone@ x=%a@]" pp_ref t.x

  let output_shape (x_shape : Vec6.shape) = Core.return x_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    let pixel x (out : Semantics.position S.index Vec6.t) = S.load x out
  end

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

module Hardtanh = struct
  (* [aten.hardtanh.default] is clamp with both bounds required — schema
     `hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1)`. It keeps its
     own constructor, JSON tag, builder function and bridge arm because it is a
     distinct ATen target with a distinct parameter contract (two mandatory
     Scalars, not two optionals), and delegates its arithmetic to [Clamp] —
     the same shape as [Conv2d_padding] delegating to [Conv2d]. *)
  type params = { min_val : float; max_val : float }
  type t = { params : params; x : Tensor_ref.t }

  let name = "Hardtanh"

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"hardtanh_params" (fun min_val max_val ->
        { min_val; max_val })
    |> Jsont.Object.mem "min_val" Json_util.f32_jsont ~enc:(fun p -> p.min_val)
    |> Jsont.Object.mem "max_val" Json_util.f32_jsont ~enc:(fun p -> p.max_val)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{min_val=%a;@ max_val=%a}@]" Fmt.float p.min_val Fmt.float
      p.max_val

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        {
          params = Json_util.req_field ms "params" params_jsont name;
          x = Json_util.req_field ms "x" Tensor_ref.jsont name;
        })
      ~enc:(fun t ->
        Json_util.jobj
          [
            ("params", Json_util.enc params_jsont t.params);
            ("x", Json_util.enc Tensor_ref.jsont t.x);
          ])
      Jsont.json

  let operands (t : t) = [ t.x ]
  let map_operands f (t : t) = { t with x = f t.x }

  let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>hardtanh@ x=%a@ params=%a@]" pp_ref t.x pp_params
      t.params

  let output_shape (x_shape : Vec6.shape) = Core.return x_shape

  (* Both bounds are always present, so the [Clamp] params this lowers to can
     never be the rejected (None, None) pair. *)
  let clamp_params (p : params) : Clamp.params =
    { min = Some p.min_val; max = Some p.max_val }

  module Compute (S : Semantics.SEMANTICS) = struct
    module C = Clamp.Compute (S)

    let pixel (p : params) x out = C.pixel (clamp_params p) x out
  end

  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = { shape : Walk_core.Shape.t; bounds : params }

    (* Atomic bound pair, as in [Clamp.Walk]: the schema default, MobileNet-v2's
       relu6 window, and a reversed pair. *)
    let candidates =
      [
        { min_val = -1.; max_val = 1. };
        { min_val = 0.; max_val = 6. };
        { min_val = 1.; max_val = -1. };
      ]

    let initial =
      {
        shape = { Walk_core.Shape.n = 1; t = 1; d = 1; h = 4; w = 4; c = 3 };
        bounds = { min_val = 0.; max_val = 6. };
      }

    let cascade c = c
    let shape (c : cfg) = Walk_bridge.vec6 c.shape

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun c s -> { c with shape = s });
          field_axis "bounds" candidates (fun c b -> { c with bounds = b });
        ]

    let pp fmt (c : cfg) =
      Fmt.pf fmt "%a %a" Walk_core.Shape.pp c.shape pp_params c.bounds
  end
end

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

module Sqrt = struct
  type t = { x : Tensor_ref.t }

  let name = "Sqrt"

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
    Fmt.pf fmt "@[<hv 2>sqrt@ x=%a@]" pp_ref t.x

  let output_shape (x_shape : Vec6.shape) = Core.return x_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    (* [sqrt] is one of the genuinely primitive values in [SEMANTICS] — unlike
       relu it is not expressible from select/lt — so this is a direct call. A
       negative input yields NaN, as the float operation does. *)
    let pixel x (out : Semantics.position S.index Vec6.t) =
      S.sqrt (S.load x out)
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
  let output_shape (x_shape : Vec6.shape) = Core.return x_shape

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
  let output_shape (x_shape : Vec6.shape) = Core.return x_shape

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
