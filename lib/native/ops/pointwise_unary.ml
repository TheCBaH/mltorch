(* Unary pointwise ops split out of pointwise.ml see pointwise.ml (the facade every external reference to e.g. [Pointwise.Clamp]
   still resolves through). *)

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
    | None, None -> Err.fail (`Clamp Shape_error.Clamp.No_bounds)
    | _ -> Err.return x_shape

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
       [min > max] yields [max] everywhere exactly as the ATen kernel does.

       [apply] takes the value directly rather than loading it, so callers that
       already have a value in hand — [Hardsigmoid]/[Hardswish] clamping
       [x + 3], not a load — can reuse the same bound order without restating
       it. [pixel] is [apply] composed with the load, for callers (this op,
       [Hardtanh]) that clamp a tensor operand directly. *)
    let apply (p : params) v =
      let is_nan = Option.fold ~none:false ~some:Float.is_nan in
      if is_nan p.min || is_nan p.max then S.const Float.nan
      else
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

    let pixel (p : params) x (out : Semantics.position S.index Vec6.t) =
      apply p (S.load x out)
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

  let output_shape (x_shape : Vec6.shape) = Err.return x_shape

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

(* [aten.expand.default]: broadcasts [x] to [params.size] with no arithmetic
   -- a genuinely new op, not a decomposition (no existing node reads a
   smaller operand through a target shape it does not itself carry). [size]
   is already resolved (any [-1] substituted, right-aligned) by the two
   importers via [Aten_shape.resolve_expand_size]; see that module's comment
   for why the resolution needs [self]'s true ATen rank and this op's own
   [output_shape] does not. *)
module Expand = struct
  type params = { size : Vec6.shape }

  let params_jsont : params Jsont.t =
    Jsont.Object.map ~kind:"expand_params" (fun size -> { size })
    |> Jsont.Object.mem "size" Vec6.shape_jsont ~enc:(fun p -> p.size)
    |> Jsont.Object.finish

  let pp_params fmt (p : params) =
    Fmt.pf fmt "@[<hv>{size=%a}@]" Vec6.pp_shape p.size

  type t = { params : params; x : Tensor_ref.t }

  let name = "Expand"

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        let get k c = Json_util.req_field ms k c name in
        { params = get "params" params_jsont; x = get "x" Tensor_ref.jsont })
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
    Fmt.pf fmt "@[<hv 2>expand@ x=%a@ params=%a@]" pp_ref t.x pp_params t.params

  (* Per axis, [x_shape] must already be broadcast-compatible with the
     target: equal, or [x_shape] is 1 (the axis that fans out). Not delegated
     to the two importers -- reachable from a raw [Graph_builder] call or a
     JSON-decoded graph too, so it is the invariant [Compute]'s
     [broadcast_coord] read actually relies on staying true. Reuses
     [`Broadcast], the same row [Pointwise_binary.broadcast_output_shape]
     reports for the symmetric two-operand case -- one axis, one lhs/rhs
     mismatch, regardless of which side is fixed. *)
  let output_shape ~(x_shape : Vec6.shape) (p : params) =
    let open Err.Syntax in
    let+ () =
      Err.List.iter
        (fun axis ->
          let x = Vec6.get x_shape axis and target = Vec6.get p.size axis in
          if Dim.equal x target || Dim.equal x Dim.one then Err.return ()
          else
            Err.fail
              (`Broadcast Shape_error.Broadcast.{ axis; lhs = x; rhs = target }))
        Axis.all
    in
    p.size

  module Compute (S : Semantics.SEMANTICS) = struct
    (* [out] already ranges over [params.size] (the output shape [Eval_op]
       drives this functor with), so no target-shape argument is needed here
       -- only [x_shape] decides which axes broadcast, the same one-sided
       reduction [Pointwise_binary.Binary]'s own operand read performs. *)
    let pixel ~(x_shape : Vec6.shape) x
        (out : Semantics.position S.index Vec6.t) =
      S.load x
        (Pointwise_binary.broadcast_coord ~index_zero:S.index_zero x_shape out)
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

  let output_shape (x_shape : Vec6.shape) = Err.return x_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    (* [sqrt] is one of the genuinely primitive values in [SEMANTICS] — unlike
       relu it is not expressible from select/lt — so this is a direct call. A
       negative input yields NaN, as the float operation does. *)
    let pixel x (out : Semantics.position S.index Vec6.t) =
      S.sqrt (S.load x out)
  end
end

(* [_to_copy.default]'s value-domain effect for the corpus's three observed
   targets. Native's own compute domain is always float (semantics.ml), so a
   dtype cast changes no storage layout -- see graph_builder.ml's "op-output
   edges are F32" note -- only the VALUE each pixel carries: identity for the
   float target (already float), truncation toward zero for the long target
   (matches ATen's [static_cast<int64_t>]), and a genuine nonzero test for the
   bool target (matches ATen's [value != 0], not an overfit to the corpus's
   own all-zero operand). Anything outside this three-way domain (narrowing to
   fp16/bf16, int8/16/32, complex) has no corpus evidence and no Native value
   representation distinct from float, so both importers reject it rather than
   silently accepting it. *)
module To_copy = struct
  type target = Bool | Float | Long
  type t = { target : target; x : Tensor_ref.t }

  let name = "To_copy"

  let target_jsont : target Jsont.t =
    Jsont.map ~kind:"to_copy_target"
      ~dec:(fun s ->
        match s with
        | "bool" -> Bool
        | "float" -> Float
        | "long" -> Long
        | _ -> Jsont.Error.msgf Jsont.Meta.none "to_copy_target: unknown %S" s)
      ~enc:(function Bool -> "bool" | Float -> "float" | Long -> "long")
      Jsont.string

  let pp_target fmt = function
    | Bool -> Fmt.string fmt "bool"
    | Float -> Fmt.string fmt "float"
    | Long -> Fmt.string fmt "long"

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        let get k c = Json_util.req_field ms k c name in
        { target = get "target" target_jsont; x = get "x" Tensor_ref.jsont })
      ~enc:(fun t ->
        Json_util.jobj
          [
            ("target", Json_util.enc target_jsont t.target);
            ("x", Json_util.enc Tensor_ref.jsont t.x);
          ])
      Jsont.json

  let operands (t : t) = [ t.x ]
  let map_operands f (t : t) = { t with x = f t.x }

  let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>to_copy@ x=%a@ target=%a@]" pp_ref t.x pp_target
      t.target

  let output_shape (x_shape : Vec6.shape) = Err.return x_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    let pixel (target : target) x (out : Semantics.position S.index Vec6.t) =
      let v = S.load x out in
      match target with
      | Float -> v
      | Long -> S.trunc v
      | Bool ->
          S.select
            (S.lt (S.const 0.) v)
            (S.const 1.)
            (S.select (S.lt v (S.const 0.)) (S.const 1.) (S.const 0.))
  end
end
