(* Activation pointwise ops split out of pointwise.ml see pointwise.ml (the facade). *)

open Pointwise_unary

module Hardsigmoid = struct
  (* [aten.hardsigmoid.default]: schema `hardsigmoid(Tensor self) -> Tensor`,
     no parameters — the bounds are the ATen kernel's own constants, not a
     user-supplied contract, so unlike [Hardtanh] there is no [params]. *)
  type t = { x : Tensor_ref.t }

  let name = "Hardsigmoid"

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
    Fmt.pf fmt "@[<hv 2>hardsigmoid@ x=%a@]" pp_ref t.x

  let output_shape (x_shape : Vec6.shape) = Err.return x_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    (* ATen (Activation.cpp:552-568): hardsigmoid(x) = min(max(x+3,0),6) / 6 —
       divides by 6, does not multiply by 1/6 (op5-impl F7). Reuses [Clamp]'s
       bound order via [apply] rather than restating min/max/select. *)
    let hard_clamp = { Clamp.min = Some 0.; max = Some 6. }

    let pixel x (out : Semantics.position S.index Vec6.t) =
      let module C = Clamp.Compute (S) in
      S.div
        (C.apply hard_clamp (S.add (S.load x out) (S.const 3.)))
        (S.const 6.)
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

module Hardswish = struct
  (* [aten.hardswish.default]: schema `hardswish(Tensor self) -> Tensor`, no
     parameters, same reason as [Hardsigmoid]. *)
  type t = { x : Tensor_ref.t }

  let name = "Hardswish"

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
    Fmt.pf fmt "@[<hv 2>hardswish@ x=%a@]" pp_ref t.x

  let output_shape (x_shape : Vec6.shape) = Err.return x_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    (* ATen (Activation.cpp:770-786): hardswish(x) = x * min(max(x+3,0),6) / 6,
       left-associated — the multiply precedes the divide (op5-impl F7). That
       order, not just the formula, is what distinguishes this from
       [x * hardsigmoid(x)] under f32 rounding: this engine materializes every
       node's output to f32, so [(x * c) / 6] and [x * (c / 6)] round
       differently. *)
    let hard_clamp = { Clamp.min = Some 0.; max = Some 6. }

    let pixel x (out : Semantics.position S.index Vec6.t) =
      let module C = Clamp.Compute (S) in
      let v = S.load x out in
      S.div (S.mul v (C.apply hard_clamp (S.add v (S.const 3.)))) (S.const 6.)
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

  let output_shape (x_shape : Vec6.shape) = Err.return x_shape

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
  let output_shape (x_shape : Vec6.shape) = Err.return x_shape

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

module Sigmoid = struct
  (* [aten.sigmoid.default]: schema `sigmoid(Tensor self) -> Tensor`, no
     parameters. sigmoid(x) = 1 / (1 + exp(-x)) -- the same denominator as
     [Silu] with numerator 1 instead of x. *)
  type t = { x : Tensor_ref.t }

  let name = "Sigmoid"

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
    Fmt.pf fmt "@[<hv 2>sigmoid@ x=%a@]" pp_ref t.x

  let output_shape (x_shape : Vec6.shape) = Err.return x_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    let pixel x (out : Semantics.position S.index Vec6.t) =
      let v = S.load x out in
      S.div (S.const 1.) (S.add (S.const 1.) (S.exp (S.sub (S.const 0.) v)))
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

module Silu = struct
  (* [aten.silu.default]: schema `silu(Tensor self) -> Tensor`, no parameters. *)
  type t = { x : Tensor_ref.t }

  let name = "Silu"

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
    Fmt.pf fmt "@[<hv 2>silu@ x=%a@]" pp_ref t.x

  let output_shape (x_shape : Vec6.shape) = Err.return x_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    (* ATen (Activation.cpp:1139): silu(x) = x / (1 + exp(-x)). The first
       [S.exp] consumer in the engine (op5-impl F4). *)
    let pixel x (out : Semantics.position S.index Vec6.t) =
      let v = S.load x out in
      S.div v (S.add (S.const 1.) (S.exp (S.sub (S.const 0.) v)))
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

module Gelu = struct
  (* [aten.gelu.default]: schema `gelu(Tensor self, *, str approximate="none")
     -> Tensor`. A closed variant, not a string: ATen's `approximate` argument
     has two spellings and this engine implements both, so the payload names
     them rather than carrying the ATen string through (same reasoning as
     [Pad.mode]). Anything other than "none"/"tanh" is rejected at the import
     boundary (Op_bridge/Native_interp) and never reaches this type. *)
  type approximate =
    | Exact  (** `approximate="none"`: the exact, erf-based form *)
    | Tanh  (** `approximate="tanh"`: the tanh-based approximation *)

  type t = { x : Tensor_ref.t; approximate : approximate }

  let name = "Gelu"

  let pp_approximate fmt = function
    | Exact -> Fmt.string fmt "none"
    | Tanh -> Fmt.string fmt "tanh"

  (* [{"none":null}] / [{"tanh":null}], the same single-key union
     [Pad.mode_jsont] uses -- tagged with ATen's own strings so a decode error
     or a raw dump of a native graph JSON reads directly against the PT2
     schema. *)
  let approximate_jsont : approximate Jsont.t =
    Jsont.map ~kind:(name ^ ".approximate")
      ~dec:
        (Json_util.union ~kind:(name ^ ".approximate")
           [ ("none", fun _ -> Exact); ("tanh", fun _ -> Tanh) ])
      ~enc:(function
        | Exact -> Json_util.single ~case:"none" Json_util.jnull
        | Tanh -> Json_util.single ~case:"tanh" Json_util.jnull)
      Jsont.json

  let jsont : t Jsont.t =
    Jsont.map ~kind:name
      ~dec:(fun json ->
        let ms = Json_util.req_obj json name in
        {
          x = Json_util.req_field ms "x" Tensor_ref.jsont name;
          approximate =
            Json_util.req_field ms "approximate" approximate_jsont name;
        })
      ~enc:(fun t ->
        Json_util.jobj
          [
            ("x", Json_util.enc Tensor_ref.jsont t.x);
            ("approximate", Json_util.enc approximate_jsont t.approximate);
          ])
      Jsont.json

  let operands (t : t) = [ t.x ]
  let map_operands f (t : t) = { t with x = f t.x }

  let pp (pp_ref : Tensor_ref.t Fmt.t) fmt (t : t) =
    Fmt.pf fmt "@[<hv 2>gelu@ x=%a@ approximate=%a@]" pp_ref t.x pp_approximate
      t.approximate

  let output_shape (x_shape : Vec6.shape) = Err.return x_shape

  module Compute (S : Semantics.SEMANTICS) = struct
    (* "none": gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2))), the exact form.

       "tanh": gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))),
       PyTorch's tanh approximation. [tanh] itself is select/exp-expressible
       (tanh y = 2*sigmoid(2y) - 1), so it costs no new SEMANTICS primitive,
       Expr.Value constructor, or eval/pp arm -- same reasoning as relu/max/min
       in semantics.ml. Written as [Sigmoid]'s own stable form, 1/(1+exp(-2y)):
       the naive (exp(2y)-1)/(exp(2y)+1) overflows to inf/inf = nan for large
       positive [y] (observed: x=10000 in the activation fixture), where the
       1/(1+exp(-2y)) form only ever divides a finite numerator by a sum that
       saturates to 0 or +inf, never producing an indeterminate ratio. *)
    let pixel (approximate : approximate) x
        (out : Semantics.position S.index Vec6.t) =
      let v = S.load x out in
      match approximate with
      | Exact ->
          S.mul
            (S.mul (S.const 0.5) v)
            (S.add (S.const 1.) (S.erf (S.div v (S.sqrt (S.const 2.)))))
      | Tanh ->
          let c0 =
            0.7978845608028654
            (* sqrt(2/pi) *)
          in
          let cubic = S.mul v (S.mul v v) in
          let inner =
            S.mul (S.const c0) (S.add v (S.mul (S.const 0.044715) cubic))
          in
          let sigmoid_2y =
            S.div (S.const 1.)
              (S.add (S.const 1.)
                 (S.exp (S.sub (S.const 0.) (S.mul (S.const 2.) inner))))
          in
          let tanh_inner = S.sub (S.mul (S.const 2.) sigmoid_2y) (S.const 1.) in
          S.mul (S.mul (S.const 0.5) v) (S.add (S.const 1.) tanh_inner)
  end

  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = { shape : Walk_core.Shape.t; approximate : approximate }

    let candidates = [ Exact; Tanh ]

    let initial =
      {
        shape = { Walk_core.Shape.n = 1; t = 1; d = 1; h = 4; w = 4; c = 3 };
        approximate = Exact;
      }

    let cascade c = c
    let shape (c : cfg) = Walk_bridge.vec6 c.shape

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun c s -> { c with shape = s });
          field_axis "approximate" candidates (fun c a ->
              { c with approximate = a });
        ]

    let pp fmt (c : cfg) =
      Fmt.pf fmt "%a %a" Walk_core.Shape.pp c.shape pp_approximate c.approximate
  end
end

(* A binary elementwise op: read each operand at the output coord reduced against
   that operand's own shape ([broadcast_coord]), so an extent-1 axis fans out
   without ever handing [load] an out-of-bounds index. [combine] is the scalar op
   (S.add, S.mul, …). *)
