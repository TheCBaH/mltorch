(* The random-walk config spaces of the ops in [Pointwise_activation], kept beside the
   [*_nwalk] files that assemble them into walk subjects. Each reads its op's
   shape rules from the op module and adds nothing to the op itself. *)

module Hardsigmoid = struct
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = { shape : Walk_core.Shape.t }

    let initial =
      { shape = { Walk_core.Shape.n = 1; t = 1; d = 1; h = 4; w = 4; c = 3 } }

    let cascade c = c
    let shape (c : cfg) = Walk_shape.vec6 c.shape

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
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = { shape : Walk_core.Shape.t }

    let initial =
      { shape = { Walk_core.Shape.n = 1; t = 1; d = 1; h = 4; w = 4; c = 3 } }

    let cascade c = c
    let shape (c : cfg) = Walk_shape.vec6 c.shape

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
  open Pointwise_activation.Hardtanh

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
    let shape (c : cfg) = Walk_shape.vec6 c.shape

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

module Leaky_relu = struct
  open Pointwise_activation.Leaky_relu

  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = { shape : Walk_core.Shape.t; params : params }

    let initial =
      {
        shape = { Walk_core.Shape.n = 1; t = 1; d = 1; h = 4; w = 4; c = 3 };
        params = { negative_slope = 0.01 };
      }

    let cascade c = c
    let shape (c : cfg) = Walk_shape.vec6 c.shape

    let axes =
      Walk_core.Walk.
        [
          shape_axis "input" L.limits
            ~get:(fun c -> c.shape)
            ~set:(fun c s -> { c with shape = s });
          field_axis "negative_slope" [ 0.; 0.01; 0.2 ] (fun c negative_slope ->
              { c with params = { negative_slope } });
        ]

    let pp fmt (c : cfg) =
      Fmt.pf fmt "%a %a" Walk_core.Shape.pp c.shape pp_params c.params
  end
end

module Relu = struct
  (* This op's own random-walk config space: a single tensor of any shape (relu
     imposes no constraint, so [cascade] is identity). A functor over the global
     Limits; the shape is one compound entry mutated within the limits' budget.
     Kept apart so it can diverge from any other backend's walk. *)
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = { shape : Walk_core.Shape.t }

    let initial =
      { shape = { Walk_core.Shape.n = 1; t = 1; d = 1; h = 4; w = 4; c = 3 } }

    let cascade c = c
    let shape (c : cfg) = Walk_shape.vec6 c.shape

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
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = { shape : Walk_core.Shape.t }

    let initial =
      { shape = { Walk_core.Shape.n = 1; t = 1; d = 1; h = 4; w = 4; c = 3 } }

    let cascade c = c
    let shape (c : cfg) = Walk_shape.vec6 c.shape

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

module Tanh = struct
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = { shape : Walk_core.Shape.t }

    let initial =
      { shape = { Walk_core.Shape.n = 1; t = 1; d = 1; h = 4; w = 4; c = 3 } }

    let cascade c = c
    let shape (c : cfg) = Walk_shape.vec6 c.shape

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
  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = { shape : Walk_core.Shape.t }

    let initial =
      { shape = { Walk_core.Shape.n = 1; t = 1; d = 1; h = 4; w = 4; c = 3 } }

    let cascade c = c
    let shape (c : cfg) = Walk_shape.vec6 c.shape

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
  open Pointwise_activation.Gelu

  module Walk (L : Walk_core.Limits.S) = struct
    type cfg = { shape : Walk_core.Shape.t; approximate : approximate }

    let candidates = [ Exact; Tanh ]

    let initial =
      {
        shape = { Walk_core.Shape.n = 1; t = 1; d = 1; h = 4; w = 4; c = 3 };
        approximate = Exact;
      }

    let cascade c = c
    let shape (c : cfg) = Walk_shape.vec6 c.shape

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
