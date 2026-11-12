(* The random-walk config spaces of the ops in [Pointwise_unary], kept beside the
   [*_nwalk] files that assemble them into walk subjects. Each reads its op's
   shape rules from the op module and adds nothing to the op itself. *)

module Abs = struct
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

module Clamp = struct
  open Pointwise_unary.Clamp

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

module Clone = struct
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

module To_copy = struct
  (* Shape-only, matching [Clone.Walk]'s own shape: [target] and the operand's
     format are NOT config-space axes here, since each walk file fixes both
     (a distinct (target, source format) pair is a distinct walk target, e.g.
     [to_copy_float_i64]) rather than drawing them randomly -- the corpus
     acceptance pattern each one exercises is a specific pairing, not an
     arbitrary combination. *)
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
